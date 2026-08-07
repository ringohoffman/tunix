# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma4 model parameters.

This provides a mapping from upstream ORBAX_FLAX NESTED checkpoints [1] to our
Tunix NNX implementation.

[1] https://github.com/google-deepmind/gemma
"""

from __future__ import annotations

import collections
from collections.abc import Callable
import contextlib
import dataclasses
import itertools
import os
import time
from typing import Any, TypeVar

from absl import logging
from etils import epath
import flax
from flax import nnx
import flax.typing
import jax
from jax import numpy as jnp
from jax._src.mesh import use_abstract_mesh
from orbax import checkpoint as ocp
import sentencepiece as spm
from tunix.models.gemma4 import classification as gemma4_classification
from tunix.models.gemma4 import model as gemma4_model
from tunix.sft import checkpoint_manager

# Pretrained
GEMMA4_E2B_PT = 'gs://gemma-data/checkpoints/gemma4-e2b-pt'
GEMMA4_E4B_PT = 'gs://gemma-data/checkpoints/gemma4-e4b-pt'
# Instruction Tuned
GEMMA4_E2B_IT = 'gs://gemma-data/checkpoints/gemma4-e2b-it'
GEMMA4_E4B_IT = 'gs://gemma-data/checkpoints/gemma4-e4b-it'
# Tokenizer
GEMMA4_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma4.model'


def _mesh_context(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
) -> contextlib.AbstractContextManager[Any]:
  if isinstance(mesh, jax.sharding.Mesh):
    return jax.set_mesh(mesh)
  if isinstance(mesh, jax.sharding.AbstractMesh):
    return use_abstract_mesh(mesh)
  return contextlib.nullcontext()


def create_tokenizer(
    path: str = GEMMA4_TOKENIZER,
) -> spm.SentencePieceProcessor:
  """Load the Gemma 4 SentencePiece tokenizer.

  Args:
    path: Path to the SentencePiece model file.

  Returns:
    A loaded SentencePieceProcessor instance.
  """
  spm_processor = spm.SentencePieceProcessor()
  model_proto = epath.Path(path).read_bytes()
  spm_processor.LoadFromSerializedProto(model_proto)
  return spm_processor


class _ShapeTracer:
  """Traces .T and __getitem__ to compute inverse PartitionSpecs for DMA.

  When loading a checkpoint with sharded DMA, Orbax needs to know what
  PartitionSpec each *upstream* tensor should have. But our downstream model
  has different keys and shapes (e.g. gating_einsum[0].T → gate_proj.kernel).

  _ShapeTracer records the operations applied by map_from_upstream_checkpoint
  so we can invert them: given a downstream PartitionSpec, compute the
  upstream PartitionSpec that produces it after the recorded transforms.
  """

  __slots__ = ('key', 'shape', '_transposed', '_slice_idx', 'perm')

  def __init__(
      self,
      key: flax.typing.PathParts | tuple[flax.typing.PathParts, ...],
      shape: tuple[int, ...],
      transposed: bool = False,
      slice_idx: int | None = None,
      perm: tuple[int, ...] | None = None,
  ) -> None:
    self.key = key
    self.shape = shape
    self._transposed = transposed
    self._slice_idx = slice_idx
    self.perm = perm

  @property
  def T(self) -> _ShapeTracer:
    return _ShapeTracer(
        self.key,
        self.shape[::-1],
        not self._transposed,
        self._slice_idx,
        self.perm,
    )

  def transpose(self, axes: tuple[int, ...]) -> _ShapeTracer:
    new_shape = tuple(self.shape[i] for i in axes)
    return _ShapeTracer(
        self.key,
        new_shape,
        self._transposed,
        self._slice_idx,
        axes,
    )

  def __getitem__(self, idx: int | slice) -> _ShapeTracer:
    if isinstance(idx, int):
      return _ShapeTracer(
          self.key,
          self.shape[1:],
          self._transposed,
          idx,
          self.perm,
      )
    return self

  def invert_spec(
      self,
      spec: jax.sharding.PartitionSpec,
  ) -> jax.sharding.PartitionSpec:
    """Given a downstream PartitionSpec, compute the upstream one."""
    s = tuple(spec)
    if self.perm is not None:
      inv_perm = [0] * len(self.perm)
      for i, p_idx in enumerate(self.perm):
        inv_perm[p_idx] = i
      s = tuple(s[i] if i < len(s) else None for i in inv_perm)
    elif self._transposed:
      s = s[::-1]
    if self._slice_idx is not None:
      s = (None,) + s
    return jax.sharding.PartitionSpec(*s)


LeafT = TypeVar('LeafT', jax.Array, _ShapeTracer)


def _stack_layers_for_scan(
    params: flax.typing.PyTree[jax.Array],
    num_layers: int,
    pattern_len: int,
    frac_shared_layers: float,
) -> flax.typing.PyTree[jax.Array]:
  """Restructure per-layer params into scan_groups/sub_layers with stacking.

  When use_scan_layers is True, the model uses vmapped scan groups instead
  of individual layer modules. This function takes the flat per-layer
  checkpoint layout (layers/0..N) and reorganizes it into the scan layout
  (scan_groups/sub_layers/0..pattern_len) with an extra leading axis.

  Args:
    params: Nested parameter dict from map_from_upstream_checkpoint.
    num_layers: Total number of layers in the model.
    pattern_len: Number of sub-layers per scan group.
    frac_shared_layers: Fraction of shared layers in Gemma 4.

  Returns:
    Parameter dict with layers restructured into scan groups.
  """
  num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)
  num_shared_layers = num_layers - num_unshared_layers
  num_unshared_groups = num_unshared_layers // pattern_len
  num_shared_groups = num_shared_layers // pattern_len

  flat = flax.traverse_util.flatten_dict(params)
  new_flat: flax.typing.FlatPyTree[jax.Array] = {}
  collector_group_count: dict[flax.typing.PathParts, int] = {}
  collector: dict[flax.typing.PathParts, dict[int, jax.Array]] = (
      collections.defaultdict(dict)
  )

  for path, param in flat.items():
    root = path[0]
    assert isinstance(root, str)
    if len(path) >= 1 and (root == 'layers' or root.startswith('layer_')):
      if root == 'layers':
        layer_idx, *param_path = path[1:]
        assert isinstance(layer_idx, int)
        param_path = tuple(param_path)
      else:
        _, layer_idx_str = root.split('_')
        layer_idx = int(layer_idx_str)
        param_path = path[1:]

      if layer_idx < num_unshared_layers:
        sub_layer_idx = layer_idx % pattern_len
        group_idx = layer_idx // pattern_len
        group_name = (
            'unshared_scan_groups' if frac_shared_layers > 0 else 'scan_groups'
        )
        target_path: flax.typing.PathParts = (
            group_name,
            'sub_layers',
            sub_layer_idx,
        ) + param_path
        collector[target_path][group_idx] = param
        collector_group_count[target_path] = num_unshared_groups
      else:
        rel_idx = layer_idx - num_unshared_layers
        sub_layer_idx = rel_idx % pattern_len
        group_idx = rel_idx // pattern_len
        target_path: flax.typing.PathParts = (
            'shared_scan_groups',
            'sub_layers',
            sub_layer_idx,
        ) + param_path
        collector[target_path][group_idx] = param
        collector_group_count[target_path] = num_shared_groups
    else:
      new_flat[path] = param

  for target_path, slices in collector.items():
    g_count = collector_group_count[target_path]
    sorted_slices = [slices[i] for i in range(g_count)]
    new_flat[target_path] = jnp.stack(sorted_slices, axis=0)

  return flax.traverse_util.unflatten_dict(new_flat)


def _build_sharded_restore_target(
    checkpoint_path: str,
    model_state: nnx.State[
        flax.typing.PathParts, nnx.Variable[jax.ShapeDtypeStruct]
    ],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    model_config: gemma4_model.ModelConfig,
) -> tuple[flax.typing.PyTree[jax.ShapeDtypeStruct], ocp.PyTreeCheckpointer]:
  """Build a sharded restore target for direct-to-device DMA loading.

  Traces map_from_upstream_checkpoint with _ShapeTracer objects to determine
  what PartitionSpec each upstream checkpoint tensor needs so that, after
  applying the value transforms (transpose, slice), the result has the
  correct downstream sharding.

  Args:
    checkpoint_path: Path to the Orbax checkpoint.
    model_state: Abstract NNX model state (from nnx.eval_shape).
    mesh: JAX sharding mesh.
    model_config: Gemma4 model configuration.

  Returns:
    (upstream_target, checkpointer) — the target tree of ShapeDtypeStructs
    with computed shardings, and the checkpointer instance (reused for
    the subsequent restore call).
  """
  ckptr = ocp.PyTreeCheckpointer()
  meta = ckptr.metadata(checkpoint_path)
  assert meta.item_metadata is not None
  item_tree = meta.item_metadata.tree
  assert flax.typing.is_pytree_of(item_tree, ocp.metadata.Metadata)
  flat_upstream = flax.traverse_util.flatten_dict(item_tree)

  mock_upstream = flax.traverse_util.unflatten_dict(
      {k: _ShapeTracer(k, v.shape) for k, v in flat_upstream.items()}
  )

  def _tracer_stack_kv(
      k: _ShapeTracer,
      v: _ShapeTracer,
  ) -> _ShapeTracer:
    return _ShapeTracer(
        (k.key, v.key),
        (2, *k.shape),
        k._transposed,
        k._slice_idx,
        k.perm,
    )

  flat_traced = flax.traverse_util.flatten_dict(
      map_from_upstream_checkpoint(
          mock_upstream,
          model_config=model_config,
          stack_kv=_tracer_stack_kv,
      )
  )

  with _mesh_context(mesh):
    shd_state = nnx.get_named_sharding(model_state, mesh)
    flat_shardings = flax.traverse_util.flatten_dict(
        nnx.to_pure_dict(shd_state)
    )

  pattern_len = (
      len(model_config.attention_pattern)
      if (
          model_config.attention_pattern is not None
          and model_config.use_scan_layers
      )
      else 0
  )

  upstream_target: flax.typing.FlatPyTree[jax.ShapeDtypeStruct] = {}
  for downstream_key, tracer in flat_traced.items():
    if (
        pattern_len > 0
        and len(downstream_key) >= 2
        and downstream_key[0] == 'layers'
    ):
      layer_idx = downstream_key[1]
      assert isinstance(layer_idx, int)
      param_path = downstream_key[2:]

      num_unshared_layers = int(
          model_config.num_layers
          - model_config.frac_shared_layers * model_config.num_layers
      )
      if layer_idx < num_unshared_layers:
        sub_layer_idx = layer_idx % pattern_len
        group_name = (
            'unshared_scan_groups'
            if model_config.frac_shared_layers > 0
            else 'scan_groups'
        )
      else:
        rel_idx = layer_idx - num_unshared_layers
        sub_layer_idx = rel_idx % pattern_len
        group_name = 'shared_scan_groups'

      scan_key = (group_name, 'sub_layers', sub_layer_idx) + param_path
      if scan_key not in flat_shardings:
        if (group_name,) + scan_key in flat_shardings:
          scan_key = (group_name,) + scan_key
        else:
          logging.info(
              'Skipping sharded restore target for pruned scan key: %s',
              '/'.join(str(p) for p in scan_key),
          )
          continue
      sharding = flat_shardings[scan_key]
      # The stacked param's spec has a leading None for the scan/vmap axis
      # (prepended by _init_scan_layers Phase 2). The checkpoint stores
      # per-layer (un-stacked) tensors without that axis, so strip it before
      # inverting to the upstream checkpoint spec.
      spec_axes = tuple(sharding.spec)
      if spec_axes and spec_axes[0] is None:
        sharding = jax.sharding.NamedSharding(
            sharding.mesh,
            jax.sharding.PartitionSpec(*spec_axes[1:]),
        )
    else:
      if downstream_key not in flat_shardings:
        logging.info(
            'Skipping sharded restore target for pruned key: %s',
            '/'.join(str(p) for p in downstream_key),
        )
        continue
      sharding = flat_shardings[downstream_key]

    upstream_spec = tracer.invert_spec(sharding.spec)
    if tracer.key and isinstance(tracer.key[0], tuple):
      # Dual-key tuple (e.g. kv_einsum stacked from separate k_einsum and
      # v_einsum keys)
      single_spec = (
          jax.sharding.PartitionSpec(*upstream_spec[1:])
          if len(upstream_spec) > 3 and upstream_spec[0] is None
          else upstream_spec
      )
      for matched_key in tracer.key:
        assert isinstance(matched_key, tuple)
        orig = flat_upstream[matched_key]
        upstream_target[matched_key] = jax.ShapeDtypeStruct(
            shape=orig.shape,
            dtype=orig.dtype,
            sharding=jax.sharding.NamedSharding(mesh, single_spec),
        )
    else:
      matched_key = tracer.key
      orig = flat_upstream[matched_key]
      upstream_target[matched_key] = jax.ShapeDtypeStruct(
          shape=orig.shape,
          dtype=orig.dtype,
          sharding=jax.sharding.NamedSharding(mesh, upstream_spec),
      )

  unflattened = flax.traverse_util.unflatten_dict(upstream_target)
  return unflattened, ckptr


def _try_restore_native_tunix(
    resolved_path: str,
    abs_model: gemma4_model.Gemma4,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    t0: float,
) -> bool:
  """Attempt to restore from a native Tunix CheckpointManager checkpoint.

  Native Tunix checkpoints live under a step directory (e.g.
  .../checkpoints/20000/model_params) and use Tunix NNX key layout directly.
  They are distinguished from upstream checkpoints by the absence of
  'token_embedder' and 'decoder' top-level keys.

  To handle checkpoints saved with or without scan layers regardless of the
  current model config, this function always restores into a flat
  ``use_scan_layers=False`` model first, then re-stacks into scan groups if
  the actual model config requires it.

  Returns True if the checkpoint was successfully restored, False if this
  is not a native Tunix checkpoint and the caller should fall through to
  the upstream mapping path.
  """
  step_parent = os.path.dirname(resolved_path.rstrip('/'))
  step_name = os.path.basename(step_parent)
  is_step_dir = (
      step_name.isdigit()
      and os.path.basename(resolved_path.rstrip('/')) == 'model_params'
  )
  if not is_step_dir:
    return False

  ckptr_meta = ocp.PyTreeCheckpointer().metadata(resolved_path)
  top_keys = (
      list(ckptr_meta.item_metadata.tree.keys())
      if ckptr_meta.item_metadata
      else []
  )
  is_native_tunix = (
      'token_embedder' not in top_keys and 'decoder' not in top_keys
  )
  if not is_native_tunix or mesh is None:
    return False

  from tunix.sft import checkpoint_manager as tunix_ckpt_mgr

  ckpt_root = os.path.dirname(step_parent)
  step_num = int(step_name)
  model_config = abs_model.config

  # Determine whether the checkpoint is flat (layers.N) or scanned
  # (scan_groups), and whether the model expects the same format.
  ckpt_has_layers = 'layers' in top_keys
  ckpt_has_scan = (
      'scan_groups' in top_keys or 'unshared_scan_groups' in top_keys
  )
  model_wants_scan = (
      model_config.attention_pattern is not None
      and model_config.use_scan_layers
  )

  # If the checkpoint format matches the model, restore directly.
  # Otherwise, build a temporary flat model and re-stack after restore.
  needs_restack = ckpt_has_layers and model_wants_scan
  needs_unstack = ckpt_has_scan and not model_wants_scan

  if needs_restack or needs_unstack:
    # Build a temporary abstract model with the OPPOSITE scan setting to
    # match the checkpoint's key layout.
    restore_config = dataclasses.replace(
        model_config,
        use_scan_layers=not model_config.use_scan_layers,
    )
    with nnx.use_eager_sharding(True), _mesh_context(mesh):
      restore_model = nnx.eval_shape(
          lambda: gemma4_model.Gemma4(restore_config, rngs=nnx.Rngs(0))
      )
    logging.info(
        'Checkpoint layout (%s) does not match model (scan=%s). '
        'Restoring with scan=%s first, then converting.',
        'layers.N' if ckpt_has_layers else 'scan_groups',
        model_wants_scan,
        restore_config.use_scan_layers,
    )
  else:
    restore_model = abs_model

  def _bind_mesh(x: Any) -> Any:
    if isinstance(x, jax.ShapeDtypeStruct) and isinstance(
        getattr(x, 'sharding', None), jax.sharding.NamedSharding
    ):
      return jax.ShapeDtypeStruct(
          x.shape,
          x.dtype,
          sharding=jax.sharding.NamedSharding(mesh, x.sharding.spec),
      )
    return x

  with _mesh_context(mesh):
    nnx.update(
        restore_model,
        jax.tree.map(_bind_mesh, nnx.state(restore_model)),
    )
    mgr = tunix_ckpt_mgr.CheckpointManager(ckpt_root)
    mgr.maybe_restore(restore_model, step=step_num)
    mgr.close()

  if needs_restack and model_config.attention_pattern is not None:
    # Convert flat layers.N → scan_groups by extracting the restored params,
    # stacking them, then updating the original abs_model.
    flat_params = nnx.to_pure_dict(nnx.state(restore_model))
    stacked = _stack_layers_for_scan(
        flat_params,
        model_config.num_layers,
        len(model_config.attention_pattern),
        model_config.frac_shared_layers,
    )
    # Prune stacked params to match the target model's state keys and update.
    model_state = nnx.state(abs_model)
    pruned = _prune_to_model_keys(stacked, model_state)
    _validate_param_shapes(pruned, model_state)
    with _mesh_context(mesh):
      shardings = nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh))
      typed = jax.tree_util.tree_map_with_path(
          lambda p, x, s: jnp.asarray(
              x, device=s, dtype=model_config.param_dtype
          ),
          pruned,
          shardings,
      )
    nnx.update(abs_model, typed)
  elif needs_unstack:
    # Convert scan_groups → flat layers.N (future-proofing).
    # For now, the restored model already has the right layout; just copy.
    nnx.update(abs_model, nnx.state(restore_model))
  elif restore_model is not abs_model:
    nnx.update(abs_model, nnx.state(restore_model))

  logging.info(
      'Restored native Tunix model from step %d via CheckpointManager in %.2fs'
      ' (restack=%s)',
      step_num,
      time.monotonic() - t0,
      needs_restack,
  )
  return True


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: gemma4_model.ModelConfig,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> gemma4_model.Gemma4:
  """Load a Gemma4 model from an Orbax checkpoint.

  Uses nnx.eval_shape to build an abstract model without allocating memory,
  then restores checkpoint parameters with sharded DMA loading when a mesh
  is provided.

  Args:
    checkpoint_path: Path to an Orbax checkpoint directory.
    model_config: Gemma4 model configuration.
    mesh: Optional JAX sharding mesh for distributed loading. When provided,
      enables direct-to-device DMA: each TPU worker reads only its required
      shard from GCS into local HBM.
    dtype: Parameter dtype (default: bfloat16).

  Returns:
    A Gemma4 model instance with loaded weights.
  """
  t0 = time.monotonic()
  _t_prev = t0

  def _log_phase(phase_name: str) -> None:
    nonlocal _t_prev
    now = time.monotonic()
    logging.info(
        '[TIMING] %s: %.1fs (cumulative: %.1fs)',
        phase_name,
        now - _t_prev,
        now - t0,
    )
    _t_prev = now

  # GCSFuse mount paths must be translated to gs:// URIs for TensorStore.
  checkpoint_path = checkpoint_manager.gcsfuse_to_gs_path(checkpoint_path)
  logging.info('Creating model from checkpoint path %s', checkpoint_path)

  clean_path = checkpoint_path.rstrip('/')
  if clean_path.endswith('/model_params'):
    resolved_path = clean_path
  elif (epath.Path(clean_path) / 'model_params').exists():
    # Tunix step dir (e.g. .../checkpoints/20000) -> append '/model_params'
    resolved_path = str(epath.Path(clean_path) / 'model_params')
  else:
    resolved_path = clean_path

  _log_phase('path_resolution')

  with (
      nnx.use_eager_sharding(True),
      _mesh_context(mesh),
  ):
    model_cls = (
        gemma4_classification.Gemma4ForClassification
        if isinstance(
            model_config, gemma4_classification.ClassificationModelConfig
        )
        else gemma4_model.Gemma4
    )
    abs_model = nnx.eval_shape(
        lambda: model_cls(model_config, rngs=nnx.Rngs(0))
    )
  model_state: nnx.State[
      flax.typing.PathParts, nnx.Variable[jax.ShapeDtypeStruct]
  ] = nnx.state(abs_model)

  _log_phase('eval_shape + model_state')

  if _try_restore_native_tunix(resolved_path, abs_model, mesh, t0):
    return abs_model

  _log_phase('_try_restore_native_tunix (skipped)')

  if mesh is not None:
    target, ckptr = _build_sharded_restore_target(
        resolved_path,
        model_state,
        mesh,
        model_config,
    )
    _log_phase('_build_sharded_restore_target')
    raw_params = ckptr.restore(
        resolved_path,
        target=target,
        partial_restore=True,
    )
    _log_phase('ckptr.restore (I/O)')
  else:
    raw_params = ocp.PyTreeCheckpointer().restore(resolved_path)
    _log_phase('PyTreeCheckpointer.restore (I/O)')

  mapped = map_from_upstream_checkpoint(
      raw_params,
      model_config=model_config,
      stack_kv=lambda k, v: jnp.stack([k, v], axis=0),
  )

  _log_phase('map_from_upstream_checkpoint')

  if (
      model_config.attention_pattern is not None
      and model_config.use_scan_layers
  ):
    mapped = _stack_layers_for_scan(
        mapped,
        model_config.num_layers,
        len(model_config.attention_pattern),
        model_config.frac_shared_layers,
    )
    _log_phase('_stack_layers_for_scan')

  pruned = _prune_to_model_keys(mapped, model_state)
  _validate_param_shapes(pruned, model_state)

  _log_phase('prune + validate')

  pure_state = nnx.to_pure_dict(model_state)

  if mesh is not None:
    with _mesh_context(mesh):
      shardings = nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh))
      _log_phase('get_named_sharding')
      # Fill missing leaves (e.g. vision weights absent from text-only ckpt)
      flat_pure = flax.traverse_util.flatten_dict(pure_state)
      flat_pruned = flax.traverse_util.flatten_dict(pruned)
      for k, v in flat_pure.items():
        if k not in flat_pruned:
          flat_pruned[k] = jnp.zeros(v.shape, dtype=dtype)
      complete_params = flax.traverse_util.unflatten_dict(flat_pruned)
      _log_phase('flatten + fill missing + unflatten')
      typed = jax.tree_util.tree_map_with_path(
          lambda p, x, s: jnp.asarray(x, device=s, dtype=dtype),
          complete_params,
          shardings,
      )
      _log_phase('jnp.asarray (dtype cast + shard placement)')
  else:
    typed = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=dtype),
        pruned,
    )
    _log_phase('jnp.asarray (dtype cast)')

  nnx.update(abs_model, typed)

  _log_phase('nnx.update (typed)')

  # partial_restore may leave ShapeDtypeStructs for unused keys (e.g. vision
  # weights in a text-only model). Replace them with zeros so subsequent
  # nnx.jit calls don't hit TraceContextErrors.
  def _materialize(x: jax.ShapeDtypeStruct | jax.Array) -> jax.Array:
    if isinstance(x, jax.ShapeDtypeStruct):
      return jnp.zeros(
          x.shape,
          dtype=x.dtype,
          device=getattr(x, 'sharding', None),
      )
    return x

  state: nnx.State[
      flax.typing.PathParts,
      nnx.Variable[jax.Array] | nnx.Variable[jax.ShapeDtypeStruct],
  ] = nnx.state(abs_model)
  nnx.update(abs_model, jax.tree_util.tree_map(_materialize, state))

  _log_phase('materialize ShapeDtypeStructs')

  logging.info(
      '[TIMING] create_model_from_checkpoint: %.1fs',
      time.monotonic() - t0,
  )
  return abs_model


def _validate_param_shapes(
    mapped_params: flax.typing.PyTree[LeafT],
    model_state: nnx.State[
        flax.typing.PathParts, nnx.Variable[jax.ShapeDtypeStruct]
    ],
) -> None:
  """Validate that mapped checkpoint params match expected model shapes.

  Args:
    mapped_params: Flattened parameter dict from the checkpoint.
    model_state: The model's NNX state (defines expected structure).

  Raises:
    ValueError: If the checkpoint is missing keys expected by the model or
      if any parameter shapes do not match.
  """
  flat_mapped = flax.traverse_util.flatten_dict(mapped_params)
  flat_model = flax.traverse_util.flatten_dict(nnx.to_pure_dict(model_state))

  mapped_keys = set(flat_mapped.keys())
  model_keys = set(flat_model.keys())

  missing_keys = model_keys - mapped_keys
  if missing_keys:
    raise ValueError(
        'Checkpoint is missing keys expected by the model: '
        f'{sorted(str(k) for k in missing_keys)}'
    )

  extra_keys = mapped_keys - model_keys
  if extra_keys:
    logging.warning(
        'Checkpoint contains extra keys not expected by the model: %s',
        sorted(str(k) for k in extra_keys),
    )

  mismatched: list[
      tuple[flax.typing.PathParts, tuple[int, ...], tuple[int, ...]]
  ] = []
  for key in mapped_keys & model_keys:
    mapped_val = flat_mapped[key]
    model_val = flat_model[key]
    if not hasattr(mapped_val, 'shape') or not hasattr(model_val, 'shape'):
      logging.warning(
          'Skipping shape check for non-array value at key %r '
          '(types: checkpoint=%s, model=%s)',
          key,
          type(mapped_val).__name__,
          type(model_val).__name__,
      )
      continue
    if mapped_val.shape != model_val.shape:
      mismatched.append((key, mapped_val.shape, model_val.shape))

  if mismatched:
    details = '\n'.join(
        f'  {k}: checkpoint={cs} vs model={ms}' for k, cs, ms in mismatched
    )
    raise ValueError(f'Shape mismatches (May need transpose):\n{details}')


def _prune_to_model_keys(
    params: flax.typing.PyTree[LeafT],
    model_state: nnx.State[
        flax.typing.PathParts, nnx.Variable[jax.ShapeDtypeStruct]
    ],
) -> flax.typing.PyTree[LeafT]:
  """Prune checkpoint params to only include keys present in the model.

  IT checkpoints contain multimodal params (audio_input_projection,
  mm_input_projection, etc.) that don't exist in the text-only Tunix model.
  These must be removed before tree_map_with_path, which requires matching
  pytree structures.

  Args:
    params: Mapped parameter dict (may contain extra keys).
    model_state: The model's NNX state (defines expected structure).

  Returns:
    A filtered copy of params with only model-expected keys.
  """
  model_dict = nnx.to_pure_dict(model_state)
  flat_params = flax.traverse_util.flatten_dict(params)
  flat_model = flax.traverse_util.flatten_dict(model_dict)

  pruned_keys = set(flat_params.keys()) - set(flat_model.keys())
  if pruned_keys:
    logging.info(
        'Pruning %d extra checkpoint keys not in model: %s',
        len(pruned_keys),
        sorted(str(k) for k in pruned_keys),
    )

  filtered: flax.typing.FlatPyTree[LeafT] = {
      k: v for k, v in flat_params.items() if k in flat_model
  }

  # Guard: if we matched almost nothing, the checkpoint is likely in the wrong
  # format (e.g. a tunix CheckpointManager checkpoint passed to
  # create_model_from_checkpoint, which expects upstream key layout).
  if flat_model and not filtered:
    raise ValueError(
        'Checkpoint has 0 keys matching the model '
        f'(checkpoint has {len(flat_params)} keys, model expects '
        f'{len(flat_model)} keys). This usually means the checkpoint is not '
        'in upstream format — tunix CheckpointManager checkpoints should be '
        'loaded via CheckpointManager.maybe_restore() instead of '
        'create_model_from_checkpoint().'
    )

  return flax.traverse_util.unflatten_dict(filtered)


def map_from_upstream_checkpoint(
    params: flax.typing.PyTree[LeafT],
    model_config: gemma4_model.ModelConfig | None = None,
    *,
    stack_kv: Callable[[LeafT, LeafT], LeafT] | None = None,
) -> flax.typing.PyTree[LeafT]:
  """Map from upstream Orbax NESTED checkpoint to Tunix NNX layout.

  Handles both key formats produced by Orbax checkpoints:

  **Semi-flat (Gemma 3 GCS pattern):**
    ('transformer/layer_0/attn/q_einsum', 'w')  → 2-tuple

  **Genuinely nested (ORBAX_FLAX NESTED format):**
    ('transformer', 'layer_0', 'attn', 'q_einsum', 'w')  → N-tuple

  Both are normalized to a uniform list of path components before remapping
  to the nested tuple keys expected by the Tunix NNX Gemma4 model.

  Unlike Gemma 3's mapper, this function does not take a ``text_only``
  parameter.  Multimodal-only keys are mapped through and pruned later
  by ``_prune_to_model_keys``.

  Args:
    params: Raw parameter dict restored from an Orbax checkpoint.

  Returns:
    A nested dict with keys matching the Tunix NNX Gemma4 model tree.
  """
  new_params: flax.typing.FlatPyTree[LeafT] = {}

  flat_params = flax.traverse_util.flatten_dict(params)
  raw_key_strings = set('/'.join(str(s) for s in k) for k in flat_params.keys())

  if any(
      isinstance(k, tuple)
      and len(k) >= 2
      and k[0]
      in (
          'layers',
          'scan_groups',
          'unshared_scan_groups',
          'shared_scan_groups',
      )
      for k in flat_params.keys()
  ):
    # Already in NNX layout (e.g. from Tunix CheckpointManager).
    # Orbax JSON metadata deserializes PyTree dictionary keys as strings ('0').
    # Normalize digit string path components to int so downstream consumers
    # (_build_sharded_restore_target, _stack_layers_for_scan) receive integer
    # layer indices matching NNX model state.
    normalized: flax.typing.FlatPyTree[LeafT] = {}
    for k, v in flat_params.items():
      norm_key = tuple(
          int(part) if isinstance(part, str) and part.isdigit() else part
          for part in k
      )
      normalized[norm_key] = v
    return flax.traverse_util.unflatten_dict(normalized)

  for key_path, value in flat_params.items():

    parts: list[str] = [
        str(segment)
        for segment in itertools.chain.from_iterable(
            (s.split('/') if isinstance(s, str) else [str(s)]) for s in key_path
        )
    ]

    if parts and parts[-1] == 'value':
      parts = parts[:-1]

    if parts and parts[0] in ('transformer', 'decoder'):
      parts = parts[1:]

    if parts and parts[0] in ('token_embedder', 'embedder'):
      parts[0] = 'embedder'
    if parts and parts[0] in ('decoder_norm', 'final_norm'):
      parts[0] = 'final_norm'

    if parts and (
        parts[0].startswith('layers_') or parts[0].startswith('layer_')
    ):
      raw_num = parts[0].removeprefix('layers_').removeprefix('layer_')
      if raw_num.isdigit():
        parts[0] = f'layer_{raw_num}'

    if not parts:
      logging.warning('Skipping empty key path: %r', key_path)
      continue

    *module_path, param_name = parts

    if module_path and module_path[0] == 'embedder':
      embed_root, *embed_sub = module_path
      if embed_sub and embed_sub[0] == 'per_layer_embeddings':
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif param_name in ('per_layer_embeddings', 'per_layer_input_embedding'):
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif param_name in ('embedding', 'input_embedding'):
        new_params[('embedder', 'input_embedding')] = value
      elif embed_sub:
        new_params[tuple(module_path + [param_name])] = value
      else:
        new_params[('embedder', param_name)] = value
      continue

    if module_path and module_path[0] == 'final_norm':
      new_params[('final_norm', param_name)] = value
      continue

    if not module_path:
      logging.warning('Unexpected bare param after transformer: %r', key_path)
      continue

    layer_segment, *layer_submodules = module_path

    if not layer_segment.startswith('layer_'):
      logging.info('Skipping non-layer module: %s', '/'.join(parts))
      continue

    layer_num = int(layer_segment.removeprefix('layer_'))
    layer_key = ('layers', layer_num)

    if len(module_path) == 1:
      leaf_name = (
          'skip_scale'
          if param_name in ('layer_scalar', 'skip_scale')
          else param_name
      )
      new_params[(*layer_key, leaf_name)] = value
      continue

    norm_submodules = {
        'pre_self_attention_norm': 'pre_attention_norm',
        'post_self_attention_norm': 'post_attention_norm',
        'self_attention': 'attn',
        'post_ffw1_norm': 'dense_post_ffw_norm',
        'pre_ffw2_norm': 'moe_pre_ffw_norm',
        'post_ffw2_norm': 'moe_post_ffw_norm',
    }
    module_path = [norm_submodules.get(p, p) for p in module_path]
    if 'attn' in module_path:
      a_idx = module_path.index('attn')
      if a_idx + 1 < len(module_path):
        sub = module_path[a_idx + 1]
        if sub == 'query':
          module_path[a_idx + 1] = 'q_einsum'
        elif sub == 'out':
          module_path[a_idx + 1] = 'attn_vec_einsum'
        elif sub == 'key':
          # Determine whether upstream 'key' maps to a separate k_einsum or a
          # shared kv_einsum. If the same layer also has a 'value' sibling, the
          # checkpoint stores K and V separately (→ k_einsum). Otherwise the
          # single 'key' tensor holds both K and V (→ kv_einsum, as used by
          # k_eq_v_global layers).
          orig_key_str = '/'.join(str(s) for s in key_path)
          prefix = orig_key_str.rsplit('self_attention', 1)[0]
          has_val = any(
              k.startswith(prefix)
              and ('/self_attention/value/' in k or '/attn/value/' in k)
              for k in raw_key_strings
          )
          module_path[a_idx + 1] = 'k_einsum' if has_val else 'kv_einsum'
        elif sub == 'value':
          module_path[a_idx + 1] = 'v_einsum'

    if (
        param_name == 'kernel'
        and module_path
        and (module_path[-1].endswith('_einsum') or module_path[-1] == 'attn')
    ):
      param_name = 'w'

    if 'mlp' in module_path or 'mlp2' in module_path:
      m_idx = (
          module_path.index('mlp2')
          if 'mlp2' in module_path
          else module_path.index('mlp')
      )
      if m_idx + 1 < len(module_path):
        sub = module_path[m_idx + 1]
        mlp_sub_map = {
            'wi_0': 'gate_proj',
            'wi_1': 'up_proj',
            'wo': 'down_proj',
        }
        if sub in mlp_sub_map:
          module_path[m_idx + 1] = mlp_sub_map[sub]

    # gating_einsum[0] → gate_proj, gating_einsum[1] → up_proj (transposed)
    if module_path[1:] in (['mlp', 'gating_einsum'], ['mlp2', 'gating_einsum']):
      if len(value.shape) == 4:
        # 4D → MoE expert-level tensor, not gate/up split
        new_params[(*layer_key, 'moe', 'gating_einsum')] = value
        continue
      if value.shape[0] != 2:
        raise ValueError(
            'Expected gating_einsum shape[0]=2 for gate/up split, got shape'
            f' {value.shape} for key {key_path}'
        )
      new_params[(*layer_key, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_key, 'mlp', 'up_proj', 'kernel')] = value[1].T
      continue

    if module_path[1:] in (['mlp', 'linear'], ['mlp2', 'linear']):
      if len(value.shape) == 3 and module_path[1] == 'mlp':
        new_params[(*layer_key, 'moe', 'linear')] = value
        continue
      new_params[(*layer_key, 'mlp', 'down_proj', 'kernel')] = value
      continue

    if len(module_path) >= 2 and module_path[1] == 'mlp':
      sub = (
          param_name
          if param_name in ('router_logits', 'router_scale', 'per_expert_scale')
          else module_path[-1]
      )
      if sub in ('router_logits', 'router_scale', 'per_expert_scale'):
        new_params[(*layer_key, 'moe', sub)] = value
        continue

    # Tunix uses underscore-prefixed norm names (_query_norm, _key_norm).
    if module_path[-1] in (
        'query_norm',
        '_query_norm',
        'key_norm',
        '_key_norm',
    ):
      target_norm = '_query_norm' if 'query' in module_path[-1] else '_key_norm'
      sub_path = (
          module_path[1:-1]
          if (
              module_path
              and (
                  str(module_path[0]).startswith('layer_')
                  or module_path[0] == 'layers'
              )
          )
          else module_path[:-1]
      )
      new_params[(*layer_key, *sub_path, target_norm, param_name)] = value
      continue

    new_params[(*layer_key, *module_path[1:], param_name)] = value

  # Linen layout is (embed, heads, head_dim); Tunix expects (heads, embed, head_dim).
  for k, v in list(new_params.items()):
    if len(k) >= 4 and k[2] == 'attn' and k[-1] == 'w':
      if (
          len(v.shape) == 3
          and model_config is not None
          and model_config.embed_dim > 0
          and v.shape[0] == model_config.embed_dim
      ):
        new_params[k] = v.transpose((1, 0, 2))

  # Global attention with k_eq_v_global shares weights: kv_einsum → k + v.
  # Local attention stacks separate k/v back: k + v → kv_einsum.
  if (
      model_config is not None
      and model_config.attention_pattern is not None
      and len(model_config.attention_pattern) > 0
  ):
    pattern_len = len(model_config.attention_pattern)
    for i in range(model_config.num_layers):
      is_global = (
          model_config.attention_pattern[i % pattern_len]
          == gemma4_model.AttentionType.GLOBAL
      ) and model_config.k_eq_v_global
      attn_key = ('layers', i, 'attn')
      if is_global:
        kv_w_key = (*attn_key, 'kv_einsum', 'w')
        if kv_w_key in new_params:
          kv_val = new_params.pop(kv_w_key)
          new_params[(*attn_key, 'k_einsum', 'w')] = kv_val
          new_params[(*attn_key, 'v_einsum', 'w')] = kv_val
      else:
        k_w_key = (*attn_key, 'k_einsum', 'w')
        v_w_key = (*attn_key, 'v_einsum', 'w')
        if k_w_key in new_params and v_w_key in new_params:
          k_val = new_params.pop(k_w_key)
          v_val = new_params.pop(v_w_key)
          if stack_kv is not None:
            new_params[(*attn_key, 'kv_einsum', 'w')] = stack_kv(k_val, v_val)

  return flax.traverse_util.unflatten_dict(new_params)
