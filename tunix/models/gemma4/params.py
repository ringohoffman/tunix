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

from collections.abc import Mapping
import itertools
import os
import time
from typing import Any

from absl import logging
from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
import sentencepiece as spm
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


def _stack_layers_for_scan(
    params: dict[str, Any],
    num_layers: int = 42,
    pattern_len: int = 6,
) -> dict[str, Any]:
  """Restructure per-layer params into scan_groups/sub_layers with stacking.

  When use_scan_layers is True, the model uses vmapped scan groups instead
  of individual layer modules. This function takes the flat per-layer
  checkpoint layout (layers/0..N) and reorganizes it into the scan layout
  (scan_groups/sub_layers/0..pattern_len) with an extra leading axis of
  size num_groups = num_layers // pattern_len.

  Args:
    params: Nested parameter dict from map_from_upstream_checkpoint.
    num_layers: Total number of layers in the model.
    pattern_len: Number of sub-layers per scan group.

  Returns:
    Parameter dict with layers restructured into scan groups.
  """
  num_groups = num_layers // pattern_len
  flat = flax.traverse_util.flatten_dict(params)
  new_flat: dict[tuple[Any, ...], Any] = {}
  collector: dict[tuple[Any, ...], dict[int, Any]] = {}

  for path, val in flat.items():
    if len(path) >= 1 and (
        path[0] == 'layers'
        or (isinstance(path[0], str) and path[0].startswith('layer_'))
    ):
      if path[0] == 'layers':
        layer_idx = int(path[1])
        param_path = path[2:]
      else:
        layer_idx = int(path[0].split('_')[1])
        param_path = path[1:]
      sub_layer_idx = layer_idx % pattern_len
      group_idx = layer_idx // pattern_len
      target_path = ('scan_groups', 'sub_layers', sub_layer_idx) + param_path
      collector.setdefault(target_path, {})[group_idx] = val
    else:
      new_flat[path] = val

  for target_path, slices in collector.items():
    sorted_slices = [slices[i] for i in range(num_groups)]
    if isinstance(sorted_slices[0], _ShapeTracer):
      new_flat[target_path] = _ShapeTracer(
          tuple(s.key for s in sorted_slices),
          (num_groups,) + sorted_slices[0].shape,
          sorted_slices[0]._transposed,
          sorted_slices[0]._slice_idx,
          sorted_slices[0].perm,
      )
    else:
      new_flat[target_path] = jnp.stack(sorted_slices, axis=0)

  return flax.traverse_util.unflatten_dict(new_flat)


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
      key: tuple[str, ...],
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


def _build_sharded_restore_target(
    checkpoint_path: str,
    model_state: Any,
    mesh: jax.sharding.Mesh,
    model_config: gemma4_model.ModelConfig,
) -> tuple[dict[str, Any], ocp.PyTreeCheckpointer]:
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
  item_tree = meta.item_metadata.tree
  flat_upstream = flax.traverse_util.flatten_dict(item_tree)

  # Trace through the key mapper with abstract _ShapeTracer values.
  mock_upstream = flax.traverse_util.unflatten_dict(
      {k: _ShapeTracer(k, v.shape) for k, v in flat_upstream.items()}
  )
  flat_traced = flax.traverse_util.flatten_dict(
      map_from_upstream_checkpoint(mock_upstream, model_config=model_config)
  )

  # Get downstream shardings from the abstract model.
  with jax.set_mesh(mesh):
    shd = nnx.get_named_sharding(model_state, mesh)
    state_shd = nnx.state(shd) if isinstance(shd, nnx.Module) else shd
    flat_shardings = flax.traverse_util.flatten_dict(
        nnx.to_pure_dict(state_shd)
    )
    flat_shardings = {
        tuple(
            int(k) if isinstance(k, str) and k.isdigit() else k for k in path
        ): s
        for path, s in flat_shardings.items()
    }
    if not flat_shardings and isinstance(model_state, nnx.State):

      def _get_ns(v: Any) -> jax.sharding.NamedSharding:
        if isinstance(v, jax.sharding.NamedSharding):
          return v
        meta = v.get_metadata() if hasattr(v, 'get_metadata') else {}
        spec = (
            getattr(v, 'sharding_names', None)
            or meta.get('out_sharding')
            or getattr(v, 'out_sharding', None)
            or getattr(v, 'sharding', None)
        )
        if isinstance(spec, jax.sharding.NamedSharding):
          return spec
        if not isinstance(spec, jax.sharding.PartitionSpec):
          spec = jax.sharding.PartitionSpec()
        return jax.sharding.NamedSharding(mesh, spec)

      flat_raw = flax.traverse_util.flatten_dict(model_state.raw_mapping)
      flat_shardings = {k: _get_ns(v) for k, v in flat_raw.items()}
  pattern_len = (
      len(model_config.attention_pattern) if model_config.use_scan_layers else 0
  )

  # Invert traced operations to compute upstream PartitionSpecs.
  upstream_target: dict[tuple[str, ...], jax.ShapeDtypeStruct] = {}
  for downstream_key, tracer in flat_traced.items():
    norm_downstream_key = tuple(
        int(k) if isinstance(k, str) and k.isdigit() else k
        for k in downstream_key
    )
    # If using scan layers, map flat downstream key to scan group key.
    if (
        pattern_len > 0
        and len(norm_downstream_key) >= 1
        and (
            norm_downstream_key[0].startswith('layer_')
            or norm_downstream_key[0] in ('scan_groups', 'layers')
        )
    ):
      if norm_downstream_key[0] == 'scan_groups':
        scan_key = norm_downstream_key
      elif norm_downstream_key[0].startswith('layer_'):
        layer_idx = int(norm_downstream_key[0].split('_')[1])
        param_path = norm_downstream_key[1:]
        sub_layer_idx = layer_idx % pattern_len
        scan_key = ('scan_groups', 'sub_layers', sub_layer_idx) + param_path
      else:
        layer_idx = norm_downstream_key[1]
        param_path = norm_downstream_key[2:]
        sub_layer_idx = layer_idx % pattern_len
        scan_key = ('scan_groups', 'sub_layers', sub_layer_idx) + param_path
      if scan_key not in flat_shardings:
        if ('scan_groups',) + scan_key in flat_shardings:
          scan_key = ('scan_groups',) + scan_key
        else:
          logging.info(
              'Skipping sharded restore target for pruned scan key: %s',
              '/'.join(str(p) for p in scan_key),
          )
          continue
      sharding = flat_shardings[scan_key]
      # The stacked param's spec has a leading None for the scan/vmap axis
      # (prepended by _init_scan_layers Phase 2).  The checkpoint stores
      # per-layer (un-stacked) tensors without that axis, so strip it before
      # inverting to the upstream checkpoint spec.
      spec_axes = tuple(sharding.spec)
      if spec_axes and spec_axes[0] is None:
        sharding = jax.sharding.NamedSharding(
            sharding.mesh,
            jax.sharding.PartitionSpec(*spec_axes[1:]),
        )
    else:
      if norm_downstream_key not in flat_shardings:
        logging.info(
            'Skipping sharded restore target for pruned key: %s',
            '/'.join(str(p) for p in norm_downstream_key),
        )
        continue
      sharding = flat_shardings[norm_downstream_key]

    def _find_orig_key(key: Any) -> Any:
      if isinstance(key, tuple):
        str_k = tuple(str(x) if isinstance(x, int) else x for x in key)
        if str_k in flat_upstream:
          return str_k
      if key in flat_upstream:
        return key
      return key

    upstream_spec = tracer.invert_spec(sharding.spec)
    if tracer.key and isinstance(tracer.key[0], tuple):
      # Dual-key tuple (e.g. kv_einsum stacked from separate k_einsum and v_einsum keys)
      single_spec = (
          jax.sharding.PartitionSpec(*upstream_spec[1:])
          if len(upstream_spec) > 3 and upstream_spec[0] is None
          else upstream_spec
      )
      for single_key in tracer.key:
        matched_key = _find_orig_key(single_key)
        orig = flat_upstream[matched_key]
        upstream_target[matched_key] = jax.ShapeDtypeStruct(
            shape=orig.shape,
            dtype=orig.dtype,
            sharding=jax.sharding.NamedSharding(mesh, single_spec),
        )
    else:
      matched_key = _find_orig_key(tracer.key)
      orig = flat_upstream[matched_key]
      upstream_target[matched_key] = jax.ShapeDtypeStruct(
          shape=orig.shape,
          dtype=orig.dtype,
          sharding=jax.sharding.NamedSharding(mesh, upstream_spec),
      )

  def _stringify_keys(d: Any) -> Any:
    if isinstance(d, dict):
      return {
          str(k) if isinstance(k, int) else k: _stringify_keys(v)
          for k, v in d.items()
      }
    return d

  unflattened = flax.traverse_util.unflatten_dict(upstream_target)
  return _stringify_keys(unflattened), ckptr


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: gemma4_model.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
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
  # Translate GCSFuse mount paths to gs:// URIs so TensorStore and
  # PyTreeCheckpointer can resolve them correctly on any backend.
  # This is a no-op for paths that are already gs:// or not on GCSFuse.
  checkpoint_path = checkpoint_manager.gcsfuse_to_gs_path(checkpoint_path)
  logging.info('Creating model from checkpoint path %s', checkpoint_path)

  # ── Phase 0: Resolve subpath ───────────────────────────────────────────
  clean_path = checkpoint_path.rstrip('/')
  if clean_path.endswith('/model_params'):
    resolved_path = clean_path
  elif (epath.Path(clean_path) / 'model_params').exists():
    # Tunix step dir (e.g. .../checkpoints/20000) -> append '/model_params'
    resolved_path = str(epath.Path(clean_path) / 'model_params')
  else:
    resolved_path = clean_path

  # ── Phase 1: Abstract model (no memory allocated) ──────────────────────
  with nnx.use_eager_sharding(True), jax.set_mesh(mesh):
    abs_model = nnx.eval_shape(
        lambda: gemma4_model.Gemma4(model_config, rngs=nnx.Rngs(0))
    )
  model_state = nnx.state(abs_model)

  # ── Phase 2: Restore checkpoint ────────────────────────────────────────
  # PyTreeCheckpointer.restore() auto-detects OCDBT vs zarr format and
  # dispatches to the most efficient compatible handler automatically.
  step_parent = os.path.dirname(resolved_path.rstrip('/'))
  step_name = os.path.basename(step_parent)
  is_step_dir = (
      step_name.isdigit()
      and os.path.basename(resolved_path.rstrip('/')) == 'model_params'
  )
  if is_step_dir:
    ckptr_meta = ocp.PyTreeCheckpointer().metadata(resolved_path)
    top_keys = (
        list(ckptr_meta.item_metadata.tree.keys())
        if ckptr_meta.item_metadata
        else []
    )
    is_native_tunix = (
        'token_embedder' not in top_keys and 'decoder' not in top_keys
    )
    if is_native_tunix:
      from tunix.sft import checkpoint_manager as tunix_ckpt_mgr

      ckpt_root = os.path.dirname(step_parent)
      step_num = int(step_name)
      with jax.set_mesh(mesh):

        def _bind_mesh(x: Any) -> Any:
          if isinstance(x, jax.ShapeDtypeStruct) and isinstance(
              getattr(x, 'sharding', None), jax.sharding.NamedSharding
          ):
            concrete_sharding = jax.sharding.NamedSharding(
                mesh, x.sharding.spec
            )
            return jax.ShapeDtypeStruct(
                x.shape, x.dtype, sharding=concrete_sharding
            )
          return x

        nnx.update(abs_model, jax.tree.map(_bind_mesh, nnx.state(abs_model)))
        mgr = tunix_ckpt_mgr.CheckpointManager(ckpt_root)
        mgr.maybe_restore(abs_model, step=step_num)
        mgr.close()
      logging.info(
          'Restored native Tunix model from step %d via CheckpointManager in'
          ' %.2fs',
          step_num,
          time.monotonic() - t0,
      )
      return abs_model

  if mesh is not None:
    target, ckptr = _build_sharded_restore_target(
        resolved_path,
        model_state,
        mesh,
        model_config,
    )
    raw_params = ckptr.restore(
        resolved_path,
        target=target,
        partial_restore=True,
    )
  else:
    raw_params = ocp.PyTreeCheckpointer().restore(resolved_path)

  # ── Phase 3: Map upstream keys → downstream layout ──────────────────────
  mapped = map_from_upstream_checkpoint(raw_params, model_config=model_config)

  # ── Phase 4: Stack layers for scan (if enabled) ────────────────────────
  if model_config.use_scan_layers:
    mapped = _stack_layers_for_scan(
        mapped,
        model_config.num_layers,
        len(model_config.attention_pattern),
    )

  # ── Phase 5: Prune and validate ────────────────────────────────────────
  pruned = _prune_to_model_keys(mapped, model_state)
  _validate_param_shapes(pruned, model_state)

  # ── Phase 6: Fill missing keys and cast dtype with target shardings ────
  pure_state = nnx.to_pure_dict(model_state)

  if mesh is not None:
    with jax.set_mesh(mesh):
      shardings = nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh))
      # Align pruned structure with pure_state by filling missing leaves
      flat_pure = flax.traverse_util.flatten_dict(pure_state)
      flat_pruned = flax.traverse_util.flatten_dict(pruned)
      for k, v in flat_pure.items():
        if k not in flat_pruned:
          flat_pruned[k] = jnp.zeros(v.shape, dtype=dtype)
      complete_params = flax.traverse_util.unflatten_dict(flat_pruned)
      typed = jax.tree_util.tree_map_with_path(
          lambda p, x, s: jnp.asarray(x, device=s, dtype=dtype),
          complete_params,
          shardings,
      )
  else:
    typed = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=dtype),
        pruned,
    )

  nnx.update(abs_model, typed)

  # ── Phase 6: Materialize any remaining abstract values ─────────────────
  # partial_restore may leave ShapeDtypeStructs for unused keys (e.g. vision
  # weights in a text-only model). Replace them with zeros so subsequent
  # nnx.jit calls don't hit TraceContextErrors.
  def _materialize(x: Any) -> Any:
    if isinstance(x, jax.ShapeDtypeStruct):
      return jnp.zeros(
          x.shape,
          dtype=x.dtype,
          device=getattr(x, 'sharding', None),
      )
    return x

  state = nnx.state(abs_model)
  nnx.update(abs_model, jax.tree_util.tree_map(_materialize, state))

  if mesh is not None:
    _log_sharding_summary(abs_model)

  logging.info(
      '[TIMING] create_model_from_checkpoint: %.1fs',
      time.monotonic() - t0,
  )
  return abs_model


def _log_sharding_summary(model: gemma4_model.Gemma4) -> None:
  """Log a sample of tensor shardings and flag large replicated tensors."""
  flat_state = jax.tree_util.tree_leaves_with_path(nnx.state(model))
  replicated_large: list[tuple[str, tuple[int, ...], int]] = []

  for i, (path, leaf) in enumerate(flat_state):
    if not hasattr(leaf, 'sharding') or not hasattr(leaf, 'shape'):
      continue
    spec = getattr(leaf.sharding, 'spec', None)
    if i < 5:
      key_str = '/'.join(str(k) for k in path)
      logging.info(
          '  [SHARDING] %s: shape=%s spec=%s', key_str, leaf.shape, spec
      )
    if (
        spec is not None
        and all(s is None for s in spec)
        and hasattr(leaf, 'nbytes')
        and leaf.nbytes > 1_000_000
    ):
      replicated_large.append(
          ('/'.join(str(k) for k in path), leaf.shape, leaf.nbytes)
      )

  if replicated_large:
    logging.warning(
        '⚠️ %d large tensors fully replicated:', len(replicated_large)
    )
    for key_str, shape, nbytes in replicated_large[:10]:
      logging.warning(
          '    %s: shape=%s (%.1f MB)', key_str, shape, nbytes / 1e6
      )


def _validate_param_shapes(
    mapped_params: Mapping[str, Any],
    model_state: Any,
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

  # Should not fire after _prune_to_model_keys; kept as defensive guard.
  extra_keys = mapped_keys - model_keys
  if extra_keys:
    logging.warning(
        'Checkpoint has extra keys not in model (will be ignored): %s',
        sorted(str(k) for k in extra_keys),
    )

  mismatched = []
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
    params: Mapping[str, Any],
    model_state: Any,
) -> dict[str, Any]:
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

  filtered = {k: v for k, v in flat_params.items() if k in flat_model}

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


def map_from_upstream_checkpoint(
    params: Mapping[str, Any],
    model_config: gemma4_model.ModelConfig | None = None,
) -> dict[str, Any]:
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
  new_params: dict[tuple[str | int, ...], Any] = {}

  flat_params = flax.traverse_util.flatten_dict(params)
  raw_key_strings = set('/'.join(str(s) for s in k) for k in flat_params.keys())

  # Pass through keys that are already in NNX layout (e.g. from Tunix CheckpointManager)
  if any(
      isinstance(k, tuple) and len(k) >= 2 and k[0] in ('layers', 'scan_groups')
      for k in flat_params.keys()
  ):
    return dict(params)

  for key_path, value in flat_params.items():
    # Normalize semi-flat or nested key_path to a flat list of components.
    parts = list(
        itertools.chain.from_iterable(
            (segment.split('/') if isinstance(segment, str) else [segment])
            for segment in key_path
        )
    )

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
    param_name = parts[-1]
    module_path = parts[:-1]

    # --- Embedder ---
    if module_path and module_path[0] == 'embedder':
      if len(module_path) > 1 and module_path[1] == 'per_layer_embeddings':
        # Rename upstream 'per_layer_embeddings' → Tunix field name.
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif param_name in ('per_layer_embeddings', 'per_layer_input_embedding'):
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif param_name in ('embedding', 'input_embedding'):
        new_params[('embedder', 'input_embedding')] = value
      elif len(module_path) > 1:
        # Sub-modules of the embedder (e.g., mm_input_projection).
        new_params[tuple(module_path + [param_name])] = value
      else:
        # Bare embedder leaf (input_embedding).
        new_params[('embedder', param_name)] = value
      continue

    # --- Final norm ---
    if module_path and module_path[0] == 'final_norm':
      new_params[('final_norm', param_name)] = value
      continue

    # --- Layer weights ---
    if not module_path:
      logging.warning('Unexpected bare param after transformer: %r', key_path)
      continue

    # Skip multimodal modules (e.g., audio_encoder).
    if not module_path[0].startswith('layer_'):
      logging.info(
          'Skipping non-layer module: %s', '/'.join(str(p) for p in parts)
      )
      continue

    layer_idx = ('layers', int(module_path[0].removeprefix('layer_')))

    # Bare leaf on the layer itself (e.g., skip_scale / layer_scalar).
    if len(module_path) == 1:
      leaf_name = (
          'skip_scale'
          if param_name in ('layer_scalar', 'skip_scale')
          else param_name
      )
      new_params[(*layer_idx, leaf_name)] = value
      continue

    # Normalize Linen/Tunix layer submodule names
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

    # MLP gating_einsum -> split into gate_proj and up_proj (dense shared MLP or
    # standard MLP)
    if module_path[1:] in (['mlp', 'gating_einsum'], ['mlp2', 'gating_einsum']):
      if len(value.shape) == 4 or value.shape[0] != 2:
        # MoE gating_einsum (e.g. 128 experts)
        new_params[(*layer_idx, 'moe', 'gating_einsum')] = value
        continue
      new_params[(*layer_idx, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_idx, 'mlp', 'up_proj', 'kernel')] = value[1].T
      continue

    # MLP linear -> down_proj (no transpose) or MoE linear
    if module_path[1:] in (['mlp', 'linear'], ['mlp2', 'linear']):
      if len(value.shape) == 3 and module_path[1] == 'mlp':
        # MoE linear (e.g. 128 experts)
        new_params[(*layer_idx, 'moe', 'linear')] = value
        continue
      new_params[(*layer_idx, 'mlp', 'down_proj', 'kernel')] = value
      continue

    # MoE router and expert scale params.
    if len(module_path) >= 2 and module_path[1] == 'mlp':
      sub = (
          param_name
          if param_name in ('router_logits', 'router_scale', 'per_expert_scale')
          else module_path[-1]
      )
      if sub in ('router_logits', 'router_scale', 'per_expert_scale'):
        new_params[(*layer_idx, 'moe', sub)] = value
        continue

    # Normalize query/key norm names to underscore-prefixed form.
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
      new_params[(*layer_idx, *sub_path, target_norm, param_name)] = value
      continue

    # Everything else: direct mapping.
    new_params[(*layer_idx, *module_path[1:], param_name)] = value

  # Transpose 3D einsum kernels from Linen layout (embed_dim, num_heads, head_dim) -> (num_heads, embed_dim, head_dim)
  for k, v in list(new_params.items()):
    if len(k) >= 4 and k[2] == 'attn' and k[-1] == 'w':
      if (
          len(v.shape) == 3
          and model_config is not None
          and model_config.embed_dim > 0
          and v.shape[0] == model_config.embed_dim
      ):
        new_params[k] = v.transpose((1, 0, 2))

  # Adapt kv_einsum <-> (k_einsum, v_einsum) based on model_config attention pattern
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
      layer_key = ('layers', i, 'attn')
      if is_global:
        kv_w_key = (*layer_key, 'kv_einsum', 'w')
        if kv_w_key in new_params:
          kv_tr = new_params.pop(kv_w_key)
          new_params[(*layer_key, 'k_einsum', 'w')] = kv_tr
          new_params[(*layer_key, 'v_einsum', 'w')] = kv_tr
      else:
        k_w_key = (*layer_key, 'k_einsum', 'w')
        v_w_key = (*layer_key, 'v_einsum', 'w')
        if k_w_key in new_params and v_w_key in new_params:
          k_tr = new_params.pop(k_w_key)
          v_tr = new_params.pop(v_w_key)
          if hasattr(k_tr, 'key'):
            new_params[(*layer_key, 'kv_einsum', 'w')] = _ShapeTracer(
                (k_tr.key, v_tr.key),
                (2, *k_tr.shape),
                k_tr._transposed,
                k_tr._slice_idx,
                k_tr.perm,
            )
          else:
            new_params[(*layer_key, 'kv_einsum', 'w')] = jnp.stack(
                [k_tr, v_tr], axis=0
            )

  return flax.traverse_util.unflatten_dict(new_params)
