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
import time
from typing import Any

from absl import logging
from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from tunix.models.gemma4 import model as model_lib

import sentencepiece as spm

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
    if len(path) >= 2 and path[0] == 'layers' and isinstance(path[1], int):
      layer_idx = path[1]
      param_path = path[2:]
      sub_layer_idx = layer_idx % pattern_len
      group_idx = layer_idx // pattern_len
      target_path = ('scan_groups', 'sub_layers', sub_layer_idx) + param_path
      collector.setdefault(target_path, {})[group_idx] = val
    else:
      new_flat[path] = val

  for target_path, slices in collector.items():
    sorted_slices = [slices[i] for i in range(num_groups)]
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

  __slots__ = ('key', 'shape', '_transposed', '_slice_idx')

  def __init__(
      self,
      key: tuple[str, ...],
      shape: tuple[int, ...],
      transposed: bool = False,
      slice_idx: int | None = None,
  ) -> None:
    self.key = key
    self.shape = shape
    self._transposed = transposed
    self._slice_idx = slice_idx

  @property
  def T(self) -> _ShapeTracer:
    return _ShapeTracer(
        self.key, self.shape[::-1],
        not self._transposed, self._slice_idx,
    )

  def __getitem__(self, idx: int | slice) -> _ShapeTracer:
    if isinstance(idx, int):
      return _ShapeTracer(
          self.key, self.shape[1:],
          self._transposed, idx,
      )
    return self

  def invert_spec(
      self, spec: jax.sharding.PartitionSpec,
  ) -> jax.sharding.PartitionSpec:
    """Given a downstream PartitionSpec, compute the upstream one."""
    s = tuple(spec)
    if self._transposed:
      s = s[::-1]
    if self._slice_idx is not None:
      s = (None,) + s
    return jax.sharding.PartitionSpec(*s)


def _build_sharded_restore_target(
    checkpoint_path: str,
    model_state: Any,
    mesh: jax.sharding.Mesh,
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

  Returns:
    (upstream_target, checkpointer) — the target tree of ShapeDtypeStructs
    with computed shardings, and the checkpointer instance (reused for
    the subsequent restore call).
  """
  ckptr = ocp.PyTreeCheckpointer()
  meta = ckptr.metadata(checkpoint_path)
  flat_upstream = flax.traverse_util.flatten_dict(meta.item_metadata.tree)

  # Trace through the key mapper with abstract _ShapeTracer values.
  mock_upstream = flax.traverse_util.unflatten_dict({
      k: _ShapeTracer(k, v.shape) for k, v in flat_upstream.items()
  })
  flat_traced = flax.traverse_util.flatten_dict(
      map_from_upstream_checkpoint(mock_upstream)
  )

  # Get downstream shardings from the abstract model.
  flat_shardings = flax.traverse_util.flatten_dict(
      nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh))
  )
  fallback = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  # Invert traced operations to compute upstream PartitionSpecs.
  upstream_target: dict[tuple[str, ...], jax.ShapeDtypeStruct] = {}
  for downstream_key, tracer in flat_traced.items():
    sharding = flat_shardings.get(downstream_key, fallback)
    upstream_spec = tracer.invert_spec(sharding.spec)
    orig = flat_upstream[tracer.key]
    upstream_target[tracer.key] = jax.ShapeDtypeStruct(
        shape=orig.shape,
        dtype=orig.dtype,
        sharding=jax.sharding.NamedSharding(mesh, upstream_spec),
    )

  return flax.traverse_util.unflatten_dict(upstream_target), ckptr


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> model_lib.Gemma4:
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

  # ── Phase 1: Abstract model (no memory allocated) ──────────────────────
  with nnx.use_eager_sharding(True), jax.set_mesh(mesh):
    abs_model = nnx.eval_shape(
        lambda: model_lib.Gemma4(model_config, rngs=nnx.Rngs(0))
    )
  model_state = nnx.state(abs_model)

  # ── Phase 2: Restore checkpoint ────────────────────────────────────────
  if mesh is not None:
    target, ckptr = _build_sharded_restore_target(
        checkpoint_path, model_state, mesh,
    )
    raw_params = ckptr.restore(
        checkpoint_path, target=target, partial_restore=True,
    )
  else:
    raw_params = ocp.PyTreeCheckpointer().restore(checkpoint_path)

  # ── Phase 3: Map upstream keys → downstream layout, prune, validate ────
  mapped = map_from_upstream_checkpoint(raw_params)
  pruned = _prune_to_model_keys(mapped, model_state)
  _validate_param_shapes(pruned, model_state)

  # ── Phase 4: Cast dtype and apply target shardings ─────────────────────
  if mesh is not None:
    shardings = nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh))
    typed = jax.tree_util.tree_map_with_path(
        lambda p, x, s: jnp.asarray(x, device=s, dtype=dtype),
        pruned, shardings,
    )
  else:
    typed = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=dtype), pruned,
    )

  # ── Phase 5: Stack layers for scan (if enabled) ────────────────────────
  if model_config.use_scan_layers:
    typed = _stack_layers_for_scan(
        typed, model_config.num_layers, len(model_config.attention_pattern),
    )

  nnx.update(abs_model, typed)

  # ── Phase 6: Materialize any remaining abstract values ─────────────────
  # partial_restore may leave ShapeDtypeStructs for unused keys (e.g. vision
  # weights in a text-only model). Replace them with zeros so subsequent
  # nnx.jit calls don't hit TraceContextErrors.
  def _materialize(x: Any) -> Any:
    if isinstance(x, jax.ShapeDtypeStruct):
      return jnp.zeros(
          x.shape, dtype=x.dtype, device=getattr(x, 'sharding', None),
      )
    return x

  state = nnx.state(abs_model)
  nnx.update(abs_model, jax.tree_util.tree_map(_materialize, state))

  if mesh is not None:
    _log_sharding_summary(abs_model)

  logging.info(
      '[TIMING] create_model_from_checkpoint: %.1fs', time.monotonic() - t0,
  )
  return abs_model


def _log_sharding_summary(model: model_lib.Gemma4) -> None:
  """Log a sample of tensor shardings and flag large replicated tensors."""
  flat_state = jax.tree_util.tree_leaves_with_path(nnx.state(model))
  replicated_large = []

  for i, (path, leaf) in enumerate(flat_state):
    if not hasattr(leaf, 'sharding') or not hasattr(leaf, 'shape'):
      continue
    spec = getattr(leaf.sharding, 'spec', None)
    if i < 5:
      key_str = '/'.join(str(k) for k in path)
      logging.info('  [SHARDING] %s: shape=%s spec=%s', key_str, leaf.shape, spec)
    if (spec is not None and all(s is None for s in spec)
        and hasattr(leaf, 'nbytes') and leaf.nbytes > 1_000_000):
      replicated_large.append(('/'.join(str(k) for k in path), leaf.shape, leaf.nbytes))

  if replicated_large:
    logging.warning('⚠️ %d large tensors fully replicated:', len(replicated_large))
    for key_str, shape, nbytes in replicated_large[:10]:
      logging.warning('    %s: shape=%s (%.1f MB)', key_str, shape, nbytes / 1e6)



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
        'Checkpoint is missing keys expected by the model:'
        f' {sorted(str(k) for k in missing_keys)}'
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


def map_from_upstream_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
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

  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    # Normalize semi-flat or nested key_path to a flat list of components.
    parts = list(
        itertools.chain.from_iterable(
            (segment.split('/') if isinstance(segment, str) else [segment])
            for segment in key_path
        )
    )

    if parts and parts[0] == 'transformer':
      parts = parts[1:]

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
      logging.info('Skipping non-layer module: %s', '/'.join(str(p) for p in parts))
      continue

    layer_idx = ('layers', int(module_path[0].removeprefix('layer_')))

    # Bare leaf on the layer itself (e.g., skip_scale).
    if len(module_path) == 1:
      new_params[(*layer_idx, param_name)] = value
      continue

    # MLP gating_einsum -> split into gate_proj and up_proj.
    if module_path[1:] == ['mlp', 'gating_einsum']:
      if value.shape[0] != 2:
        raise ValueError(
            f'Expected gating_einsum shape[0]=2, got {value.shape[0]} for'
            f' {"/".join(str(p) for p in parts)}'
        )
      new_params[(*layer_idx, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_idx, 'mlp', 'up_proj', 'kernel')] = value[1].T
      continue

    # MLP linear -> down_proj (no transpose).
    if module_path[1:] == ['mlp', 'linear']:
      new_params[(*layer_idx, 'mlp', 'down_proj', 'kernel')] = value
      continue

    # Normalize query/key norm names to underscore-prefixed form.
    if module_path[-1] in ('query_norm', '_query_norm'):
      new_params[
          (*layer_idx, *module_path[1:-1], '_query_norm', param_name)
      ] = value
      continue
    if module_path[-1] in ('key_norm', '_key_norm'):
      new_params[(*layer_idx, *module_path[1:-1], '_key_norm', param_name)] = (
          value
      )
      continue

    # Everything else: direct mapping.
    new_params[(*layer_idx, *module_path[1:], param_name)] = value

  return flax.traverse_util.unflatten_dict(new_params)
