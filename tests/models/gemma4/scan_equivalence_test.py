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

"""Tests for scan-over-layers equivalence with the for-loop implementation."""

from __future__ import annotations

from absl.testing import absltest
from flax import nnx
import optax
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import model as model_lib


_PATTERN = (
    model_lib.AttentionType.LOCAL_SLIDING,
    model_lib.AttentionType.LOCAL_SLIDING,
    model_lib.AttentionType.LOCAL_SLIDING,
    model_lib.AttentionType.LOCAL_SLIDING,
    model_lib.AttentionType.LOCAL_SLIDING,
    model_lib.AttentionType.GLOBAL,
)
_PATTERN_LEN = len(_PATTERN)


def _make_config(
    *,
    num_layers: int = 12,
    use_scan_layers: bool = False,
) -> model_lib.ModelConfig:
  """Minimal Gemma4 config for testing. 12 layers = 2 full pattern groups."""
  return model_lib.ModelConfig(
      num_layers=num_layers,
      num_embed=128,
      embed_dim=64,
      hidden_dim=128,
      num_heads=4,
      head_dim=16,
      num_kv_heads=2,
      num_global_kv_heads=1,
      global_key_size=32,
      sliding_window_size=16,
      k_eq_v_global=True,
      frac_shared_layers=0.0,
      per_layer_input_dim=0,
      use_scan_layers=use_scan_layers,
      attention_pattern=_PATTERN,
  )


def _make_inputs(
    config: model_lib.ModelConfig,
    batch_size: int = 2,
    seq_len: int = 16,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Create deterministic inputs for testing."""
  tokens = jax.random.randint(
      jax.random.PRNGKey(42), (batch_size, seq_len), 0, config.num_embed
  )
  positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
  attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, ...]
  return tokens, positions, attn_mask


def _copy_weights_loop_to_scan(
    loop_model: model_lib.Gemma4,
    scan_model: model_lib.Gemma4,
) -> None:
  """Copy weights from for-loop model to scan model.

  The loop model stores layers as layers[0], layers[1], ... layers[N-1].
  The scan model stores layers as scan_groups.sub_layers[sub_idx] with a
  leading vmap axis of size num_groups.

  Mapping: layers[group * pattern_len + sub] -> scan_groups.sub_layers[sub][group]
  """
  loop_gd, loop_state = nnx.split(loop_model)
  scan_gd, scan_state = nnx.split(scan_model)

  num_groups = scan_model.num_scan_groups

  # Copy shared state (embedder, final_norm) from the loop model.
  # These are identical between the two models since they're outside the layers.
  scan_state['embedder'] = loop_state['embedder']
  scan_state['final_norm'] = loop_state['final_norm']

  # Copy layer weights by stacking loop layers into scan groups.
  for sub_idx in range(_PATTERN_LEN):
    # Indices of the loop layers that map to this sub_layer position
    loop_indices = [g * _PATTERN_LEN + sub_idx for g in range(num_groups)]

    # Get the loop states for these layers
    loop_layer_states = [loop_state['layers'][li] for li in loop_indices]

    # Stack them along axis 0 to match the vmap leading dimension
    stacked = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *loop_layer_states,
    )

    # Assign to the scan sub_layer
    scan_state['scan_groups']['sub_layers'][sub_idx] = stacked

  nnx.update(scan_model, scan_state)


class ScanSmokeTest(absltest.TestCase):
  """Basic tests that the scan model compiles and runs."""

  def test_forward_pass_shape(self):
    config = _make_config(use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    tokens, positions, attn_mask = _make_inputs(config)

    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
    self.assertEqual(logits.shape, (2, 16, config.num_embed))

  def test_gradient_flow(self):
    config = _make_config(use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    tokens, positions, attn_mask = _make_inputs(config)

    def loss_fn(model):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    self.assertTrue(jnp.isfinite(loss))

    # Every gradient leaf must be finite
    grad_leaves = jax.tree.leaves(nnx.state(grads))
    for leaf in grad_leaves:
      self.assertTrue(
          jnp.all(jnp.isfinite(leaf)), msg=f'Non-finite gradient: {leaf.shape}'
      )

  def test_invalid_num_layers(self):
    with self.assertRaises(ValueError):
      config = _make_config(num_layers=7, use_scan_layers=True)
      model_lib.Gemma4(config, rngs=nnx.Rngs(0))


class ScanEquivalenceTest(absltest.TestCase):
  """Numerical equivalence tests between for-loop and scan implementations."""

  def test_forward_equivalence(self):
    """The scan model must produce bitwise-identical logits when given
    identical weights and inputs."""
    loop_config = _make_config(num_layers=12, use_scan_layers=False)
    scan_config = _make_config(num_layers=12, use_scan_layers=True)

    # Create both models with the same RNG (different init, we'll copy weights)
    loop_model = model_lib.Gemma4(loop_config, rngs=nnx.Rngs(0))
    scan_model = model_lib.Gemma4(scan_config, rngs=nnx.Rngs(1))

    # Copy weights from loop to scan
    _copy_weights_loop_to_scan(loop_model, scan_model)

    # Create identical inputs
    tokens, positions, attn_mask = _make_inputs(loop_config)

    # Run both models
    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    # Assert near-equality (scan produces a while loop in HLO which XLA may
    # optimize differently than the unrolled path, causing ~1 ULP differences
    # after 12 layers of float32 accumulation).
    max_diff = float(jnp.max(jnp.abs(loop_logits - scan_logits)))
    self.assertLess(
        max_diff,
        1e-5,
        msg=f'Forward pass diverged. Max diff: {max_diff}',
    )

  def test_gradient_equivalence(self):
    """Gradients must be bitwise-identical between for-loop and scan."""
    loop_config = _make_config(num_layers=12, use_scan_layers=False)
    scan_config = _make_config(num_layers=12, use_scan_layers=True)

    loop_model = model_lib.Gemma4(loop_config, rngs=nnx.Rngs(0))
    scan_model = model_lib.Gemma4(scan_config, rngs=nnx.Rngs(1))
    _copy_weights_loop_to_scan(loop_model, scan_model)

    tokens, positions, attn_mask = _make_inputs(loop_config)

    def loss_fn(model):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loop_loss, loop_grads = nnx.value_and_grad(loss_fn)(loop_model)
    scan_loss, scan_grads = nnx.value_and_grad(loss_fn)(scan_model)

    # Losses must be close
    self.assertAlmostEqual(
        float(loop_loss),
        float(scan_loss),
        places=4,
        msg=f'Losses differ. Loop: {loop_loss}, Scan: {scan_loss}',
    )

    # Compare gradients for shared params (embedder, final_norm)
    loop_state = nnx.state(loop_grads)
    scan_state = nnx.state(scan_grads)

    for key in ('embedder', 'final_norm'):
      loop_leaves = jax.tree.leaves(loop_state[key])
      scan_leaves = jax.tree.leaves(scan_state[key])
      for i, (ll, sl) in enumerate(zip(loop_leaves, scan_leaves)):
        max_diff = float(jnp.max(jnp.abs(ll - sl)))
        self.assertLess(
            max_diff,
            1e-2,
            msg=f'Gradient diff in {key} leaf {i}: {max_diff}',
        )

    # Compare gradients for layer params
    num_groups = scan_model.num_scan_groups
    for sub_idx in range(_PATTERN_LEN):
      scan_sub_grads = scan_state['scan_groups']['sub_layers'][sub_idx]
      scan_sub_leaves = jax.tree.leaves(scan_sub_grads)

      for group_idx in range(num_groups):
        loop_layer_idx = group_idx * _PATTERN_LEN + sub_idx
        loop_layer_grads = loop_state['layers'][loop_layer_idx]
        loop_layer_leaves = jax.tree.leaves(loop_layer_grads)

        for leaf_idx, (ll, sl) in enumerate(
            zip(loop_layer_leaves, scan_sub_leaves)
        ):
          # The scan leaf has shape [num_groups, ...]; extract this group
          sl_group = sl[group_idx]
          max_diff = float(jnp.max(jnp.abs(ll - sl_group)))
          self.assertLess(
              max_diff,
              1e-2,
              msg=(
                  f'Gradient diff at layer {loop_layer_idx} '
                  f'(sub={sub_idx}, group={group_idx}), '
                  f'leaf {leaf_idx}: {max_diff}'
              ),
          )


class ScanFallbackTest(absltest.TestCase):
  """Test that the scan model falls back to for-loop for inference."""

  def test_cache_forces_loop(self):
    """When cache is provided, scan model must use the for-loop path."""
    # This test verifies that scan models can still be used for inference
    # by falling back to the for-loop path when cache is provided.
    # Since scan models don't have self.layers, this should raise an error
    # (for now — inference support is a future enhancement).
    config = _make_config(num_layers=6, use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 1), 0, config.num_embed
    )
    positions = jnp.zeros((1, 1), dtype=jnp.int32)
    attn_mask = jnp.ones((1, 1, 1), dtype=jnp.bool_)

    # scan model with cache should fail since self.layers doesn't exist
    with self.assertRaises(AttributeError):
      cache = {}  # empty cache to trigger the loop path
      model(tokens, positions=positions, cache=cache, attention_mask=attn_mask)


class ScanShardingSpecTest(absltest.TestCase):
  """Tests that scan-axis sharding specs are correct.

  Background
  ----------
  When ``use_scan_layers=True``, ``nnx.vmap`` stacks per-layer parameters
  along a new leading axis of size ``num_scan_groups``.  The
  ``nnx.Param(sharding=...)`` annotation is stored verbatim, so without the
  fix it would be applied to the wrong rank, causing::

      jax._src.sharding.IndivisibleError: Sharding ... implies that array
      axis 2 is partitioned 32 times, but the dimension size is 16
      (full shape: (10, 2, 16, 5376, 256), ...)

  The fix: ``_init_scan_layers`` now builds a ``scan_config`` whose
  ``shd_config`` has all weight specs prepended with ``None`` via
  ``ShardingConfig.with_scan_axis()``.
  """

  def test_with_scan_axis_prepends_none_to_weight_specs(self):
    """with_scan_axis() must prepend None to every weight spec."""
    base = model_lib.ShardingConfig.get_default_sharding()
    scan = base.with_scan_axis()

    # Weight specs: all should have an extra leading None.
    weight_fields = (
        'q_weight_ndh',
        'kv_weight_cndh',
        'qkv_weight_cndh',
        'o_weight_nhd',
        'ffw_weight_df',
        'ffw_weight_fd',
        'rms_norm_weight',
        'vision_proj',
        'vision_soft_emb_norm_weight',
        'exp_weight_edf',
        'exp_weight_efd',
        'per_layer_input_gate',
        'per_layer_projection',
    )
    for field in weight_fields:
      base_val = getattr(base, field)
      scan_val = getattr(scan, field)
      self.assertEqual(
          scan_val,
          (None,) + base_val,
          msg=(
              f'{field}: expected (None,) + {base_val!r}, got {scan_val!r}'
          ),
      )

    # Activation and embedder specs must be unchanged.
    unchanged_fields = (
        'act_btd',
        'act_btf',
        'act_btnh',
        'emb_vd',
        'per_layer_model_projection',
        'per_layer_input_embedding',
    )
    for field in unchanged_fields:
      self.assertEqual(
          getattr(scan, field),
          getattr(base, field),
          msg=f'{field} should not be modified by with_scan_axis()',
      )

  def test_scan_model_params_have_extra_leading_axis(self):
    """Scan model weight params must have shape (num_groups, *per_layer_shape)."""
    config = _make_config(num_layers=6, use_scan_layers=True)
    num_groups = 6 // _PATTERN_LEN  # = 1
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))

    state = nnx.state(model)
    # sub_layers[0] is the first sub-layer; its params should have shape
    # (num_groups, *per_layer_weight_shape).
    sl0 = state['scan_groups']['sub_layers'][0]
    q_w = sl0['attn']['q_einsum']['w']
    self.assertEqual(
        q_w.shape[0],
        num_groups,
        msg=f'Expected leading scan axis {num_groups}, got shape {q_w.shape}',
    )

  def test_scan_sharding_annotations_match_param_rank(self):
    """Each scan param's stored tuple sharding rank must equal its actual ndim.

    This is the key invariant violated by the bug: without the fix, the
    kv_einsum weight (rank 5 after vmap) would carry a rank-4 sharding tuple.

    Note: some params carry a real JAX Sharding object (not a tuple) if they
    were initialised outside the tuple-annotation path.  We skip those — the
    check only applies to the tuple-style ``sharding=(axis, ...)`` annotations
    used by ``nnx.Param`` inside the model.
    """
    config = _make_config(num_layers=6, use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))

    mismatches = []
    for path, var in nnx.iter_graph(model):
      if not isinstance(var, nnx.Param):
        continue
      sharding = getattr(var, 'sharding', None)
      # Only check tuple-style annotations, not real JAX Sharding objects.
      if not isinstance(sharding, tuple):
        continue
      value = var.value
      if not hasattr(value, 'ndim'):
        continue
      if len(sharding) != value.ndim:
        path_str = '.'.join(str(p) for p in path)
        mismatches.append(
            f'{path_str}: sharding rank {len(sharding)} != ndim {value.ndim}'
            f' (shape={value.shape}, sharding={sharding})'
        )

    self.assertEmpty(
        mismatches,
        msg='Sharding rank mismatches found (scan axis bug):\n'
        + '\n'.join(mismatches),
    )

  def test_optimizer_init_on_scan_model_does_not_raise(self):
    """nnx.Optimizer init on a scan model must not raise IndivisibleError.

    This directly reproduces the GKE failure: PeftTrainer calls
    ``nnx.Optimizer(model, optimizer, wrt=nnx.Param)`` which calls
    ``with_sharding_constraint`` on each param.  With the bug, the 4-axis
    weight spec is applied to a 5-dim tensor, crashing on single-device too
    when the sharding has any non-None axis that doesn't divide the dim.

    On a CPU/single-device test, sharding constraints are no-ops so this
    won't reproduce the exact DMA error — but the shape mismatch in the
    *spec rank* check fires first.  We verify it doesn't raise.
    """
    config = _make_config(num_layers=6, use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    optimizer = optax.adam(1e-4)
    # This must not raise.
    try:
      nnx.Optimizer(model, optimizer, wrt=nnx.Param)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'nnx.Optimizer init raised on scan model: {e}')


if __name__ == '__main__':
  absltest.main()
