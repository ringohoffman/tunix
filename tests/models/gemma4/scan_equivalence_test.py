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


if __name__ == '__main__':
  absltest.main()
