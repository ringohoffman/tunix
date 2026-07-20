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

"""Tests for scan-over-layers equivalence with the for-loop implementation.

Tests are organized by code path:

  ScanSmokeTest
    Basic "does it run" checks (shape, gradient flow, invalid config).

  ScanForwardEquivalenceTest
    Training-mode forward (cache=None).  Exercises the true ``nnx.scan``
    path inside ``_forward_scan``.

  ScanGenerationEquivalenceTest
    Inference-mode forward (cache≠None).  Exercises the unrolled for-loop
    inside ``_forward_scan``.

  ScanShardingSpecTest
    Sharding annotation correctness under a production-like abstract mesh.

Non-bitwise equivalence between scan and loop
----------------------------------------------
``nnx.scan`` lowers to ``jax.lax.scan`` which compiles into an XLA
``while_loop``.  XLA makes different optimization choices (op fusion,
instruction scheduling) for ``while_loop`` bodies compared to unrolled
graphs, which changes floating-point results due to non-associativity.
Empirical testing confirms:

* **Same model, manual unroll vs. nnx.scan: ~1e-5 forward diff on CPU.**
* Both paths are individually deterministic (run-to-run exact).
* Gradient diffs are amplified (~1e-3 for embedder grads after 12 layers).

Tolerances are set accordingly:

* Forward: ``1e-5``
* Gradient: ``2e-3``
"""

from __future__ import annotations

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax._src.mesh import use_abstract_mesh
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models.gemma4 import model as model_lib

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
    frac_shared_layers: float = 0.0,
    per_layer_input_dim: int = 0,
) -> model_lib.ModelConfig:
  """Minimal Gemma4 config for testing.

  Default: 12 layers = 2 full pattern groups, no KV sharing, no PLE.
  """
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
      frac_shared_layers=frac_shared_layers,
      per_layer_input_dim=per_layer_input_dim,
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

  Mapping: layers[group * pattern_len + sub] ->
  scan_groups.sub_layers[sub][group]
  """
  loop_gd, loop_state = nnx.split(loop_model)
  scan_gd, scan_state = nnx.split(scan_model)

  num_groups = scan_model.num_scan_groups

  # Copy shared state (embedder, final_norm) from the loop model.
  scan_state['embedder'] = loop_state['embedder']
  scan_state['final_norm'] = loop_state['final_norm']

  # Copy layer weights by stacking loop layers into scan groups.
  for sub_idx in range(_PATTERN_LEN):
    loop_indices = [g * _PATTERN_LEN + sub_idx for g in range(num_groups)]
    loop_layer_states = [loop_state['layers'][li] for li in loop_indices]

    stacked = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *loop_layer_states,
    )
    scan_state['scan_groups']['sub_layers'][sub_idx] = stacked

  nnx.update(scan_model, scan_state)


def _make_paired_models(
    *,
    num_layers: int = 12,
    frac_shared_layers: float = 0.0,
    per_layer_input_dim: int = 0,
) -> tuple[model_lib.Gemma4, model_lib.Gemma4, model_lib.ModelConfig]:
  """Create loop + scan model pair with identical weights.

  Returns (loop_model, scan_model, loop_config).
  """
  loop_config = _make_config(
      num_layers=num_layers,
      use_scan_layers=False,
      frac_shared_layers=frac_shared_layers,
      per_layer_input_dim=per_layer_input_dim,
  )
  scan_config = _make_config(
      num_layers=num_layers,
      use_scan_layers=True,
      frac_shared_layers=frac_shared_layers,
      per_layer_input_dim=per_layer_input_dim,
  )

  loop_model = model_lib.Gemma4(loop_config, rngs=nnx.Rngs(0))
  scan_model = model_lib.Gemma4(scan_config, rngs=nnx.Rngs(1))
  _copy_weights_loop_to_scan(loop_model, scan_model)

  return loop_model, scan_model, loop_config


# XLA while_loop introduces ~1e-5 forward diff vs unrolled on CPU.
_FORWARD_ATOL = 1e-5
# Gradient diffs are amplified through backprop; ~1e-3 observed after 12 layers.
_GRADIENT_ATOL = 2e-3


def _assert_close(
    test_case: absltest.TestCase,
    loop_arr: jax.Array,
    scan_arr: jax.Array,
    *,
    atol: float = _FORWARD_ATOL,
    msg: str = '',
):
  """Assert two arrays are element-wise close, with a useful error message."""
  max_diff = float(jnp.max(jnp.abs(loop_arr - scan_arr)))
  test_case.assertLess(
      max_diff,
      atol,
      msg=f'{msg} Max diff: {max_diff}',
  )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


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

    grad_leaves = jax.tree.leaves(nnx.state(grads))
    for leaf in grad_leaves:
      self.assertTrue(
          jnp.all(jnp.isfinite(leaf)), msg=f'Non-finite gradient: {leaf.shape}'
      )

  def test_invalid_num_layers(self):
    with self.assertRaises(ValueError):
      config = _make_config(num_layers=7, use_scan_layers=True)
      model_lib.Gemma4(config, rngs=nnx.Rngs(0))


# ---------------------------------------------------------------------------
# Forward (training) equivalence — exercises the nnx.scan path (cache=None)
# ---------------------------------------------------------------------------


class ScanForwardEquivalenceTest(absltest.TestCase):
  """Numerical equivalence tests between for-loop and scan (cache=None).

  This exercises the true ``nnx.scan`` path inside ``_forward_scan``.
  """

  def test_forward_equivalence_two_groups(self):
    """Standard case: 12 layers = 2 full pattern groups."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config)

    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    _assert_close(
        self, loop_logits, scan_logits, msg='Forward pass (2 groups) diverged.'
    )

  def test_forward_equivalence_single_group(self):
    """Edge case: num_layers == pattern_len (1 scan group)."""
    loop_model, scan_model, config = _make_paired_models(num_layers=6)
    tokens, positions, attn_mask = _make_inputs(config)

    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    _assert_close(
        self, loop_logits, scan_logits, msg='Forward pass (1 group) diverged.'
    )

  def test_forward_equivalence_three_groups(self):
    """18 layers = 3 full pattern groups — exercises more iterations."""
    loop_model, scan_model, config = _make_paired_models(num_layers=18)
    tokens, positions, attn_mask = _make_inputs(config)

    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    _assert_close(
        self, loop_logits, scan_logits, msg='Forward pass (3 groups) diverged.'
    )

  def test_forward_equivalence_batch_size_1(self):
    """Edge case: batch_size=1."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config, batch_size=1)

    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    _assert_close(
        self, loop_logits, scan_logits, msg='Forward pass (batch=1) diverged.'
    )

  def test_forward_equivalence_with_segment_ids(self):
    """Training-mode equivalence with segment_ids for document packing."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)
    segment_ids = jnp.array(
        [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 3, 3, 3, 3]],
        dtype=jnp.int32,
    )

    loop_logits, _ = loop_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        segment_ids=segment_ids,
    )
    scan_logits, _ = scan_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        segment_ids=segment_ids,
    )

    _assert_close(
        self,
        loop_logits,
        scan_logits,
        msg='Forward pass with segment_ids diverged.',
    )

  def test_forward_equivalence_return_hidden_states(self):
    """return_hidden_states=True must produce identical hidden states."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config)

    loop_out = loop_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        return_hidden_states=True,
    )
    scan_out = scan_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        return_hidden_states=True,
    )

    self.assertIsNotNone(loop_out.hidden_states)
    self.assertIsNotNone(scan_out.hidden_states)
    _assert_close(
        self,
        loop_out.hidden_states,
        scan_out.hidden_states,
        msg='Hidden states diverged.',
    )
    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Logits with return_hidden_states diverged.',
    )

  def test_forward_equivalence_target_indices(self):
    """target_indices must produce identical sparse logits."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config, seq_len=16)
    target_indices = jnp.array([[0, 7, 15], [3, 10, 15]], dtype=jnp.int32)

    loop_out = loop_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        target_indices=target_indices,
    )
    scan_out = scan_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        target_indices=target_indices,
    )

    self.assertEqual(loop_out.logits.shape, (2, 3, config.num_embed))
    self.assertEqual(scan_out.logits.shape, (2, 3, config.num_embed))
    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Logits with target_indices diverged.',
    )

  def test_forward_equivalence_decode_only_last_token(self):
    """decode_only_last_token must produce identical single-token logits."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config, seq_len=16)

    loop_out = loop_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        decode_only_last_token=True,
    )
    scan_out = scan_model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        decode_only_last_token=True,
    )

    self.assertEqual(loop_out.logits.shape, (2, 1, config.num_embed))
    self.assertEqual(scan_out.logits.shape, (2, 1, config.num_embed))
    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Logits with decode_only_last_token diverged.',
    )

  def test_forward_equivalence_with_per_layer_inputs(self):
    """Training-mode forward equivalence with per_layer_input_dim > 0."""
    loop_model, scan_model, config = _make_paired_models(
        num_layers=12, per_layer_input_dim=16
    )
    tokens, positions, attn_mask = _make_inputs(config)

    loop_logits, _ = loop_model(
        tokens, positions=positions, attention_mask=attn_mask
    )
    scan_logits, _ = scan_model(
        tokens, positions=positions, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_logits,
        scan_logits,
        msg='Forward pass with per_layer_inputs diverged.',
    )

  def test_forward_raises_for_shared_kv_layers(self):
    """Training-mode forward (cache=None) must raise ValueError for frac_shared_layers > 0."""
    _, scan_model, config = _make_paired_models(
        num_layers=12, frac_shared_layers=0.5
    )
    tokens, positions, attn_mask = _make_inputs(config)

    with self.assertRaises(ValueError):
      scan_model(tokens, positions=positions, attention_mask=attn_mask)

  def test_gradient_equivalence(self):
    """Gradients must be near-identical between for-loop and scan."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    tokens, positions, attn_mask = _make_inputs(config)

    def loss_fn(model):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loop_loss, loop_grads = nnx.value_and_grad(loss_fn)(loop_model)
    scan_loss, scan_grads = nnx.value_and_grad(loss_fn)(scan_model)

    # Losses must be close.
    self.assertAlmostEqual(
        float(loop_loss),
        float(scan_loss),
        places=4,
        msg=f'Losses differ. Loop: {loop_loss}, Scan: {scan_loss}',
    )

    # Compare gradients for shared params (embedder, final_norm).
    loop_state = nnx.state(loop_grads)
    scan_state = nnx.state(scan_grads)

    for key in ('embedder', 'final_norm'):
      loop_leaves = jax.tree.leaves(loop_state[key])
      scan_leaves = jax.tree.leaves(scan_state[key])
      for i, (ll, sl) in enumerate(zip(loop_leaves, scan_leaves)):
        max_diff = float(jnp.max(jnp.abs(ll - sl)))
        self.assertLess(
            max_diff,
            _GRADIENT_ATOL,
            msg=f'Gradient diff in {key} leaf {i}: {max_diff}',
        )

    # Compare gradients for layer params.
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
          sl_group = sl[group_idx]
          max_diff = float(jnp.max(jnp.abs(ll - sl_group)))
          self.assertLess(
              max_diff,
              _GRADIENT_ATOL,
              msg=(
                  f'Gradient diff at layer {loop_layer_idx} '
                  f'(sub={sub_idx}, group={group_idx}), '
                  f'leaf {leaf_idx}: {max_diff}'
              ),
          )


# ---------------------------------------------------------------------------
# Generation (inference) equivalence — exercises the unrolled cache path
# ---------------------------------------------------------------------------


class ScanGenerationEquivalenceTest(absltest.TestCase):
  """Numerical equivalence tests for generation/inference with KV cache.

  When ``cache is not None``, ``_forward_scan`` falls back to an
  unrolled for-loop over individually merged layers.
  """

  def test_prefill_and_single_decode_step(self):
    """Basic prefill + one decode step equivalence."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)

    loop_cache = loop_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)

    loop_out = loop_model(
        tokens, positions=positions, cache=loop_cache, attention_mask=attn_mask
    )
    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Prefill logits diverged.',
    )

    # Single decode step.
    tok_decode = jax.random.randint(
        jax.random.PRNGKey(1), (2, 1), 0, config.num_embed
    )
    pos_decode = jnp.full((2, 1), 8)
    mask_decode = jnp.ones((2, 1, 16), dtype=jnp.bool_)

    loop_dec = loop_model(
        tok_decode,
        positions=pos_decode,
        cache=loop_out.cache,
        attention_mask=mask_decode,
    )
    scan_dec = scan_model(
        tok_decode,
        positions=pos_decode,
        cache=scan_out.cache,
        attention_mask=mask_decode,
    )

    _assert_close(
        self,
        loop_dec.logits,
        scan_dec.logits,
        msg='Single decode step logits diverged.',
    )

  def test_multi_step_autoregressive_decode(self):
    """Multi-step decode must stay in sync across all steps."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)
    max_seq_len = 24
    prefill_len = 8
    num_decode_steps = 5

    loop_cache = loop_model.init_cache(
        batch_size=2, max_seq_len=max_seq_len, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=max_seq_len, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(
        config, batch_size=2, seq_len=prefill_len
    )

    loop_out = loop_model(
        tokens, positions=positions, cache=loop_cache, attention_mask=attn_mask
    )
    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Prefill logits diverged (multi-step).',
    )

    loop_cache = loop_out.cache
    scan_cache = scan_out.cache

    for step in range(num_decode_steps):
      pos = prefill_len + step
      tok = jax.random.randint(
          jax.random.PRNGKey(100 + step), (2, 1), 0, config.num_embed
      )
      pos_arr = jnp.full((2, 1), pos)
      mask = jnp.ones((2, 1, max_seq_len), dtype=jnp.bool_)

      loop_dec = loop_model(
          tok, positions=pos_arr, cache=loop_cache, attention_mask=mask
      )
      scan_dec = scan_model(
          tok, positions=pos_arr, cache=scan_cache, attention_mask=mask
      )

      _assert_close(
          self,
          loop_dec.logits,
          scan_dec.logits,
          msg=f'Decode step {step} logits diverged.',
      )

      loop_cache = loop_dec.cache
      scan_cache = scan_dec.cache

  def test_generation_batch_size_1(self):
    """Edge case: batch_size=1 with cache."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)

    loop_cache = loop_model.init_cache(
        batch_size=1, max_seq_len=16, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=1, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=1, seq_len=8)

    loop_out = loop_model(
        tokens, positions=positions, cache=loop_cache, attention_mask=attn_mask
    )
    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Prefill logits (batch=1) diverged.',
    )

  def test_generation_single_group(self):
    """Edge case: num_layers == pattern_len with cache."""
    loop_model, scan_model, config = _make_paired_models(num_layers=6)

    loop_cache = loop_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)

    loop_out = loop_model(
        tokens, positions=positions, cache=loop_cache, attention_mask=attn_mask
    )
    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Single-group generation logits diverged.',
    )

  def test_generation_with_segment_ids(self):
    """Inference with segment_ids for document packing."""
    loop_model, scan_model, config = _make_paired_models(num_layers=12)

    loop_cache = loop_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)
    segment_ids = jnp.array(
        [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 3, 3, 3, 3]],
        dtype=jnp.int32,
    )

    loop_out = loop_model(
        tokens,
        positions=positions,
        cache=loop_cache,
        attention_mask=attn_mask,
        segment_ids=segment_ids,
    )
    scan_out = scan_model(
        tokens,
        positions=positions,
        cache=scan_cache,
        attention_mask=attn_mask,
        segment_ids=segment_ids,
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Segment IDs generation logits diverged.',
    )

  def test_generation_with_shared_kv_layers(self):
    """Verify scan prefill + decode is internally consistent with KV sharing.

    When ``frac_shared_layers > 0``, the scan prefill path intentionally
    differs from the unrolled loop path: the loop path forwards origin
    layer KVs via ``kv_shared_cache`` to shared layers during prefill,
    while the scan path lets each layer compute its own KVs (matching
    training behavior).  This makes a direct logit comparison invalid.

    Instead we verify:
      1. The scan model produces finite prefill output.
      2. Cache entries are correctly populated for origin layers.
      3. A decode step after prefill produces finite, valid logits.
    """
    _, scan_model, config = _make_paired_models(
        num_layers=12, frac_shared_layers=0.5
    )

    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)

    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    # 1. Prefill logits are finite.
    self.assertTrue(
        jnp.all(jnp.isfinite(scan_out.logits)),
        msg='Scan prefill with shared KV layers produced non-finite logits.',
    )

    # 2. Cache entries exist for origin (non-shared) layers.
    unstacked_cache = (
        _unstack_cache(
            scan_out.cache,
            config.num_layers,
            len(_PATTERN),
            scan_model.kv_cache_sharing_patterns,
        )
        if isinstance(scan_out.cache, tuple)
        else scan_out.cache
    )
    for i in range(config.num_layers):
      layer_name = f'layer_{i}'
      if scan_model.kv_cache_sharing_patterns[i] == i:
        self.assertIn(
            layer_name, unstacked_cache, msg=f'Missing cache for {layer_name}'
        )

    # 3. Decode step after prefill produces finite logits.
    tok_decode = jax.random.randint(
        jax.random.PRNGKey(1), (2, 1), 0, config.num_embed
    )
    pos_decode = jnp.full((2, 1), 8)
    mask_decode = jnp.ones((2, 1, 16), dtype=jnp.bool_)

    scan_dec = scan_model(
        tok_decode,
        positions=pos_decode,
        cache=scan_out.cache,
        attention_mask=mask_decode,
    )

    self.assertTrue(
        jnp.all(jnp.isfinite(scan_dec.logits)),
        msg=(
            'Decode after scan prefill with shared KV layers produced'
            ' non-finite logits.'
        ),
    )

  def test_generation_with_per_layer_inputs(self):
    """Equivalence test with per-layer inputs enabled (cache path)."""
    loop_model, scan_model, config = _make_paired_models(
        num_layers=12, per_layer_input_dim=16
    )

    loop_cache = loop_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )
    scan_cache = scan_model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    tokens, positions, attn_mask = _make_inputs(config, batch_size=2, seq_len=8)

    loop_out = loop_model(
        tokens, positions=positions, cache=loop_cache, attention_mask=attn_mask
    )
    scan_out = scan_model(
        tokens, positions=positions, cache=scan_cache, attention_mask=attn_mask
    )

    _assert_close(
        self,
        loop_out.logits,
        scan_out.logits,
        msg='Per-layer input generation logits diverged.',
    )


# ---------------------------------------------------------------------------
# Sharding spec tests
# ---------------------------------------------------------------------------


class ScanShardingSpecTest(absltest.TestCase):
  """Tests that scan-axis sharding specs are correct.

  **Why a mesh is required to catch these errors locally**

  Without an active JAX mesh, ``nnx.Param(sharding=spec)`` calls
  ``shard_value`` which immediately returns the value unchanged — the
  ``with_sharding_constraint`` is a **no-op**.  Both the ``IndivisibleError``
  and the ``ValueError: rank ≥ N required`` only fire when JAX resolves the
  ``PartitionSpec`` against real axis sizes.

  We use ``jax.sharding.AbstractMesh`` + ``nnx.eval_shape`` to simulate the
  production Pathways topology (fsdp=32, tp=1) without physical TPU devices.
  This is the same code path as ``create_model_from_checkpoint`` and is where
  both production crashes occurred.
  """

  _MESH = jax.sharding.AbstractMesh((32, 1), ('fsdp', 'tp'))
  _CONFIG_31B = model_lib.ModelConfig.gemma4_31b()

  def _eval_model(self, config):
    """Trace model init under the production abstract mesh."""
    with use_abstract_mesh(self._MESH):
      nnx.eval_shape(lambda: model_lib.Gemma4(config, rngs=nnx.Rngs(0)))

  def test_with_scan_axis_prepends_none_to_weight_specs(self):
    """with_scan_axis() must prepend None to every weight spec."""
    base = model_lib.ShardingConfig.get_default_sharding()
    scan = base.with_scan_axis()

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
          msg=f'{field}: expected (None,) + {base_val!r}, got {scan_val!r}',
      )

    # Activation / embedder specs must be unchanged.
    for field in (
        'act_btd',
        'act_btf',
        'act_btnh',
        'emb_vd',
        'per_layer_model_projection',
        'per_layer_input_embedding',
    ):
      self.assertEqual(
          getattr(scan, field),
          getattr(base, field),
          msg=f'{field} should not be modified by with_scan_axis()',
      )

  def test_non_scan_31b_init_under_production_mesh(self):
    """Baseline: gemma4_31b without scan must init cleanly under fsdp=32."""
    config = dataclasses.replace(self._CONFIG_31B, use_scan_layers=False)
    try:
      self._eval_model(config)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Non-scan 31b init raised: {e}')

  def test_scan_31b_init_under_production_mesh(self):
    """Model init must not raise under the production mesh (fsdp=32, tp=1)."""
    config = dataclasses.replace(self._CONFIG_31B, use_scan_layers=True)
    try:
      self._eval_model(config)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Scan 31b init raised under fsdp=32 mesh: {e}')

  def test_optimizer_init_31b_under_production_mesh(self):
    """Optimizer init must not raise under the production mesh."""
    config = dataclasses.replace(self._CONFIG_31B, use_scan_layers=True)
    with use_abstract_mesh(self._MESH):
      model = nnx.eval_shape(lambda: model_lib.Gemma4(config, rngs=nnx.Rngs(0)))
    try:
      with use_abstract_mesh(self._MESH):
        nnx.eval_shape(
            lambda: nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
        )
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'nnx.Optimizer(31b scan) raised under fsdp=32 mesh: {e}')

  def test_checkpoint_restore_target_31b_under_production_mesh(self):
    """Building restore targets must not raise under the production mesh."""
    import tempfile
    import shutil
    import orbax.checkpoint as ocp
    from tunix.models.gemma4 import params as params_lib

    config = dataclasses.replace(self._CONFIG_31B, use_scan_layers=True)
    with use_abstract_mesh(self._MESH):
      model = nnx.eval_shape(lambda: model_lib.Gemma4(config, rngs=nnx.Rngs(0)))
    model_state = nnx.state(model)

    # Real un-stacked checkpoint shapes (31B layout).
    fake_upstream = {}
    for i in range(config.num_layers):
      fake_upstream[f'layer_{i}'] = {
          'attn': {
              'q_einsum': {'w': np.zeros((32, 5376, 256))},
              'kv_einsum': {'w': np.zeros((2, 16, 5376, 256))},
              'pre_attention_norm': {'scale': np.zeros((5376,))},
          }
      }
    fake_upstream['embedder'] = {'input_embedding': np.zeros((262144, 5376))}
    fake_upstream['final_norm'] = {'scale': np.zeros((5376,))}

    temp_dir = tempfile.mkdtemp()
    try:
      ckptr = ocp.PyTreeCheckpointer()
      ckptr.save(temp_dir + '/ckpt', fake_upstream)

      with use_abstract_mesh(self._MESH):
        target, _ = params_lib._build_sharded_restore_target(
            temp_dir + '/ckpt', model_state, self._MESH, config
        )
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'_build_sharded_restore_target raised: {e}')
    finally:
      shutil.rmtree(temp_dir)

    # Verify specs match checkpoint rank (scan axis stripped).
    q_w_sharding = target['layer_0']['attn']['q_einsum']['w'].sharding
    self.assertEqual(len(q_w_sharding.spec), 3)
    self.assertEqual(
        q_w_sharding.spec,
        jax.sharding.PartitionSpec('tp', 'fsdp', None),
    )

    kv_w_sharding = target['layer_0']['attn']['kv_einsum']['w'].sharding
    self.assertEqual(len(kv_w_sharding.spec), 4)
    self.assertEqual(
        kv_w_sharding.spec,
        jax.sharding.PartitionSpec(None, 'tp', 'fsdp', None),
    )

  def test_scan_sharding_annotations_match_param_rank(self):
    """After Phase-2 patching, every tuple out_sharding rank must equal ndim."""
    config = _make_config(num_layers=6, use_scan_layers=True)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))

    mismatches = []
    for path, var in nnx.iter_graph(model):
      if not isinstance(var, nnx.Param):
        continue
      spec = var.get_metadata('out_sharding', None)
      if not isinstance(spec, tuple):
        continue
      value = var.get_value()
      if not hasattr(value, 'ndim'):
        continue
      if len(spec) != value.ndim:
        path_str = '.'.join(str(p) for p in path)
        mismatches.append(
            f'{path_str}: spec rank {len(spec)} != ndim {value.ndim}'
            f' (shape={value.shape}, spec={spec})'
        )

    self.assertEmpty(
        mismatches,
        msg='out_sharding rank mismatches after vmap:\n'
        + '\n'.join(mismatches),
    )

  def test_non_scan_31b_optimizes_under_production_mesh(self):
    """The for-loop 31b model must init and optimize cleanly under fsdp=32."""
    config = dataclasses.replace(self._CONFIG_31B, use_scan_layers=False)
    try:
      with use_abstract_mesh(self._MESH):
        model = nnx.eval_shape(
            lambda: model_lib.Gemma4(config, rngs=nnx.Rngs(0))
        )
        nnx.eval_shape(
            lambda: nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
        )
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Non-scan 31b optimizer raised under fsdp=32 mesh: {e}')

  def test_init_cache_stacked_cache_sharding_under_mesh(self):
    """StackedCache tensors must be partitioned by batch axis (shd_b) under mesh."""
    config = _make_config(num_layers=12, use_scan_layers=True)
    with use_abstract_mesh(self._MESH):
      model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
      stacked_cache = model.init_cache(
          batch_size=32, max_seq_len=64, dtype=jnp.float32
      )
      self.assertIsInstance(stacked_cache, tuple)
      for sub_cache in stacked_cache:
        if sub_cache is None:
          continue
        k_sharding = getattr(sub_cache['k'], 'sharding', None)
        v_sharding = getattr(sub_cache['v'], 'sharding', None)
        self.assertIsNotNone(
            k_sharding, msg="Stacked cache 'k' missing sharding spec"
        )
        self.assertIsNotNone(
            v_sharding, msg="Stacked cache 'v' missing sharding spec"
        )
        if isinstance(k_sharding, jax.sharding.NamedSharding):
          self.assertEqual(k_sharding.spec[0], None)
          self.assertEqual(k_sharding.spec[1], 'fsdp')


def _unstack_cache(
    stacked_cache: model_lib.StackedCache,
    num_layers: int,
    pattern_len: int = 6,
    kv_cache_sharing_patterns: list[int] | None = None,
) -> model_lib.Cache:
  """Converts a stacked cache tuple into an unstacked per-layer cache dict."""
  dict_cache: model_lib.Cache = {}
  if kv_cache_sharing_patterns is None:
    kv_cache_sharing_patterns = list(range(num_layers))

  for i in range(num_layers):
    if kv_cache_sharing_patterns[i] != i:
      continue  # Shared layers have no individual cache entry.

    group_idx = i // pattern_len
    sub_idx = i % pattern_len
    c = stacked_cache[sub_idx]
    if c is not None:
      dict_cache[f'layer_{i}'] = {
          'k': c['k'][group_idx],
          'v': c['v'][group_idx],
          'end_index': c['end_index'][group_idx],
      }
  return dict_cache


def _stack_cache(
    cache: model_lib.Cache,
    num_layers: int,
    pattern_len: int = 6,
    kv_cache_sharing_patterns: list[int] | None = None,
) -> model_lib.StackedCache:
  """Converts an unstacked per-layer cache dict into a stacked cache tuple."""
  num_scan_groups = num_layers // pattern_len
  if kv_cache_sharing_patterns is None:
    kv_cache_sharing_patterns = list(range(num_layers))

  scan_cache_list: list[model_lib.LayerCache | None] = []
  for sub_idx in range(pattern_len):
    group_layer_indices = [
        g * pattern_len + sub_idx for g in range(num_scan_groups)
    ]
    proto_i = next(
        (i for i in group_layer_indices if kv_cache_sharing_patterns[i] == i),
        None,
    )
    if proto_i is not None and f'layer_{proto_i}' in cache:
      proto_cache = cache[f'layer_{proto_i}']
      k_shape = proto_cache['k'].shape
      v_shape = proto_cache['v'].shape
      end_idx_shape = proto_cache['end_index'].shape
      k_dtype = proto_cache['k'].dtype
      v_dtype = proto_cache['v'].dtype
      end_idx_dtype = proto_cache['end_index'].dtype
    else:
      proto_cache = None

    if proto_cache is not None:
      ks = []
      vs = []
      end_indices = []
      for i in group_layer_indices:
        if kv_cache_sharing_patterns[i] == i and f'layer_{i}' in cache:
          c = cache[f'layer_{i}']
          ks.append(c['k'])
          vs.append(c['v'])
          end_indices.append(c['end_index'])
        else:
          ks.append(jnp.zeros(k_shape, dtype=k_dtype))
          vs.append(jnp.zeros(v_shape, dtype=v_dtype))
          end_indices.append(jnp.zeros(end_idx_shape, dtype=end_idx_dtype))

      scan_cache_list.append({
          'k': jnp.stack(ks, axis=0),
          'v': jnp.stack(vs, axis=0),
          'end_index': jnp.stack(end_indices, axis=0),
      })
    else:
      scan_cache_list.append(None)

  return tuple(scan_cache_list)


class CacheConversionTest(absltest.TestCase):
  """Tests for _stack_cache and _unstack_cache conversion utilities."""

  def test_roundtrip_cache_conversion(self):
    """Test converting dict cache -> stacked cache -> dict cache roundtrip."""
    config = _make_config(num_layers=12, use_scan_layers=False)
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    dict_cache = model.init_cache(
        batch_size=2, max_seq_len=16, dtype=jnp.float32
    )

    stacked = _stack_cache(
        dict_cache, config.num_layers, pattern_len=len(_PATTERN)
    )
    self.assertIsInstance(stacked, tuple)
    self.assertEqual(len(stacked), len(_PATTERN))

    unstacked = _unstack_cache(
        stacked, config.num_layers, pattern_len=len(_PATTERN)
    )
    self.assertEqual(set(dict_cache.keys()), set(unstacked.keys()))

    for key in dict_cache:
      for k in ('k', 'v', 'end_index'):
        np.testing.assert_allclose(dict_cache[key][k], unstacked[key][k])


if __name__ == '__main__':
  absltest.main()
