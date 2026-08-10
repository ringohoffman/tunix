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

"""Tests for Gemma 4 model."""

from __future__ import annotations

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import model as model_lib


class ModelTest(absltest.TestCase):

  def test_forward_pass_dense(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
    self.assertEqual(logits.shape, (2, 32, config.num_embed))
    print(f'{logits.shape=}')

  def test_forward_pass_moe(self):
    config = model_lib.ModelConfig.gemma4_26b_a4b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.expert_dim = 128

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]
    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)

  def test_flash_attention_cache_fallback(self):
    """Verify flash attention gracefully falls back for cached prefill."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0
    config.use_flash_attention = True
    config.flash_attention_block_size = 1024

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    # Prefix length 228 (not divisible by 1024).
    pfx_len = 228
    cache_len = 512
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, pfx_len), 0, config.num_embed
    )
    positions = jnp.arange(pfx_len)[None, :]
    attn_mask = jnp.pad(
        jnp.tril(jnp.ones((pfx_len, pfx_len), dtype=jnp.bool_))[None, ...],
        ((0, 0), (0, 0), (0, cache_len - pfx_len)),
    )
    cache = model.init_cache(batch_size=1, max_seq_len=cache_len, dtype=jnp.float32)

    # Should run cleanly via fallback without raising ValueError on q_block_size.
    out = model(tokens, positions=positions, cache=cache, attention_mask=attn_mask)
    self.assertEqual(out.logits.shape, (1, pfx_len, config.num_embed))
    self.assertIsNotNone(out.cache)

  def test_remat_block(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.BLOCK
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    def loss_fn(model, tokens, positions, attn_mask):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loss, grads = nnx.value_and_grad(loss_fn)(
        model, tokens, positions, attn_mask
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(grads)

  def test_remat_decoder(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.DECODER
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    def loss_fn(model, tokens, positions, attn_mask):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loss, grads = nnx.value_and_grad(loss_fn)(
        model, tokens, positions, attn_mask
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(grads)

  def test_remat_while_loop_trace_context(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.BLOCK
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    graphdef, state = nnx.split(model, nnx.Param)

    def decode_fn(params):
      def body_fn(step, _):
        transformer = nnx.merge(graphdef, params)
        logits, _ = transformer(
            tokens, positions=positions, attention_mask=attn_mask
        )
        return step + 1, logits

      return jax.lax.while_loop(
          lambda state: state[0] < 1,
          lambda state: body_fn(state[0], state[1]),
          (jnp.array(0), jnp.zeros((2, 32, config.num_embed))),
      )

    compiled_decode = jax.jit(decode_fn)
    _, logits = compiled_decode(state)
    self.assertEqual(logits.shape, (2, 32, config.num_embed))

  def test_forward_loop_shared_scan(self):
    """Verify _forward_loop works on a scan model with shared layers."""
    config = _make_shared_scan_config()
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    tokens, positions, attn_mask = _make_test_inputs(config)

    scan_out = model(tokens, positions=positions, attention_mask=attn_mask)
    # Manually invoke _forward_loop on the scan model
    x = model.embedder.encode(tokens)
    per_layer_inputs = None
    loop_x, _ = model._forward_loop(
        x,
        positions,
        cache=None,
        attention_mask=attn_mask,
        per_layer_inputs=per_layer_inputs,
        new_cache={},
        transient_kvs={},
        is_prefill=True,
        segment_ids=None,
    )
    loop_logits = model.embedder.decode(model.final_norm(loop_x)).astype(
        jnp.float32
    )
    max_diff = float(jnp.max(jnp.abs(scan_out.logits - loop_logits)))
    self.assertLess(
        max_diff,
        1e-5,
        msg=f'_forward_loop vs _forward_scan diff too large: {max_diff}',
    )


def _make_shared_scan_config() -> model_lib.ModelConfig:
  """Minimal config with shared layers + scan for skip_kv_projection tests.

  12 layers, pattern_len=6 → 2 groups.
  frac_shared=0.5 → 6 unshared (1 group), 6 shared (1 group).
  """
  return model_lib.ModelConfig(
      num_layers=12,
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
      frac_shared_layers=0.5,
      use_scan_layers=True,
      attention_pattern=(
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.LOCAL_SLIDING,
          model_lib.AttentionType.GLOBAL,
      ),
  )


def _make_test_inputs(config, batch_size=2, seq_len=8):
  tokens = jax.random.randint(
      jax.random.PRNGKey(42), (batch_size, seq_len), 0, config.num_embed
  )
  positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
  attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, ...]
  return tokens, positions, attn_mask


class SkipKVProjectionTest(absltest.TestCase):
  """Numerical equivalence tests for the skip_kv_projection optimization.

  Proves that skipping the KV einsum for shared layers in the two-scan
  inference path produces bitwise-identical outputs to the baseline
  (computing KV then discarding via jnp.where).
  """

  def _set_skip_kv(self, model, value):
    self.assertTrue(hasattr(model, 'shared_scan_groups'))
    model.shared_scan_groups.skip_kv_projection = value

  def test_prefill_equivalence(self):
    """Prefill logits must be identical with and without skip_kv_projection."""
    config = _make_shared_scan_config()
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    self.assertTrue(model.shared_scan_groups.skip_kv_projection)

    tokens, positions, attn_mask = _make_test_inputs(config)

    # Baseline: compute KV then discard via jnp.where.
    self._set_skip_kv(model, False)
    cache_b = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    out_b = model(
        tokens,
        positions=positions,
        cache=cache_b,
        attention_mask=attn_mask,
    )

    # Optimized: skip KV einsum entirely.
    self._set_skip_kv(model, True)
    cache_o = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    out_o = model(
        tokens,
        positions=positions,
        cache=cache_o,
        attention_mask=attn_mask,
    )

    max_diff = float(jnp.max(jnp.abs(out_b.logits - out_o.logits)))
    self.assertEqual(
        max_diff, 0.0, f'Prefill logits differ: max_diff={max_diff}'
    )

    # Non-shared layer caches must also be identical.
    if isinstance(out_b.cache, dict):
      for key in out_b.cache:
        for field in ('k', 'v'):
          cache_diff = float(
              jnp.max(
                  jnp.abs(out_b.cache[key][field] - out_o.cache[key][field])
              )
          )
          self.assertEqual(
              cache_diff,
              0.0,
              f'Cache {key}/{field} differs: max_diff={cache_diff}',
          )

  def test_decode_equivalence(self):
    """Decode logits after prefill must be identical."""
    config = _make_shared_scan_config()
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    tokens, positions, attn_mask = _make_test_inputs(config, seq_len=8)

    # Prefill with each setting.
    self._set_skip_kv(model, False)
    cache_b = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    pfill_b = model(
        tokens,
        positions=positions,
        cache=cache_b,
        attention_mask=attn_mask,
    )

    self._set_skip_kv(model, True)
    cache_o = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    pfill_o = model(
        tokens,
        positions=positions,
        cache=cache_o,
        attention_mask=attn_mask,
    )

    # Decode step.
    tok_dec = jax.random.randint(
        jax.random.PRNGKey(1), (2, 1), 0, config.num_embed
    )
    pos_dec = jnp.full((2, 1), 8)
    mask_dec = jnp.ones((2, 1, 16), dtype=jnp.bool_)

    self._set_skip_kv(model, False)
    dec_b = model(
        tok_dec,
        positions=pos_dec,
        cache=pfill_b.cache,
        attention_mask=mask_dec,
    )

    self._set_skip_kv(model, True)
    dec_o = model(
        tok_dec,
        positions=pos_dec,
        cache=pfill_o.cache,
        attention_mask=mask_dec,
    )

    max_diff = float(jnp.max(jnp.abs(dec_b.logits - dec_o.logits)))
    self.assertEqual(
        max_diff, 0.0, f'Decode logits differ: max_diff={max_diff}'
    )

  def test_multi_decode_steps(self):
    """Multiple consecutive decode steps must stay identical."""
    config = _make_shared_scan_config()
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    tokens, positions, attn_mask = _make_test_inputs(config, seq_len=8)

    # Prefill.
    self._set_skip_kv(model, False)
    cache_b = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    out_b = model(
        tokens,
        positions=positions,
        cache=cache_b,
        attention_mask=attn_mask,
    )
    cache_b = out_b.cache

    self._set_skip_kv(model, True)
    cache_o = model.init_cache(batch_size=2, max_seq_len=16, dtype=jnp.float32)
    out_o = model(
        tokens,
        positions=positions,
        cache=cache_o,
        attention_mask=attn_mask,
    )
    cache_o = out_o.cache

    # 4 decode steps.
    for step in range(4):
      tok = jax.random.randint(
          jax.random.PRNGKey(step + 10), (2, 1), 0, config.num_embed
      )
      pos = jnp.full((2, 1), 8 + step)
      mask = jnp.ones((2, 1, 16), dtype=jnp.bool_)

      self._set_skip_kv(model, False)
      dec_b = model(tok, positions=pos, cache=cache_b, attention_mask=mask)
      cache_b = dec_b.cache

      self._set_skip_kv(model, True)
      dec_o = model(tok, positions=pos, cache=cache_o, attention_mask=mask)
      cache_o = dec_o.cache

      max_diff = float(jnp.max(jnp.abs(dec_b.logits - dec_o.logits)))
      self.assertEqual(
          max_diff,
          0.0,
          f'Decode step {step} logits differ: max_diff={max_diff}',
      )


if __name__ == '__main__':
  absltest.main()
