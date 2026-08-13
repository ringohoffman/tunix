# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for PrefixCache autoregressive generation equivalence on Gemma 4."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import functional
from tunix.models.gemma4 import model as gemma4


class Gemma4PrefixCacheTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.config = gemma4.ModelConfig(
        num_layers=6,
        num_embed=1000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=128,
        use_sliding_window_kv_cache=False,
        dtype=jnp.float32,
    )
    self.model = gemma4.Gemma4(self.config, rngs=nnx.Rngs(42))

  @parameterized.named_parameters(
      dict(
          testcase_name="standard_full_cache",
          prefix_len=64,
          suffix_len=8,
          max_new_tokens=16,
          batch_size=2,
      ),
      dict(
          testcase_name="batch_size_4",
          prefix_len=32,
          suffix_len=12,
          max_new_tokens=8,
          batch_size=4,
      ),
  )
  def test_prefix_cache_matches_cold_generation(
      self,
      prefix_len: int,
      suffix_len: int,
      max_new_tokens: int,
      batch_size: int,
  ):
    cache_size = prefix_len + suffix_len + max_new_tokens + 32
    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    suffix_tokens = jax.random.randint(
        jax.random.key(2), (batch_size, suffix_len), 10, 900
    )

    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    # 1. Cold monolithic generation from scratch
    @nnx.jit
    def gen_scratch(m: gemma4.Gemma4, prompt: jax.Array):
      return functional.generate(
          m,
          prompt,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_scratch = gen_scratch(self.model, full_prompt)
    scratch_tokens = np.asarray(out_scratch.tokens)

    # 2. Reusable Prefix prefill (B=1)
    @nnx.jit
    def prefill(m: gemma4.Gemma4, pfx: jax.Array):
      return functional.prefill_prefix(
          m,
          pfx,
          cache_size=cache_size,
          dtype=jnp.float32,
      )

    prefix_cache = prefill(self.model, prefix_tokens)

    # 3. Suffix generation using prefilled prefix cache
    @nnx.jit
    def gen_with_prefix(
        m: gemma4.Gemma4, sfx: jax.Array, pfx_c: functional.PrefixCache
    ):
      return functional.generate(
          m,
          sfx,
          prefix_cache=pfx_c,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_prefix = gen_with_prefix(self.model, suffix_tokens, prefix_cache)
    prefix_tokens_out = np.asarray(out_prefix.tokens)

    # Verify bit-for-bit identical generated tokens
    np.testing.assert_array_equal(scratch_tokens, prefix_tokens_out)

  def test_sampler_prefix_cache_matches_cold_generation(self):
    """Verifies that Sampler.generate_from_tokens with prefix_cache produces identical output to cold generation."""
    from tunix.generate import sampler as sampler_lib
    from tunix.tests import test_common as tc

    prefix_len = 32
    suffix_len = 12
    max_new_tokens = 8
    batch_size = 4
    cache_size = prefix_len + suffix_len + max_new_tokens + 32

    mock_vocab = tc.MockVocab()
    mock_vocab.DecodeIds = lambda ids: " ".join(map(str, ids))
    cache_config = sampler_lib.CacheConfig(
        cache_size=cache_size,
        num_layers=self.config.num_layers,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
    )
    sampler = sampler_lib.Sampler(self.model, mock_vocab, cache_config)

    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    suffix_tokens = jax.random.randint(
        jax.random.key(2), (batch_size, suffix_len), 10, 900
    )
    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    # Cold generation with Sampler
    res_cold = sampler.generate_from_tokens(
        full_prompt,
        max_generation_steps=max_new_tokens,
        temperature=0.0,
    )
    cold_tokens = np.asarray(res_cold.tokens)

    # Prefill prefix then generate suffix
    pfx_cache = sampler.prefill_prefix(np.asarray(prefix_tokens[0]))
    res_cached = sampler.generate_from_tokens(
        full_prompt,
        max_generation_steps=max_new_tokens,
        temperature=0.0,
        prefix_cache=pfx_cache,
    )
    cached_tokens = np.asarray(res_cached.tokens)

    np.testing.assert_array_equal(cold_tokens, cached_tokens)

  def test_scan_layers_prefix_cache_matches_cold_generation(self):
    """Verifies that decoupled prefix cache works correctly with use_scan_layers=True."""
    scan_config = gemma4.ModelConfig(
        num_layers=6,
        num_embed=1000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=128,
        use_sliding_window_kv_cache=False,
        use_scan_layers=True,
        dtype=jnp.float32,
    )
    scan_model = gemma4.Gemma4(scan_config, rngs=nnx.Rngs(42))

    prefix_len = 32
    suffix_len = 12
    max_new_tokens = 8
    batch_size = 4
    cache_size = prefix_len + suffix_len + max_new_tokens + 32

    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    suffix_tokens = jax.random.randint(
        jax.random.key(2), (batch_size, suffix_len), 10, 900
    )
    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    # Cold generation
    @nnx.jit
    def gen_scratch(m: gemma4.Gemma4, prompt: jax.Array):
      return functional.generate(
          m,
          prompt,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_scratch = gen_scratch(scan_model, full_prompt)
    scratch_tokens = np.asarray(out_scratch.tokens)

    # Prefill prefix
    @nnx.jit
    def prefill(m: gemma4.Gemma4, pfx: jax.Array):
      return functional.prefill_prefix(
          m,
          pfx,
          cache_size=cache_size,
          dtype=jnp.float32,
      )

    prefix_cache = prefill(scan_model, prefix_tokens)

    # Suffix generation with prefix cache
    @nnx.jit
    def gen_with_prefix(
        m: gemma4.Gemma4, sfx: jax.Array, pfx_c: functional.PrefixCache
    ):
      return functional.generate(
          m,
          sfx,
          prefix_cache=pfx_c,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_prefix = gen_with_prefix(scan_model, suffix_tokens, prefix_cache)
    prefix_tokens_out = np.asarray(out_prefix.tokens)

    np.testing.assert_array_equal(scratch_tokens, prefix_tokens_out)

  def test_padded_suffix_prefix_cache_matches_cold_generation(self):
    """Verifies that decoupled prefix cache handles padded suffix inputs correctly."""
    prefix_len = 32
    max_suffix_len = 16
    max_new_tokens = 8
    batch_size = 4
    pad_id = 0
    cache_size = prefix_len + max_suffix_len + max_new_tokens + 32

    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    # Create suffixes with variable lengths padded with pad_id=0
    suffix_raw = jax.random.randint(
        jax.random.key(2), (batch_size, max_suffix_len), 10, 900
    )
    lengths = [8, 12, 14, 16]
    suffix_list = []
    for i, length in enumerate(lengths):
      padded = jnp.where(
          jnp.arange(max_suffix_len) < length,
          suffix_raw[i],
          pad_id,
      )
      suffix_list.append(padded)
    suffix_tokens = jnp.stack(suffix_list, axis=0)

    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    @nnx.jit
    def gen_scratch(m: gemma4.Gemma4, prompt: jax.Array):
      return functional.generate(
          m,
          prompt,
          max_new_tokens=max_new_tokens,
          pad_id=pad_id,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_scratch = gen_scratch(self.model, full_prompt)
    scratch_tokens = np.asarray(out_scratch.tokens)

    @nnx.jit
    def prefill(m: gemma4.Gemma4, pfx: jax.Array):
      return functional.prefill_prefix(
          m,
          pfx,
          cache_size=cache_size,
          dtype=jnp.float32,
      )

    prefix_cache = prefill(self.model, prefix_tokens)

    @nnx.jit
    def gen_with_prefix(
        m: gemma4.Gemma4, sfx: jax.Array, pfx_c: functional.PrefixCache
    ):
      return functional.generate(
          m,
          sfx,
          prefix_cache=pfx_c,
          max_new_tokens=max_new_tokens,
          pad_id=pad_id,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_prefix = gen_with_prefix(self.model, suffix_tokens, prefix_cache)
    prefix_tokens_out = np.asarray(out_prefix.tokens)

    np.testing.assert_array_equal(scratch_tokens, prefix_tokens_out)

  def test_bfloat16_prefix_cache_logits_match_cold_generation(self):
    """Verifies that decoupled prefix cache with bfloat16 matches cold generation logits within bf16 precision."""
    bf16_config = gemma4.ModelConfig(
        num_layers=6,
        num_embed=1000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=128,
        use_sliding_window_kv_cache=False,
        dtype=jnp.bfloat16,
    )
    bf16_model = gemma4.Gemma4(bf16_config, rngs=nnx.Rngs(42))

    prefix_len = 32
    suffix_len = 12
    max_new_tokens = 1
    batch_size = 2
    cache_size = prefix_len + suffix_len + max_new_tokens + 32

    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    suffix_tokens = jax.random.randint(
        jax.random.key(2), (batch_size, suffix_len), 10, 900
    )
    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    @nnx.jit
    def gen_scratch(m: gemma4.Gemma4, prompt: jax.Array):
      return functional.generate(
          m,
          prompt,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
          return_logits=True,
      )

    out_scratch = gen_scratch(bf16_model, full_prompt)

    @nnx.jit
    def prefill(m: gemma4.Gemma4, pfx: jax.Array):
      return functional.prefill_prefix(
          m,
          pfx,
          cache_size=cache_size,
          dtype=jnp.bfloat16,
      )

    prefix_cache = prefill(bf16_model, prefix_tokens)

    @nnx.jit
    def gen_with_prefix(
        m: gemma4.Gemma4, sfx: jax.Array, pfx_c: functional.PrefixCache
    ):
      return functional.generate(
          m,
          sfx,
          prefix_cache=pfx_c,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
          return_logits=True,
      )

    out_prefix = gen_with_prefix(bf16_model, suffix_tokens, prefix_cache)

    # Verify logits match within bf16 numerical tolerance across 6 layers
    assert out_scratch.logits is not None and out_prefix.logits is not None
    np.testing.assert_allclose(
        np.asarray(out_scratch.logits, dtype=np.float32),
        np.asarray(out_prefix.logits, dtype=np.float32),
        atol=0.08,
        rtol=0.08,
    )

  def test_kv_sharing_prefix_cache_matches_cold_generation(self):
    """Verifies that decoupled prefix cache works with KV cache sharing patterns."""
    sharing_config = gemma4.ModelConfig(
        num_layers=6,
        num_embed=1000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        global_key_size=64,
        sliding_window_size=128,
        use_sliding_window_kv_cache=False,
        frac_shared_layers=2.0 / 6,
        dtype=jnp.float32,
    )
    sharing_model = gemma4.Gemma4(sharing_config, rngs=nnx.Rngs(42))

    prefix_len = 32
    suffix_len = 12
    max_new_tokens = 8
    batch_size = 4
    cache_size = prefix_len + suffix_len + max_new_tokens + 32

    prefix_tokens = jax.random.randint(
        jax.random.key(1), (1, prefix_len), 10, 900
    )
    suffix_tokens = jax.random.randint(
        jax.random.key(2), (batch_size, suffix_len), 10, 900
    )
    full_prompt = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_tokens, (batch_size, prefix_len)),
            suffix_tokens,
        ],
        axis=1,
    )

    @nnx.jit
    def gen_scratch(m: gemma4.Gemma4, prompt: jax.Array):
      return functional.generate(
          m,
          prompt,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_scratch = gen_scratch(sharing_model, full_prompt)
    scratch_tokens = np.asarray(out_scratch.tokens)

    @nnx.jit
    def prefill(m: gemma4.Gemma4, pfx: jax.Array):
      return functional.prefill_prefix(
          m,
          pfx,
          cache_size=cache_size,
          dtype=jnp.float32,
      )

    prefix_cache = prefill(sharing_model, prefix_tokens)

    @nnx.jit
    def gen_with_prefix(
        m: gemma4.Gemma4, sfx: jax.Array, pfx_c: functional.PrefixCache
    ):
      return functional.generate(
          m,
          sfx,
          prefix_cache=pfx_c,
          max_new_tokens=max_new_tokens,
          pad_id=0,
          eos_ids=1,
          cache_size=cache_size,
          temperature=0.0,
      )

    out_prefix = gen_with_prefix(sharing_model, suffix_tokens, prefix_cache)
    prefix_tokens_out = np.asarray(out_prefix.tokens)

    np.testing.assert_array_equal(scratch_tokens, prefix_tokens_out)


if __name__ == "__main__":
  absltest.main()
