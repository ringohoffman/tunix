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

"""Tests for functional autoregressive generation in Tunix."""

import re

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import constrained
from tunix.generate import functional
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.tests import test_common as tc


class ToyTransformerWithCache(tc.ToyTransformer):
  """ToyTransformer subclass that defines init_cache."""

  def init_cache(
      self, batch_size: int, cache_size: int, dtype=jnp.float32
  ) -> dict[str, jax.Array]:
    return {
        "k": jnp.zeros((batch_size, cache_size, 16), dtype=dtype),
        "v": jnp.zeros((batch_size, cache_size, 16), dtype=dtype),
    }


class FunctionalGenerateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tc.MockVocab()
    self.model = ToyTransformerWithCache(
        config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    self.pad_id = self.vocab.pad_id()
    self.eos_id = self.vocab.eos_id()

  def test_output_shape(self):
    prompt = jnp.array([[3, 4, 5], [6, 7, 8]], dtype=jnp.int32)
    max_new_tokens = 5
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=max_new_tokens,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )
    self.assertIsInstance(out, functional.GenerateOutput)
    self.assertEqual(out.tokens.shape, (2, 5))
    self.assertEqual(out.tokens.dtype, jnp.int32)
    self.assertIsNone(out.logits)
    self.assertIsNone(out.logprobs)

  def test_greedy_deterministic(self):
    prompt = jnp.array([[3, 4, 5, 6]], dtype=jnp.int32)
    out1 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=6,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )
    out2 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=6,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )
    np.testing.assert_array_equal(out1.tokens, out2.tokens)

  def test_temperature_sampling_varies(self):
    prompt = jnp.array([[3, 4, 5, 6]], dtype=jnp.int32)
    out1 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=6,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=1.0,
        key=jax.random.key(1),
    )
    out2 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=6,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=1.0,
        key=jax.random.key(2),
    )
    self.assertFalse(np.array_equal(out1.tokens, out2.tokens))

  def test_jit_compatibility(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)

    @jax.jit
    def jitted_generate(mdl, p):
      return functional.generate(
          mdl,
          p,
          max_new_tokens=4,
          pad_id=self.pad_id,
          eos_id=self.eos_id,
          cache_size=32,
          temperature=0.0,
      )

    out_jit = jitted_generate(self.model, prompt)
    out_nojit = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )
    np.testing.assert_array_equal(out_jit.tokens, out_nojit.tokens)

  def test_stop_gradient_blocks_grads(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)

    def loss_fn(model):
      out = functional.generate(
          model,
          prompt,
          max_new_tokens=4,
          pad_id=self.pad_id,
          eos_id=self.eos_id,
          cache_size=32,
          temperature=0.0,
      )
      return jax.lax.stop_gradient(out.tokens.astype(jnp.float32)).sum()

    grads = nnx.grad(loss_fn)(self.model)
    for leaf in jax.tree.leaves(nnx.state(grads)):
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_matches_sampler_greedy(self):
    prompt_ids = np.array([[3, 4, 5]], dtype=np.int32)
    max_new_tokens = 4

    out_func = functional.generate(
        self.model,
        jnp.array(prompt_ids),
        max_new_tokens=max_new_tokens,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )

    sampler = sampler_lib.Sampler(
        transformer=self.model,
        tokenizer=self.vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=32,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    out_sampler = sampler.generate_from_tokens(
        prompt_ids,
        max_generation_steps=max_new_tokens,
        temperature=0.0,
        pad_output=True,
    )

    np.testing.assert_array_equal(
        out_func.tokens,
        out_sampler.tokens,
    )

  def test_forbidden_tokens(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    forbidden = jnp.array([7], dtype=jnp.int32)
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        forbidden_token_ids=forbidden,
    )
    self.assertNotIn(7, np.asarray(out.tokens).tolist()[0])

  def test_constrained_decoding(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    ta = tok_adapter.TokenizerAdapter(self.vocab)
    tables = constrained.chain_constraints(
        r"(hello|world)",
        token_id_to_str=ta.token_id_to_str,
        vocab_size=ta.vocab_size,
        eos_token_ids=(self.eos_id,),
    )
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=2,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        constraint_tables=tables,
    )
    generated_ids = [
        int(x)
        for x in out.tokens[0]
        if int(x) not in (self.pad_id, self.eos_id)
    ]
    decoded = ta.decode(generated_ids)
    self.assertIn(decoded.strip(), ("hello", "world"))

  def test_constrained_decoding_with_wildcard_thinking(self):
    custom_vocab = tc.MockVocab(
        mapping_text_to_id={
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<thought>": 3,
            "</thought>": 4,
            "\n": 5,
            "Safe": 6,
            "Unsafe": 7,
            "random": 8,
            "filler": 9,
            "words": 10,
        }
    )
    custom_model = ToyTransformerWithCache(
        config=tc.ModelConfig(vocab_size=custom_vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    ta = tok_adapter.TokenizerAdapter(custom_vocab)
    pattern = (
        r"<thought>"
        + constrained.bounded_until(r"</thought>", max_tokens=2)
        + r"\n(Safe|Unsafe)"
    )
    tables = constrained.chain_constraints(
        pattern,
        token_id_to_str=ta.token_id_to_str,
        vocab_size=ta.vocab_size,
        eos_token_ids=(custom_vocab.eos_id(),),
    )
    self.assertIsNotNone(tables.default_transitions)

    prompt = jnp.array([[1]], dtype=jnp.int32)
    out = functional.generate(
        custom_model,
        prompt,
        max_new_tokens=8,
        pad_id=custom_vocab.pad_id(),
        eos_id=custom_vocab.eos_id(),
        cache_size=32,
        temperature=0.0,
        constraint_tables=tables,
    )
    generated_ids = [
        int(x)
        for x in out.tokens[0]
        if int(x) not in (custom_vocab.pad_id(), custom_vocab.eos_id())
    ]
    # The first 5 tokens generated must be: <thought>(3), random(8), </thought>(4), \n(5), Unsafe/Safe(7/6)
    self.assertEqual(generated_ids[:4], [3, 8, 4, 5])
    self.assertIn(generated_ids[4], (6, 7))

    decoded = ta.decode(generated_ids)
    # Output must match the constrained pattern
    self.assertTrue(
        re.search(
            r"<thought>.*?</thought>\s*(Safe|Unsafe)", decoded, re.DOTALL
        ),
        f"Generated text '{decoded}' does not match pattern",
    )

  def test_return_logits(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    max_new_tokens = 4
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=max_new_tokens,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        return_logits=True,
    )
    self.assertIsNotNone(out.logits)
    self.assertEqual(
        out.logits.shape,
        (1, max_new_tokens, self.vocab.GetPieceSize()),
    )
    # logits should be finite floats
    self.assertTrue(jnp.all(jnp.isfinite(out.logits)))

  def test_return_logprobs(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    max_new_tokens = 4
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=max_new_tokens,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=1.0,
        key=jax.random.key(42),
        return_logprobs=True,
    )
    self.assertIsNotNone(out.logprobs)
    self.assertEqual(out.logprobs.shape, (1, max_new_tokens))
    # log-probs should be <= 0
    self.assertTrue(jnp.all(out.logprobs <= 0.0))

  def test_top_p_sampling(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=1.0,
        top_p=0.9,
        key=jax.random.key(42),
    )
    self.assertEqual(out.tokens.shape, (1, 4))

  def test_top_k_sampling(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=1.0,
        top_k=5,
        key=jax.random.key(42),
    )
    self.assertEqual(out.tokens.shape, (1, 4))

  def test_beam_search_output_shape(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    max_new_tokens = 4
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=max_new_tokens,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        beam_size=3,
    )
    # After finalization, output should be [B, max_new_tokens] (best beam).
    self.assertEqual(out.tokens.shape, (1, max_new_tokens))

  def test_beam_search_deterministic(self):
    prompt = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    out1 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        beam_size=3,
    )
    out2 = functional.generate(
        self.model,
        prompt,
        max_new_tokens=4,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
        beam_size=3,
    )
    np.testing.assert_array_equal(out1.tokens, out2.tokens)

  def test_batched_generation(self):
    prompt = jnp.array([[0, 3, 4, 5], [6, 7, 8, 9]], dtype=jnp.int32)
    out = functional.generate(
        self.model,
        prompt,
        max_new_tokens=3,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=32,
        temperature=0.0,
    )
    self.assertEqual(out.tokens.shape, (2, 3))

  def test_prefix_caching_parity(self):
    from tunix.models.gemma4 import model as gemma4_model_lib

    config = gemma4_model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=16,
        hidden_dim=16,
        num_heads=4,
        head_dim=16,
        num_kv_heads=1,
        per_layer_input_dim=16,
        sliding_window_size=16,
        param_dtype=jnp.float32,
        attention_pattern=(
            gemma4_model_lib.AttentionType.GLOBAL,
            gemma4_model_lib.AttentionType.GLOBAL,
        ),
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_key_size=16,
        k_eq_v_global=False,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
    )
    rngs = nnx.Rngs(42)
    gemma_model = gemma4_model_lib.Gemma4(config, rngs=rngs)

    prefix = [5, 6, 7, 8, 9, 10]
    prompt1 = prefix + [11, 12, 13]
    prompt2 = prefix + [14, 15, 16]
    input_ids = jnp.array([prompt1, prompt2], dtype=jnp.int32)

    # 1. Standard generate (no prefix cache)
    out_std = functional.generate(
        gemma_model,
        input_ids,
        max_new_tokens=8,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=64,
        temperature=0.0,
        return_logits=True,
    )

    # 2. Prefill prefix cache
    pfx_cache = functional.prefill_prefix(
        gemma_model,
        prefix,
        cache_size=64,
        dtype=jnp.float32,
    )
    self.assertEqual(pfx_cache.prefix_length, len(prefix))

    # 3. Generate with full input_ids [B, P+S] and prefix_cache
    out_cached = functional.generate(
        gemma_model,
        input_ids,
        max_new_tokens=8,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=64,
        temperature=0.0,
        return_logits=True,
        prefix_cache=pfx_cache,
    )

    np.testing.assert_array_equal(out_std.tokens, out_cached.tokens)
    np.testing.assert_allclose(
        out_std.logits, out_cached.logits, atol=1e-5, rtol=1e-5
    )

    # 4. Generate with suffix-only [B, S] and prefix_cache
    suffix_ids = jnp.array([[11, 12, 13], [14, 15, 16]], dtype=jnp.int32)
    out_suffix = functional.generate(
        gemma_model,
        suffix_ids,
        max_new_tokens=8,
        pad_id=self.pad_id,
        eos_id=self.eos_id,
        cache_size=64,
        temperature=0.0,
        return_logits=True,
        prefix_cache=pfx_cache,
    )

    np.testing.assert_array_equal(out_std.tokens, out_suffix.tokens)
    np.testing.assert_allclose(
        out_std.logits, out_suffix.logits, atol=1e-5, rtol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
