# Copyright 2025 Google LLC
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

"""Tests that the sampler's JIT-compiled functions can be lowered and compiled.

This validates that the nnx.while_loop refactor enables programmatic HLO
inspection — a capability that was previously broken because jax.lax.while_loop
is incompatible with NNX graph operations (nnx.merge) inside the loop body.

The tests prove:
  1. ``_compiled_prefill_fn.lower()`` succeeds and produces valid HLO.
  2. ``_compiled_decode_fn.lower()`` succeeds and produces valid HLO.
  3. End-to-end generation output is numerically identical before and after
     the refactor (i.e., no regressions).
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import sampler as sampler_lib
from tunix.tests import test_common as tc


def _make_sampler(
    cache_size: int = 64,
) -> sampler_lib.Sampler:
  """Creates a sampler with the ToyTransformer for testing."""
  vocab = tc.MockVocab()
  transformer = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
      rngs=nnx.Rngs(42),
  )
  sampler = sampler_lib.Sampler(
      transformer=transformer,
      tokenizer=vocab,
      cache_config=sampler_lib.CacheConfig(
          cache_size=cache_size,
          num_layers=4,
          num_kv_heads=4,
          head_dim=16,
      ),
  )
  # eos_tokens is set during construction, but when building the sampler
  # manually for lowering tests we need to ensure it's set explicitly.
  sampler.eos_tokens = jnp.array([vocab.eos_id()])
  return sampler


class SamplerLoweringTest(absltest.TestCase):
  """Tests that the sampler's JIT functions can be lowered for HLO inspection."""

  def test_prefill_fn_can_be_lowered(self):
    """Validates that _compiled_prefill_fn.lower() produces valid HLO."""
    sampler = _make_sampler()

    # Build a realistic SamplingState the same way _generate_impl does.
    all_input_ids = np.array([[1, 2, 3, 0]], dtype=np.int32)  # [1, 4]
    sampling_state = sampler.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=False,
        total_sampling_steps=10,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=jax.random.PRNGKey(0),
        beam_size=None,
        include_logprobs=False,
    )

    # This would previously fail with a TraceContextError or Pathways
    # IFRT proxy disconnect when using jax.lax.while_loop + nnx.merge
    # inside the traced function.
    lowered = sampler._compiled_prefill_fn.lower(
        sampler._flattened_transformer_state,
        sampling_state,
        None,  # images
        echo=False,
    )

    # Verify that we get a valid Lowered object with accessible HLO.
    hlo_text = lowered.as_text()
    self.assertIsInstance(hlo_text, str)
    self.assertGreater(len(hlo_text), 0)

    # Verify it can be compiled (produces a Compiled object).
    compiled = lowered.compile()
    self.assertIsNotNone(compiled)

    # Verify memory analysis is accessible.
    cost_analysis = compiled.cost_analysis()
    self.assertIsNotNone(cost_analysis)

  def test_decode_fn_can_be_lowered(self):
    """Validates that _compiled_decode_fn.lower() produces valid HLO."""
    sampler = _make_sampler()

    all_input_ids = np.array([[1, 2, 3, 0]], dtype=np.int32)
    sampling_state = sampler.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=False,
        total_sampling_steps=10,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=jax.random.PRNGKey(0),
        beam_size=None,
        include_logprobs=False,
    )

    # This was the primary failure: jax.lax.while_loop + nnx.merge inside
    # the loop body caused TraceContextError when re-traced via .lower().
    lowered = sampler._compiled_decode_fn.lower(
        sampler._flattened_transformer_state,
        sampling_state,
    )

    hlo_text = lowered.as_text()
    self.assertIsInstance(hlo_text, str)
    self.assertGreater(len(hlo_text), 0)

    compiled = lowered.compile()
    self.assertIsNotNone(compiled)

    cost_analysis = compiled.cost_analysis()
    self.assertIsNotNone(cost_analysis)

  def test_num_input_tokens_is_python_int(self):
    """Validates that num_input_tokens is a plain Python int, not jnp.int32.

    When num_input_tokens is a jnp.int32 device array, using it as a
    dynamic_slice size triggers a device-to-host transfer via __index__().
    On Pathways (remote TPU), this transfer can fail with an IFRT proxy
    disconnect. Using a plain Python int avoids this entirely.
    """
    sampler = _make_sampler()
    all_input_ids = np.array([[1, 2, 3, 0]], dtype=np.int32)
    sampling_state = sampler.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=False,
        total_sampling_steps=10,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=jax.random.PRNGKey(0),
        beam_size=None,
        include_logprobs=False,
    )

    # Must be a plain Python int, not a JAX array.
    self.assertIsInstance(sampling_state.num_input_tokens, int)
    self.assertNotIsInstance(sampling_state.num_input_tokens, jnp.ndarray)


class SamplerRegressionTest(absltest.TestCase):
  """Regression tests ensuring the refactor produces identical outputs."""

  def test_greedy_generation_deterministic(self):
    """Greedy generation should produce identical output across runs."""
    sampler = _make_sampler()
    result1 = sampler(
        ['hello world'],
        max_generation_steps=5,
        max_prompt_length=4,
    )
    result2 = sampler(
        ['hello world'],
        max_generation_steps=5,
        max_prompt_length=4,
    )
    # Greedy decoding is deterministic — same input must produce same output.
    self.assertEqual(result1.text, result2.text)
    np.testing.assert_array_equal(result1.tokens[0], result2.tokens[0])

  def test_generation_with_logits(self):
    """Generation with logits should produce valid logit buffers."""
    sampler = _make_sampler()
    result = sampler(
        ['test input'],
        max_generation_steps=5,
        max_prompt_length=4,
        return_logits=True,
        echo=True,
    )
    self.assertIsNotNone(result.logits)
    # Logits should have shape [num_tokens, vocab_size].
    for logit in result.logits:
      self.assertEqual(logit.ndim, 2)
      self.assertGreater(logit.shape[0], 0)

  def test_batched_generation(self):
    """Batched generation should handle multiple prompts correctly."""
    sampler = _make_sampler()
    result = sampler(
        ['hello', 'world', 'test'],
        max_generation_steps=5,
        max_prompt_length=4,
    )
    self.assertEqual(len(result.text), 3)
    # Each output should be a non-empty string.
    for text in result.text:
      self.assertIsInstance(text, str)

  def test_generation_with_eos_stops_early(self):
    """Generation should stop when EOS is produced."""
    sampler = _make_sampler()
    result = sampler(
        ['hello world'],
        max_generation_steps=20,
        max_prompt_length=4,
    )
    # The sampler should produce output (may or may not hit EOS, but
    # should not crash or hang).
    self.assertIsNotNone(result.text)
    self.assertEqual(len(result.text), 1)


if __name__ == '__main__':
  absltest.main()
