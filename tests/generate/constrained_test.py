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

"""Tests for constrained generation integrated into the tunix sampler.

Demonstrates the JIT'd FSM approach end-to-end: a regex pattern is compiled
into a token-level DFA transition table, which is then used inside the
sampler's ``jax.lax.while_loop`` to guarantee that the generated output
conforms to the grammar.

Uses the ``ToyTransformer`` / ``MockVocab`` micro-model setup from
``tunix.tests.test_common`` to exercise real JIT compilation without
requiring a full Gemma checkpoint.
"""

import re

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import constrained
from tunix.generate import sampler as sampler_lib
from tunix.tests import test_common as tc

# ---------------------------------------------------------------------------
# Vocabulary for constrained generation testing
# ---------------------------------------------------------------------------
# We need a vocab that includes the structural and category tokens required
# for JSON-list output, so we extend the default MockVocab mapping.

_CONSTRAINED_VOCAB_MAPPING = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "[": 3,
    "]": 4,
    '"': 5,
    ", ": 6,
    "none": 7,
    "spam": 8,
    "gore": 9,
    "hello": 10,  # distractor — not a valid category
    "world": 11,  # distractor
    "input": 12,
    "string": 13,
    "no": 14,  # partial token (prefix of "none")
    "ne": 15,  # partial token (suffix of "none")
    "apple": 16,
    "banana": 17,
    "pear": 18,
}

_CATEGORIES = ["none", "spam", "gore"]

# The regex pattern that constrains output to JSON category arrays.
# e.g. ["none"], ["spam", "gore"], etc.
_CATEGORY_REGEX = r'\["(none|spam|gore)"(, "(none|spam|gore)")*\]'


def _build_token_id_to_str(
    vocab: tc.MockVocab,
) -> dict[int, str]:
  """Build token_id → decoded_string mapping from a MockVocab."""
  reverse = {v: k for k, v in vocab._mapping_text_to_id.items()}
  result = {}
  for tid, text in reverse.items():
    if text in ("<pad>", "<s>"):
      result[tid] = ""  # structural tokens — empty decode
    elif text == "</s>":
      result[tid] = ""  # EOS — handled specially
    else:
      result[tid] = text
  return result


class _ConcatVocab(tc.MockVocab):
  """MockVocab that concatenates tokens without spaces for JSON output."""

  def DecodeIds(self, ids):
    reverse = {v: k for k, v in self._mapping_text_to_id.items()}
    return "".join(
        reverse[e]
        for e in ids
        if e in reverse and reverse[e] not in ("<pad>", "<s>", "</s>")
    )


class ConstrainedSamplerTest(parameterized.TestCase):
  """Tests that constrained generation works inside the JIT'd sampler."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._cached_vocab = _ConcatVocab(
        mapping_text_to_id=_CONSTRAINED_VOCAB_MAPPING
    )
    cls._cached_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=cls._cached_vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    cls._cached_sampler = sampler_lib.Sampler(
        transformer=cls._cached_transformer,
        tokenizer=cls._cached_vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=128,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    token_id_to_str = _build_token_id_to_str(cls._cached_vocab)
    cls._cached_constraint = constrained.build_regex_constraint(
        pattern=_CATEGORY_REGEX,
        token_id_to_str=token_id_to_str,
        vocab_size=cls._cached_vocab.GetPieceSize(),
        eos_token_ids=[cls._cached_vocab.eos_id()],
    )

  def _make_sampler_and_constraint(self):
    """Returns the cached sampler with ToyTransformer and pre-built constraint tables."""
    return self._cached_sampler, self._cached_vocab, self._cached_constraint

  def test_constrained_generation_produces_valid_output(self):
    """Generated text matches the JSON category list regex."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()

    result = sampler(
        ["input string"],
        max_generation_steps=20,
        constraint=constraint,
    )

    generated_text = result.text[0]
    self.assertRegex(
        generated_text,
        _CATEGORY_REGEX,
        f"Generated text {generated_text!r} does not match the constraint "
        f"regex {_CATEGORY_REGEX!r}",
    )

  def test_constrained_generation_batch(self):
    """Constraint works for batched inputs."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()

    result = sampler(
        ["input string", "hello world"],
        max_generation_steps=20,
        constraint=constraint,
    )

    for i, text in enumerate(result.text):
      self.assertRegex(
          text,
          _CATEGORY_REGEX,
          f"Batch item {i}: {text!r} does not match regex",
      )

  def test_constrained_generation_with_temperature(self):
    """Constraint holds under temperature sampling."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()

    result = sampler(
        ["input string"],
        max_generation_steps=20,
        temperature=1.0,
        seed=42,
        constraint=constraint,
    )

    generated_text = result.text[0]
    self.assertRegex(
        generated_text,
        _CATEGORY_REGEX,
        f"With temperature=1.0: {generated_text!r} does not match regex",
    )

  def test_constrained_generation_unique_items_end_to_end(self):
    """End-to-end generation with uniqueItems array schema produces unique enum items."""
    sampler, vocab, _ = self._make_sampler_and_constraint()
    schema = {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": ["apple", "banana", "pear"]},
    }

    result = sampler(
        ["input string"],
        max_generation_steps=20,
        constraint=schema,
    )

    generated_text = result.text[0]
    import json

    items = json.loads(generated_text)
    self.assertIsInstance(items, list)
    self.assertEqual(
        len(items), len(set(items)), f"Duplicates found in {generated_text!r}"
    )
    for item in items:
      self.assertIn(item, ["apple", "banana", "pear"])

  def test_unconstrained_generation_differs(self):
    """Without constraint, the micro-model does NOT produce valid JSON."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()

    result_unconstrained = sampler(
        ["input string"],
        max_generation_steps=20,
    )

    # The unconstrained micro-model output should NOT match the regex.
    unconstrained_text = result_unconstrained.text[0]
    match = re.fullmatch(_CATEGORY_REGEX, unconstrained_text)
    self.assertIsNone(
        match,
        "Unconstrained output unexpectedly matches regex: "
        f"{unconstrained_text!r}",
    )

  def test_sampler_compile_constraint_pattern(self):
    """Sampler.compile_constraint with pattern and automatic compile via constraint."""
    sampler, vocab, _ = self._make_sampler_and_constraint()

    # Test explicit compile_constraint
    tables = sampler.compile_constraint(_CATEGORY_REGEX)
    self.assertIsInstance(tables, constrained.ConstraintTables)
    self.assertGreater(tables.num_states, 0)

    # Test automatic compilation via constraint
    result = sampler(
        ["input string"],
        max_generation_steps=20,
        constraint=_CATEGORY_REGEX,
    )
    self.assertRegex(result.text[0], _CATEGORY_REGEX)

  def test_sampler_compile_constraint_schema(self):
    """Sampler.compile_constraint with JSON schema."""
    sampler, vocab, _ = self._make_sampler_and_constraint()
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "maxLength": 5},
        },
        "required": ["label"],
    }

    # Test explicit compile_constraint
    tables = sampler.compile_constraint(schema)
    self.assertIsInstance(tables, constrained.ConstraintTables)
    self.assertGreater(tables.num_states, 0)

  def test_sampler_compile_unique_items_constraint(self):
    """compile_constraint with uniqueItems enum array schema."""
    sampler, vocab, _ = self._make_sampler_and_constraint()
    schema = {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": ["apple", "banana", "pear"]},
    }
    tables = sampler.compile_constraint(schema)
    self.assertIsInstance(tables, constrained.ConstraintTables)
    self.assertIsNotNone(tables.unique_items)
    self.assertIsInstance(
        tables.unique_items, constrained.UniqueItemsConstraint
    )
    self.assertGreater(tables.num_states, 0)

  def test_sampler_wildcard_thinking_constraint(self):
    """Sampler with chained wildcard thinking block and category output."""
    vocab_map = dict(_CONSTRAINED_VOCAB_MAPPING)
    vocab_map["<thought>"] = 19
    vocab_map["</thought>"] = 20
    vocab_map["\n"] = 21
    vocab = _ConcatVocab(vocab_map)
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=len(vocab_map)),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    pattern = (
        r"<thought>"
        + constrained.bounded_until(r"</thought>", max_tokens=2)
        + r"\n"
        + _CATEGORY_REGEX
    )
    result = sampler(
        ["input string"],
        max_generation_steps=20,
        constraint=pattern,
    )
    text = result.text[0]
    self.assertTrue(
        re.search(
            r"<thought>.*?</thought>\n" + _CATEGORY_REGEX, text, re.DOTALL
        ),
        f"Output '{text}' did not match thinking + category pattern",
    )

  def test_chain_constraints(self):
    """chain_constraints caches results across identical calls."""
    sampler, vocab, _ = self._make_sampler_and_constraint()
    schema = {"type": "integer"}

    tables1 = sampler.compile_constraint(schema)
    tables2 = sampler.compile_constraint(schema)

    self.assertIs(tables1, tables2)

  def test_diagnose_constraint_tables(self):
    """diagnose_constraint_tables produces valid metadata."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()
    diag = constrained.diagnose_constraint_tables(
        constraint, token_id_to_str=sampler.tokenizer.token_id_to_str
    )
    self.assertEqual(diag["num_states"], constraint.num_states)
    self.assertEqual(diag["vocab_size"], len(constraint.active_tokens))
    self.assertIsInstance(diag["per_state"], list)
    self.assertIsInstance(diag["dead_end_states"], list)

  def test_constraint_tables_metadata(self):
    """Verify constraint tables have expected structure."""
    _, vocab, constraint = self._make_sampler_and_constraint()

    self.assertEqual(
        constraint.token_transitions.shape,
        (constraint.num_states, len(constraint.active_tokens)),
    )
    self.assertEqual(constraint.token_transitions.dtype, np.int32)
    self.assertEqual(constraint.active_tokens.dtype, np.int32)
    self.assertGreater(constraint.num_states, 0)
    self.assertLen(constraint.accept_states, 1)  # single accept state

  def test_deterministic_state_fast_forward_analysis(self):
    """Verify DFA identifies deterministic (single valid token) states for fast forwarding."""
    vocab = {0: "hello ", 1: "world", 2: "there", 3: ""}
    tables = constrained.build_regex_constraint(
        pattern="hello (world|there)",
        token_id_to_str=vocab,
        vocab_size=4,
        eos_token_ids=[3],
    )

    valid_counts = (tables.token_transitions != constrained.INVALID_STATE).sum(
        axis=1
    )

    # Initial state should only accept token 0 ('hello ') -> deterministic state
    initial_valid = valid_counts[tables.initial_state]
    self.assertEqual(initial_valid, 1)

    next_state = tables.token_transitions[tables.initial_state, 0]
    # State after 'hello ' should accept 2 tokens ('world' and 'there') -> branching state
    branch_valid = valid_counts[next_state]
    self.assertEqual(branch_valid, 2)

  def test_shared_prefix_fast_forward_chain(self):
    """Shared prefix literal forms a deterministic fast-forward chain."""
    vocab = {0: "prefix_", 1: "choice_a", 2: "choice_b", 3: ""}
    tables = constrained.build_regex_constraint(
        pattern="prefix_(choice_a|choice_b)",
        token_id_to_str=vocab,
        vocab_size=4,
        eos_token_ids=[3],
    )

    state = tables.initial_state
    # State 0 must deterministically transition on token 0 ('prefix_')
    valid_tids = np.where(
        tables.token_transitions[state] != constrained.INVALID_STATE
    )[0]
    self.assertListEqual(list(valid_tids), [0])

    next_state = tables.token_transitions[state, 0]
    # Next state branches between token 1 and 2
    branch_tids = sorted(
        np.where(
            tables.token_transitions[next_state] != constrained.INVALID_STATE
        )[0]
    )
    self.assertListEqual(branch_tids, [1, 2])

  def test_verify_shared_prefix_fast_forward_branching(self):
    """Char-level DFA verifies deterministic shared prefix and post-branch deterministic fast-forward."""
    pattern = "(hyperparameterization|hyperparameters)"
    nfa_s, nfa_a = constrained._regex_to_nfa(pattern)
    ct, init, acc, ns = constrained._nfa_to_dfa(nfa_s, nfa_a)
    ct, init, acc, ns = constrained._minimize_dfa(ct, init, acc, ns)

    # 1. Trace deterministic prefix 'hyperparameter'
    state = init
    det_prefix_chars: list[str] = []
    while True:
      valid_chars = [
          chr(b)
          for b in range(256)
          if (state, chr(b)) in ct
          and ct[(state, chr(b))] != constrained.INVALID_STATE
      ]
      if len(valid_chars) == 1:
        ch = valid_chars[0]
        det_prefix_chars.append(ch)
        state = ct[(state, ch)]
      else:
        branch_state = state
        branch_chars = sorted(valid_chars)
        break

    self.assertEqual("".join(det_prefix_chars), "hyperparameter")
    self.assertListEqual(branch_chars, ["i", "s"])

    # 2. Branch choice 'i' -> deterministic suffix 'zation' -> accept state
    state_i = ct[(branch_state, "i")]
    suffix_i: list[str] = []
    curr = state_i
    while True:
      valid = [
          chr(b)
          for b in range(256)
          if (curr, chr(b)) in ct
          and ct[(curr, chr(b))] != constrained.INVALID_STATE
      ]
      if len(valid) == 1:
        ch = valid[0]
        suffix_i.append(ch)
        curr = ct[(curr, ch)]
      else:
        break
    self.assertEqual("".join(suffix_i), "zation")
    self.assertIn(curr, acc)

    # 3. Branch choice 's' -> immediate accept state
    state_s = ct[(branch_state, "s")]
    self.assertIn(state_s, acc)

  def test_generate_from_tokens_constrained(self):
    """Constraint works with generate_from_tokens API too."""
    sampler, vocab, constraint = self._make_sampler_and_constraint()

    # Manually tokenize and pad.
    input_ids = np.array(vocab.EncodeAsIds("input string"), dtype=np.int32)
    # Left-pad to length 4.
    padded = np.full((1, 4), vocab.pad_id(), dtype=np.int32)
    padded[0, -len(input_ids) :] = input_ids

    result = sampler.generate_from_tokens(
        input_ids=padded,
        max_generation_steps=20,
        constraint=constraint,
    )

    generated_text = result.text[0]
    self.assertRegex(
        generated_text,
        _CATEGORY_REGEX,
        f"generate_from_tokens: {generated_text!r} does not match regex",
    )


class RegexEngineTest(absltest.TestCase):
  """Tests for the regex → DFA → token-transitions pipeline in isolation."""

  def test_simple_alternation(self):
    """Pattern 'a|b' accepts 'a' and 'b' only."""
    vocab = {0: "a", 1: "b", 2: "c", 3: ""}
    tables = constrained.build_regex_constraint(
        pattern="a|b",
        token_id_to_str=vocab,
        vocab_size=4,
        eos_token_ids=[3],
    )
    tt = tables.token_transitions
    s0 = tables.initial_state

    # 'a' and 'b' should lead to accept states.
    self.assertIn(tt[s0, 0], tables.accept_states)
    self.assertIn(tt[s0, 1], tables.accept_states)
    # 'c' should be invalid.
    self.assertEqual(tt[s0, 2], constrained.INVALID_STATE)

  def test_repetition_star(self):
    """Pattern 'a*' accepts '', 'a', 'aa', etc."""
    vocab = {0: "a", 1: "b", 2: ""}
    tables = constrained.build_regex_constraint(
        pattern="a*",
        token_id_to_str=vocab,
        vocab_size=3,
        eos_token_ids=[2],
    )
    tt = tables.token_transitions
    s0 = tables.initial_state

    # Empty string (EOS from initial) should be valid since a* matches ''.
    self.assertIn(s0, tables.accept_states)
    # 'a' should lead to an accept state.
    s1 = tt[s0, 0]
    self.assertNotEqual(s1, constrained.INVALID_STATE)
    self.assertIn(s1, tables.accept_states)

  def test_json_category_pattern(self):
    """The actual JSON category regex compiles and works."""
    vocab = {
        0: "[",
        1: "]",
        2: '"',
        3: ", ",
        4: "none",
        5: "spam",
        6: "gore",
        7: "hello",  # invalid
        8: "",  # EOS
    }
    tables = constrained.build_regex_constraint(
        pattern=_CATEGORY_REGEX,
        token_id_to_str=vocab,
        vocab_size=9,
        eos_token_ids=[8],
    )
    tt = tables.token_transitions

    # Trace ["spam"] through the transition table.
    state = tables.initial_state
    for tok in [0, 2, 5, 2, 1]:  # [, ", spam, ", ]
      idx = np.searchsorted(tables.active_tokens, tok)
      state = (
          tt[state, idx]
          if idx < len(tables.active_tokens)
          and tables.active_tokens[idx] == tok
          else constrained.INVALID_STATE
      )
      self.assertNotEqual(
          state, constrained.INVALID_STATE, f"Failed at token {tok}"
      )
    self.assertIn(state, tables.accept_states)

    # Trace ["none", "gore"] — multi-category.
    state = tables.initial_state
    for tok in [0, 2, 4, 2, 3, 2, 6, 2, 1]:
      idx = np.searchsorted(tables.active_tokens, tok)
      state = (
          tt[state, idx]
          if idx < len(tables.active_tokens)
          and tables.active_tokens[idx] == tok
          else constrained.INVALID_STATE
      )
      self.assertNotEqual(
          state, constrained.INVALID_STATE, f"Failed at token {tok}"
      )
    self.assertIn(state, tables.accept_states)

    # 'hello' (token 7) should not be an active token
    self.assertNotIn(7, tables.active_tokens)

  def test_verify_fast_forward_multi_char_analysis(self):
    """Verify multi-character fast-forward state identification and transitions."""
    schema = {
        "type": "object",
        "properties": {
            "severity": {"enum": ["low", "medium", "high", "critical"]},
        },
        "required": ["severity"],
    }
    pattern = constrained.json_schema_to_regex(schema)

    # Mock vocabulary with multi-char syntax and category tokens
    vocab_map = {
        0: "<pad>",
        1: "<s>",
        2: "</s>",
        3: '{\n  "severity": "',
        4: "low",
        5: "medium",
        6: "high",
        7: "critical",
        8: '"\n}',
        9: '": "',
        10: "severity",
    }
    tables = constrained.build_regex_constraint(
        pattern=pattern,
        token_id_to_str=vocab_map,
        vocab_size=11,
        eos_token_ids=[2],
    )

    valid_mask = tables.token_transitions != constrained.INVALID_STATE
    ff_map = {}
    for state in range(tables.num_states):
      valid_col_indices = np.where(valid_mask[state])[0]
      valid_tids = [tables.active_tokens[c] for c in valid_col_indices]
      multi_char = [t for t in valid_tids if len(vocab_map.get(int(t), "")) > 1]
      if multi_char:
        best_tid = max(multi_char, key=lambda t: len(vocab_map.get(int(t), "")))
        ff_map[state] = int(best_tid)

    # Asserts multi-char fast-forward states exist
    self.assertGreater(len(ff_map), 0)
    init_s = tables.initial_state
    self.assertIn(init_s, ff_map)
    # Fast forward from initial state advances DFA
    best_col = np.searchsorted(tables.active_tokens, ff_map[init_s])
    next_s = tables.token_transitions[init_s, best_col]
    self.assertNotEqual(next_s, constrained.INVALID_STATE)

  def test_constrained_logits_jit(self):
    """constrained_logits compiles under jax.jit."""
    vocab = {0: "a", 1: "b", 2: ""}
    tables = constrained.build_regex_constraint(
        pattern="a|b",
        token_id_to_str=vocab,
        vocab_size=3,
        eos_token_ids=[2],
    )
    jax_tt = jnp.array(tables.token_transitions)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    logits = jnp.ones((1, 1, 3))
    states = jnp.array([tables.initial_state], dtype=jnp.int32)

    result = jax.jit(constrained.constrained_logits)(
        logits, states, jax_tt, active_toks
    )
    self.assertEqual(result.shape, (1, 1, 3))

  def test_while_loop_compatible(self):
    """Constraint ops work inside jax.lax.while_loop."""
    vocab = {0: "a", 1: "b", 2: ""}
    tables = constrained.build_regex_constraint(
        pattern="ab",
        token_id_to_str=vocab,
        vocab_size=3,
        eos_token_ids=[2],
    )
    jax_tt = jnp.array(tables.token_transitions)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    target = jnp.array([0, 1, 2])  # a, b, EOS
    n = len(target)

    def body(carry):
      state, step, buf = carry
      tok = target[step]
      new_state = constrained.advance_state(
          state[None], tok[None], jax_tt, active_toks
      )[0]
      buf = buf.at[step].set(tok)
      return new_state, step + 1, buf

    def cond(carry):
      _, step, _ = carry
      return step < n

    init = (
        jnp.int32(tables.initial_state),
        jnp.int32(0),
        jnp.zeros(n, dtype=jnp.int32),
    )
    final_state, _, final_buf = jax.jit(
        lambda: jax.lax.while_loop(cond, body, init)
    )()

    self.assertIn(int(final_state), tables.accept_states)
    np.testing.assert_array_equal(np.asarray(final_buf), np.asarray(target))

  def test_bounded_until_helper(self):
    """Test bounded_until helper creates correct regex and DFA bounds."""
    pattern = (
        constrained.bounded_until("</thought>", min_chars=2, max_chars=4)
        + r'\["(none|spam)"\]'
    )
    vocab = {
        0: "a",
        1: "b",
        2: "<",
        3: "/",
        4: "t",
        5: "h",
        6: "o",
        7: "u",
        8: "g",
        9: "h",
        10: "t",
        11: ">",
        12: "[",
        13: '"',
        14: "none",
        15: "spam",
        16: "]",
        17: "",
    }
    tables = constrained.build_regex_constraint(
        pattern=pattern,
        token_id_to_str=vocab,
        vocab_size=18,
        eos_token_ids=[17],
    )

    tt = tables.dense_token_transitions(18)
    s0 = tables.initial_state

    # State 0: min_chars=2 not reached yet.
    # Should accept 'a' (token 0) or 'b' (token 1).
    s1 = tt[s0, 0]
    self.assertNotEqual(s1, constrained.INVALID_STATE)

    # After 2 chars ('a' + 'b'), min length is met, terminal string is allowed.
    s2 = tt[s1, 1]
    self.assertNotEqual(s2, constrained.INVALID_STATE)

  def test_quantifier_unrolling_state_scaling_assertion(self):
    """Assert state count & memory scaling of unrolled quantifiers vs

    un-bounded structural DFA.
    """
    schema_unbounded = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
        },
        "required": ["analysis"],
    }
    pattern_unbounded = (
        constrained.bounded_until("</thought>")
        + "\n"
        + constrained.json_schema_to_regex(schema_unbounded)
    )
    nfa_s, nfa_a = constrained._regex_to_nfa(pattern_unbounded)
    ct, init, acc, ns_unbounded = constrained._nfa_to_dfa(nfa_s, nfa_a)

    # Build unrolled patterns for max_chars = 5, 10, 20
    states_minimized: list[int] = []
    for max_len in [5, 10, 20]:
      schema_bounded = {
          "type": "object",
          "properties": {
              "analysis": {
                  "type": "string",
                  "minLength": 2,
                  "maxLength": max_len,
              },
          },
          "required": ["analysis"],
      }
      pattern_bounded = (
          constrained.bounded_until(
              "</thought>", min_chars=2, max_chars=max_len
          )
          + "\n"
          + constrained.json_schema_to_regex(schema_bounded)
      )
      nfa_s, nfa_a = constrained._regex_to_nfa(pattern_bounded)
      ct, init, acc, n_unmin = constrained._nfa_to_dfa(nfa_s, nfa_a)
      _, _, _, n_min = constrained._minimize_dfa(ct, init, acc, n_unmin)
      states_minimized.append(n_min)

    # 1. Unrolled quantifier DFA states strictly increase with max_chars
    self.assertLess(states_minimized[0], states_minimized[1])
    self.assertLess(states_minimized[1], states_minimized[2])

    # 2. Memory footprint scales with max_len (max_len=20 vs max_len=5)
    mem_mb_5 = states_minimized[0] * 262144 * 4 / (1024**2)
    mem_mb_20 = states_minimized[2] * 262144 * 4 / (1024**2)
    self.assertGreater(mem_mb_20, mem_mb_5)

  def test_json_schema_string_basic(self):
    """Basic string schema produces character class with escaped quote support."""
    pattern = constrained.json_schema_to_regex({"type": "string"})
    self.assertEqual(pattern, r'"([^"\\]|\\.)*"')
    # Validate against DFA.
    self._assert_dfa_accepts(pattern, '"hello"')
    self._assert_dfa_accepts(pattern, '""')
    self._assert_dfa_accepts(pattern, r'"hello \"world\""')
    self._assert_dfa_accepts(pattern, r'"back\\slash"')
    self._assert_dfa_rejects(pattern, "hello")
    self._assert_dfa_rejects(pattern, "42")

  def test_json_schema_string_length(self):
    """minLength / maxLength constrain the character class quantifier."""
    pattern = constrained.json_schema_to_regex(
        {"type": "string", "minLength": 2, "maxLength": 5}
    )
    self.assertEqual(pattern, r'"([^"\\]|\\.){2,5}"')
    self._assert_dfa_accepts(pattern, '"ab"')
    self._assert_dfa_accepts(pattern, '"abcde"')
    self._assert_dfa_rejects(pattern, '"a"')
    self._assert_dfa_rejects(pattern, '"abcdef"')

  def test_json_schema_string_pattern(self):
    """Explicit pattern is passed through (anchors stripped)."""
    pattern = constrained.json_schema_to_regex(
        {"type": "string", "pattern": "^[a-z]+$"}
    )
    self.assertEqual(pattern, '"([a-z]+)"')
    self._assert_dfa_accepts(pattern, '"abc"')
    self._assert_dfa_rejects(pattern, '"ABC"')

  def test_json_schema_string_format_date(self):
    """Known format produces a built-in pattern."""
    pattern = constrained.json_schema_to_regex(
        {"type": "string", "format": "date"}
    )
    self._assert_dfa_accepts(pattern, '"2025-01-15"')
    self._assert_dfa_rejects(pattern, '"not-a-date"')

  def test_json_schema_integer(self):
    pattern = constrained.json_schema_to_regex({"type": "integer"})
    self.assertEqual(pattern, r"-?[0-9]+")
    self._assert_dfa_accepts(pattern, "42")
    self._assert_dfa_accepts(pattern, "-7")
    self._assert_dfa_rejects(pattern, "3.14")

  def test_json_schema_number(self):
    pattern = constrained.json_schema_to_regex({"type": "number"})
    self._assert_dfa_accepts(pattern, "42")
    self._assert_dfa_accepts(pattern, "3.14")
    self._assert_dfa_accepts(pattern, "-1.5")

  def test_json_schema_boolean(self):
    pattern = constrained.json_schema_to_regex({"type": "boolean"})
    self._assert_dfa_accepts(pattern, "true")
    self._assert_dfa_accepts(pattern, "false")
    self._assert_dfa_rejects(pattern, "True")

  def test_json_schema_null(self):
    pattern = constrained.json_schema_to_regex({"type": "null"})
    self._assert_dfa_accepts(pattern, "null")
    self._assert_dfa_rejects(pattern, "None")

  def test_json_schema_enum_strings(self):
    pattern = constrained.json_schema_to_regex({"enum": ["low", "high"]})
    self._assert_dfa_accepts(pattern, '"low"')
    self._assert_dfa_accepts(pattern, '"high"')
    self._assert_dfa_rejects(pattern, '"medium"')

  def test_json_schema_enum_mixed(self):
    """Mixed-type enum serializes each value to its JSON literal."""
    pattern = constrained.json_schema_to_regex({"enum": [1, "x", True, None]})
    self._assert_dfa_accepts(pattern, "1")
    self._assert_dfa_accepts(pattern, '"x"')
    self._assert_dfa_accepts(pattern, "true")
    self._assert_dfa_accepts(pattern, "null")
    self._assert_dfa_rejects(pattern, '"y"')

  def test_json_schema_const(self):
    pattern = constrained.json_schema_to_regex({"const": "fixed"})
    self._assert_dfa_accepts(pattern, '"fixed"')
    self._assert_dfa_rejects(pattern, '"other"')

  def test_json_schema_object_simple(self):
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "boolean"},
        },
        "required": ["x", "y"],
    }
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, '{\n  "x": 42,\n  "y": true\n}')
    self._assert_dfa_rejects(pattern, "{}")

  def test_json_schema_object_nested(self):
    """Nested objects get correct multi-level indentation."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 10},
            "inner": {
                "type": "object",
                "properties": {
                    "val": {"type": "integer"},
                },
            },
        },
    }
    pattern = constrained.json_schema_to_regex(schema)
    expected_json = '{\n  "name": "hi",\n  "inner": {\n    "val": 99\n  }\n}'
    self._assert_dfa_accepts(pattern, expected_json)

  def test_json_schema_array_basic(self):
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "maxItems": 5,
    }
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, "[]")
    self._assert_dfa_accepts(pattern, "[1]")
    self._assert_dfa_accepts(pattern, "[1, 2, 3]")

  def test_json_schema_array_bounds(self):
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 3,
    }
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, "[1, 2]")
    self._assert_dfa_accepts(pattern, "[1, 2, 3]")
    self._assert_dfa_rejects(pattern, "[1]")

  def test_json_schema_array_prefix_items(self):
    """Tuple validation via prefixItems."""
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string", "maxLength": 5},
            {"type": "integer"},
        ],
    }
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, '["hi", 42]')
    self._assert_dfa_rejects(pattern, '[42, "hi"]')

  def test_unique_items_char_dfa_construction(self):
    """Test factored character-level DFA for uniqueItems enum array."""
    choices = ['"apple"', '"banana"', '"cherry"']
    (
        char_trans,
        initial,
        accept,
        num_states,
        comp_map,
        is_bound,
        is_after,
        is_done,
        *_,
    ) = constrained._build_unique_items_char_dfa(
        choices, min_items=1, max_items=3
    )
    # The structural DFA should be compact (~O(n*L) states, not O(2^n))
    self.assertEqual(num_states, 29)
    self.assertEqual(initial, 0)
    self.assertIn(28, accept)

    # Verify item completion tracking for completed items
    completion_entries = [
        (s, comp_map[s]) for s in range(num_states) if comp_map[s] >= 0
    ]
    self.assertEqual(
        sorted(completion_entries),
        [(24, 0), (25, 1), (26, 2)],
    )

    # Simulate matching valid array strings through char_trans
    for test_str in [
        '["apple"]',
        '["apple", "banana"]',
        '["cherry", "apple", "banana"]',
    ]:
      state = initial
      for ch in test_str:
        self.assertIn((state, ch), char_trans)
        state = char_trans[(state, ch)]
      self.assertIn(state, accept)

  def test_unique_items_constraint_enforcement(self):
    """Test factored DFA + bitmask unique-items enforcement at token level."""
    token_map = {
        0: "[",
        1: "]",
        2: ",",
        3: " ",
        4: '"apple"',
        5: '"banana"',
        6: '"cherry"',
    }
    vocab_size = 8

    tables = constrained.build_unique_items_constraint(
        choices=["apple", "banana", "cherry"],
        min_items=2,
        max_items=3,
        token_id_to_str=token_map,
        vocab_size=vocab_size,
        eos_token_ids=[7],
    )
    unique = tables.unique_items
    self.assertEqual(tables.num_states, 29)

    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    unique_state = constrained.init_unique_items_loop_state(
        unique, batch_size=1
    )
    state = jnp.array([tables.initial_state], dtype=jnp.int32)

    # Generate ["apple", and advance state + seen_mask
    for tok in [0, 4, 2, 3]:  # [, "apple", ,, ' '
      state, unique_state = constrained.advance_state_unique(
          state, jnp.array([tok]), tt, active_toks, unique_state
      )
    self.assertEqual(int(unique_state.seen_mask[0]), 0b001)

    # Check logits at the item boundary: "apple" (4) should be blocked,
    # "banana" (5) & "cherry" (6) valid
    logits = jnp.zeros((1, 1, vocab_size))
    masked = constrained.constrained_logits_unique(
        logits, state, tt, active_toks, unique_state
    )
    valid = jnp.where(masked[0, 0] > -jnp.inf)[0].tolist()
    self.assertNotIn(4, valid)
    self.assertIn(5, valid)
    self.assertIn(6, valid)

    # Pick "banana", check that at after-item, ] is allowed (count=2 >= min=2)
    state, unique_state = constrained.advance_state_unique(
        state, jnp.array([5]), tt, active_toks, unique_state
    )
    self.assertEqual(int(unique_state.seen_mask[0]), 0b011)
    masked2 = constrained.constrained_logits_unique(
        jnp.zeros((1, 1, vocab_size)),
        state,
        tt,
        active_toks,
        unique_state,
    )
    valid2 = jnp.where(masked2[0, 0] > -jnp.inf)[0].tolist()
    self.assertIn(1, valid2)
    self.assertIn(2, valid2)

    # Continue with "," and " ", check at next boundary only "cherry" (6) is
    # valid
    state, unique_state = constrained.advance_state_unique(
        state, jnp.array([2]), tt, active_toks, unique_state
    )
    state, unique_state = constrained.advance_state_unique(
        state, jnp.array([3]), tt, active_toks, unique_state
    )
    masked3 = constrained.constrained_logits_unique(
        jnp.zeros((1, 1, vocab_size)),
        state,
        tt,
        active_toks,
        unique_state,
    )
    valid3 = jnp.where(masked3[0, 0] > -jnp.inf)[0].tolist()
    self.assertEqual(valid3, [6])

  def test_unique_items_min_max_bounds_blocking(self):
    """Test explicit blocking of ']' when count < minItems and ',' when count >= maxItems."""
    token_map = {
        0: "[",
        1: "]",
        2: ",",
        3: " ",
        4: '"apple"',
        5: '"banana"',
        6: '"cherry"',
        7: '"durian"',
    }
    vocab_size = 9

    tables = constrained.build_unique_items_constraint(
        choices=["apple", "banana", "cherry", "durian"],
        min_items=2,
        max_items=2,
        token_id_to_str=token_map,
        vocab_size=vocab_size,
        eos_token_ids=[8],
    )
    unique = tables.unique_items
    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    unique_state = constrained.init_unique_items_loop_state(
        unique, batch_size=1
    )
    state = jnp.array([tables.initial_state], dtype=jnp.int32)

    # 1. Advance through "[" and '"apple"' -> count = 1 (< min_items=2)
    for tok in [0, 4]:
      state, unique_state = constrained.advance_state_unique(
          state, jnp.array([tok]), tt, active_toks, unique_state
      )
    self.assertEqual(int(unique_state.seen_mask[0]), 0b0001)

    # At count=1, ']' (1) must be BLOCKED because min_items=2; ',' (2) must be allowed
    masked1 = constrained.constrained_logits_unique(
        jnp.zeros((1, 1, vocab_size)), state, tt, active_toks, unique_state
    )
    valid1 = jnp.where(masked1[0, 0] > -jnp.inf)[0].tolist()
    self.assertNotIn(1, valid1, "']' should be blocked when count < min_items")
    self.assertIn(2, valid1, "',' should be allowed when count < max_items")

    # 2. Advance through "," and " " and '"banana"' -> count = 2 (== max_items=2)
    for tok in [2, 3, 5]:
      state, unique_state = constrained.advance_state_unique(
          state, jnp.array([tok]), tt, active_toks, unique_state
      )
    self.assertEqual(int(unique_state.seen_mask[0]), 0b0011)

    # At count=2, ']' (1) must be ALLOWED (min_items satisfied); ',' (2) must be BLOCKED (max_items reached)
    masked2 = constrained.constrained_logits_unique(
        jnp.zeros((1, 1, vocab_size)), state, tt, active_toks, unique_state
    )
    valid2 = jnp.where(masked2[0, 0] > -jnp.inf)[0].tolist()
    self.assertIn(1, valid2, "']' should be allowed when count >= min_items")
    self.assertNotIn(2, valid2, "',' should be blocked when count >= max_items")

  def test_unique_items_multi_char_tokens_spanning_boundaries(self):
    """Regression: multi-char tokens like '\",' must not bypass uniqueness.

    Subword tokenizers routinely produce tokens that span item boundaries.
    For example, token '\",' contains the closing quote (completing an item)
    and the separator comma.  Token ' \"' contains the space and the opening
    quote of the next item.  The uniqueness enforcement must correctly track
    completions at intermediate character positions within such tokens.
    """
    # Vocabulary with boundary-spanning multi-char tokens.
    token_map = {
        0: "[",
        1: "]",
        2: ",",
        3: " ",
        4: '"',
        5: '",',  # closing quote + comma — spans item completion
        6: ' "',  # space + opening quote — spans separator + boundary
        7: "apple",
        8: "banana",
        9: "cherry",
    }
    vocab_size = 10

    tables = constrained.build_unique_items_constraint(
        choices=["apple", "banana", "cherry"],
        min_items=1,
        max_items=3,
        token_id_to_str=token_map,
        vocab_size=vocab_size,
        eos_token_ids=[],
    )
    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    unique = tables.unique_items
    init_u = constrained.init_unique_items_loop_state(unique, batch_size=1)

    # Simulate: ["apple", "banana", ...
    # Using multi-char tokens: [, ", apple, ",  (token 5 = '",'), ' "' (token 6), banana, ",
    state = jnp.array([tables.initial_state], dtype=jnp.int32)
    u = init_u

    # [ " apple ",  <-- tokens 0, 4, 7, 5 (the last one spans completion+comma)
    for tok in [0, 4, 7, 5]:
      state, u = constrained.advance_state_unique(
          state, jnp.array([tok]), tt, active_toks, u
      )
    # "apple" should be marked as seen even though completion happened mid-token
    self.assertNotEqual(
        int(u.seen_mask[0]),
        0,
        "seen_mask must be updated by multi-char token that spans completion",
    )
    self.assertEqual(
        int(u.seen_mask[0]) & 1, 1, "apple (item 0) should be marked as seen"
    )

    # ' "' (token 6) then banana (token 8) then '",' (token 5)
    for tok in [6, 8, 5]:
      state, u = constrained.advance_state_unique(
          state, jnp.array([tok]), tt, active_toks, u
      )
    self.assertEqual(
        int(u.seen_mask[0]) & 0b011,
        0b011,
        "both apple and banana should be seen",
    )

    # At this point, only "cherry" should be valid.  Verify via logits masking.
    # After '",' we need ' "' (token 6) to start the next item.
    state, u = constrained.advance_state_unique(
        state, jnp.array([6]), tt, active_toks, u
    )
    logits = jnp.zeros((1, 1, vocab_size))
    masked = constrained.constrained_logits_unique(
        logits, state, tt, active_toks, u
    )
    valid = jnp.where(masked[0, 0] > -jnp.inf)[0].tolist()

    # "cherry" (9) should be allowed; "apple" (7) and "banana" (8) should be blocked
    self.assertIn(9, valid, "cherry should be allowed (not yet seen)")
    self.assertNotIn(7, valid, "apple should be blocked (already seen)")
    self.assertNotIn(8, valid, "banana should be blocked (already seen)")

  def test_unique_items_exhaustive_with_multi_char_tokens(self):
    """Exhaustive language test with multi-char tokens spanning boundaries."""
    import itertools
    import json

    choices = ["apple", "banana", "cherry"]
    min_items = 1
    max_items = 3

    expected_strings = set()
    for k in range(min_items, max_items + 1):
      for perm in itertools.permutations(choices, k):
        expected_strings.add(json.dumps(list(perm)))

    # Vocabulary includes boundary-spanning tokens alongside single-char ones.
    token_map = {
        0: "[",
        1: "]",
        2: ",",
        3: " ",
        4: '"',
        5: '",',  # multi-char: completion + separator
        6: ' "',  # multi-char: separator + boundary
        7: '"apple"',
        8: '"banana"',
        9: '"cherry"',
        10: "apple",
        11: "banana",
        12: "cherry",
        13: '", "',  # multi-char: completion + separator + space + boundary
    }
    vocab_size = 14

    tables = constrained.build_unique_items_constraint(
        choices=choices,
        min_items=min_items,
        max_items=max_items,
        token_id_to_str=token_map,
        vocab_size=vocab_size,
        eos_token_ids=[],
    )
    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    init_unique = constrained.init_unique_items_loop_state(
        tables.unique_items, batch_size=1
    )

    accepted_strings = set()
    stack = [(tables.initial_state, init_unique, [])]

    while stack:
      dfa_st, u_state, tok_history = stack.pop()

      if dfa_st in tables.accept_states:
        gen_str = "".join(token_map[t] for t in tok_history)
        accepted_strings.add(gen_str)

      logits = jnp.zeros((1, 1, vocab_size))
      dfa_st_arr = jnp.array([dfa_st], dtype=jnp.int32)
      masked = constrained.constrained_logits_unique(
          logits, dfa_st_arr, tt, active_toks, u_state
      )
      valid_tokens = jnp.where(masked[0, 0] > -jnp.inf)[0].tolist()

      for next_tok in valid_tokens:
        next_dfa_arr, next_u_state = constrained.advance_state_unique(
            dfa_st_arr, jnp.array([next_tok]), tt, active_toks, u_state
        )
        stack.append(
            (int(next_dfa_arr[0]), next_u_state, tok_history + [next_tok])
        )

    self.assertEqual(accepted_strings, expected_strings)

  def test_unique_items_exhaustive_language_simulation(self):
    """Exhaustively traverse all paths in the constraint state-space to prove language equivalence."""
    choices = ["apple", "banana", "cherry"]
    min_items = 1
    max_items = 3

    # Ground truth: all permutations of choices for length 1..3
    import itertools
    import json

    expected_strings = set()
    for k in range(min_items, max_items + 1):
      for perm in itertools.permutations(choices, k):
        expected_strings.add(json.dumps(list(perm)))

    token_map = {
        0: "[",
        1: "]",
        2: ",",
        3: " ",
        4: '"apple"',
        5: '"banana"',
        6: '"cherry"',
    }
    vocab_size = 7

    tables = constrained.build_unique_items_constraint(
        choices=choices,
        min_items=min_items,
        max_items=max_items,
        token_id_to_str=token_map,
        vocab_size=vocab_size,
        eos_token_ids=[],
    )
    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    init_unique = constrained.init_unique_items_loop_state(
        tables.unique_items, batch_size=1
    )
    accept_states = tables.accept_states

    accepted_strings = set()
    stack = [(tables.initial_state, init_unique, [])]

    while stack:
      dfa_st, u_state, tok_history = stack.pop()

      if dfa_st in accept_states:
        gen_str = "".join(token_map[t] for t in tok_history)
        accepted_strings.add(gen_str)

      logits = jnp.zeros((1, 1, vocab_size))
      dfa_st_arr = jnp.array([dfa_st], dtype=jnp.int32)
      masked = constrained.constrained_logits_unique(
          logits, dfa_st_arr, tt, active_toks, u_state
      )
      valid_tokens = jnp.where(masked[0, 0] > -jnp.inf)[0].tolist()

      for next_tok in valid_tokens:
        next_dfa_arr, next_u_state = constrained.advance_state_unique(
            dfa_st_arr, jnp.array([next_tok]), tt, active_toks, u_state
        )
        stack.append(
            (int(next_dfa_arr[0]), next_u_state, tok_history + [next_tok])
        )

    self.assertEqual(accepted_strings, expected_strings)

  def test_unique_items_enum_20_with_thinking(self):
    """Test 20-item enum array with uniqueItems and thinking_terminal prefix."""
    choices_20 = [f"item_{i}" for i in range(20)]
    schema = {
        "type": "array",
        "uniqueItems": True,
        "minItems": 1,
        "maxItems": 20,
        "items": {"enum": choices_20},
    }

    token_map = {0: "<", 1: "/", 2: "t", 3: ">", 4: "\n", 5: "[", 6: "]"}
    for i, item in enumerate(choices_20):
      token_map[7 + i] = f'"{item}"'

    tables = constrained.chain_constraints(
        constraints=[constrained.bounded_until("> ") + "\n", schema],
        token_id_to_str=token_map,
        vocab_size=len(token_map),
        eos_token_ids=[99],
    )
    self.assertIsNotNone(tables.unique_items)
    self.assertEqual(tables.unique_items.num_items, 20)
    self.assertGreater(tables.num_states, 0)

  def test_chain_constraints_multi_stage(self):
    """Test chaining multiple schemas and regex patterns sequentially."""
    stage1 = constrained.bounded_until("> ") + "\n"
    stage2 = {
        "type": "array",
        "uniqueItems": True,
        "minItems": 1,
        "maxItems": 3,
        "items": {"enum": ["apple", "banana", "cherry"]},
    }
    stage3 = "\nEnd"

    token_map = {
        0: "<",
        1: "/",
        2: "t",
        3: ">",
        4: "\n",
        5: "[",
        6: "]",
        7: '"apple"',
        8: '"banana"',
        9: '"cherry"',
        10: "E",
        11: "n",
        12: "d",
    }

    tables = constrained.chain_constraints(
        constraints=[stage1, stage2, stage3],
        token_id_to_str=token_map,
        vocab_size=len(token_map),
        eos_token_ids=[99],
    )

    self.assertIsNotNone(tables.unique_items)
    self.assertEqual(tables.unique_items.num_items, 3)
    self.assertGreater(tables.num_states, 0)
    self.assertGreater(len(tables.accept_states), 0)

  def test_json_schema_anyof(self):
    schema = {
        "anyOf": [{"type": "string", "maxLength": 10}, {"type": "integer"}]
    }
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, '"hello"')
    self._assert_dfa_accepts(pattern, "42")

  def test_json_schema_oneof(self):
    """oneOf is treated identically to anyOf at the regex level."""
    schema = {"oneOf": [{"type": "boolean"}, {"type": "null"}]}
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, "true")
    self._assert_dfa_accepts(pattern, "null")

  def test_json_schema_allof_merge(self):
    """allOf merges constraints before converting."""
    schema = {
        "allOf": [
            {"type": "string"},
            {"maxLength": 5},
        ]
    }
    pattern = constrained.json_schema_to_regex(schema)
    self.assertEqual(pattern, r'"([^"\\]|\\.){0,5}"')

  def test_json_schema_multi_type_nullable(self):
    """type as list (e.g. nullable) produces alternation."""
    schema = {"type": ["string", "null"], "maxLength": 10}
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, '"hi"')
    self._assert_dfa_accepts(pattern, "null")
    self._assert_dfa_rejects(pattern, "42")

  def test_json_schema_empty_object(self):
    """Object with no properties matches empty JSON object."""
    schema = {"type": "object"}
    pattern = constrained.json_schema_to_regex(schema)
    self._assert_dfa_accepts(pattern, "{}")

  def test_json_schema_unsupported_raises(self):
    """Empty schema with no type raises ValueError."""
    with self.assertRaises(ValueError):
      constrained.json_schema_to_regex({})

  def test_progressive_constrained_grammar_levels(self):
    """Test progressive levels of regex grammar complexity (bare alternation -> full schema)."""
    levels = [
        {
            "name": "Level 0: Bare alternation",
            "pattern": "(low|medium|high|critical)",
            "valid": ["low", "medium", "high", "critical"],
            "invalid": ["hello", "LOW", "lo", ""],
        },
        {
            "name": "Level 1: JSON category array",
            "pattern": r'\["(none|spam|gore)"(, "(none|spam|gore)")*\]',
            "valid": [
                '["none"]',
                '["spam", "gore"]',
                '["none", "spam", "gore"]',
            ],
            "invalid": ["[]", '["hello"]', "none", '["none"'],
        },
        {
            "name": "Level 2: Severity enum",
            "pattern": r'\{"severity": "(low|medium|high|critical)"\}',
            "valid": ['{"severity": "low"}', '{"severity": "critical"}'],
            "invalid": ['{"severity": "none"}', '{"severity": low}'],
        },
        {
            "name": "Level 3: Full schema pattern",
            "pattern": (
                constrained.bounded_until("</thought>")
                + "\n"
                + constrained.json_schema_to_regex({
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "maxLength": 15},
                        "severity": {
                            "enum": ["low", "medium", "high", "critical"]
                        },
                    },
                    "required": ["analysis", "severity"],
                })
            ),
            "valid": [
                '<thought>Reviewing.</thought>\n{\n  "analysis": "Insult",\n '
                ' "severity": "high"\n}'
            ],
            "invalid": ['{"severity": "high"}', "just text"],
        },
    ]

    for lvl in levels:
      pattern = lvl["pattern"]
      # Validate DFA string acceptance/rejection across all levels
      for val in lvl["valid"]:
        self._assert_dfa_accepts(pattern, val)
      for inv in lvl["invalid"]:
        self._assert_dfa_rejects(pattern, inv)

  def test_token_surface_additivity_invariant(self):
    """Assert token surface concatenation matches full sequence decode."""
    vocab_map = {
        0: "<pad>",
        1: "<s>",
        2: "</s>",
        3: "{\n  ",
        4: '"analysis": "',
        5: "harmful",
        6: '",\n  ',
        7: '"severity": "',
        8: "high",
        9: '"\n}',
    }
    token_seq = [3, 4, 5, 6, 7, 8, 9]
    concatenated = "".join(vocab_map[t] for t in token_seq)
    expected = '{\n  "analysis": "harmful",\n  "severity": "high"\n}'
    self.assertEqual(concatenated, expected)

  def test_dfa_simulation_equivalence_across_token_boundaries(self):
    """Assert token-by-token DFA simulation reaches same accept state as full string simulation."""
    pattern = r'\{"severity": "(low|medium|high|critical)"\}'
    nfa_s, nfa_a = constrained._regex_to_nfa(pattern)
    ct, init, acc, ns = constrained._nfa_to_dfa(nfa_s, nfa_a)
    ct, init, acc, ns = constrained._minimize_dfa(ct, init, acc, ns)

    token_surfaces = ['{"severity": "', "high", '"}']

    # 1. Full string char-by-char step
    full_str = "".join(token_surfaces)
    state_str = init
    for ch in full_str:
      state_str = constrained._dfa_step(ct, state_str, ch)

    # 2. Token-by-token surface step
    state_tok = init
    for tok_surface in token_surfaces:
      for ch in tok_surface:
        state_tok = constrained._dfa_step(ct, state_tok, ch)

    self.assertEqual(state_str, state_tok)
    self.assertIn(state_tok, acc)

  def _assert_dfa_accepts(self, pattern: str, text: str):
    """Assert the DFA built from *pattern* accepts *text*."""
    nfa_s, nfa_a = constrained._regex_to_nfa(pattern)
    ct, init, acc, _ = constrained._nfa_to_dfa(nfa_s, nfa_a)
    ct, init, acc, _ = constrained._minimize_dfa(ct, init, acc, _)
    self.assertTrue(
        constrained.validate_string(text, ct, init, acc),
        f"DFA should accept {text!r} for pattern {pattern!r}",
    )

  def _assert_dfa_rejects(self, pattern: str, text: str):
    """Assert the DFA built from *pattern* rejects *text*."""
    nfa_s, nfa_a = constrained._regex_to_nfa(pattern)
    ct, init, acc, _ = constrained._nfa_to_dfa(nfa_s, nfa_a)
    ct, init, acc, _ = constrained._minimize_dfa(ct, init, acc, _)
    self.assertFalse(
        constrained.validate_string(text, ct, init, acc),
        f"DFA should reject {text!r} for pattern {pattern!r}",
    )

  def test_freeze_helper(self):
    """_freeze converts mutable dicts/lists into hashable tuples."""
    schema = {
        "b": 2,
        "a": [1, {"c": 3}],
    }
    frozen1 = constrained._freeze(schema)
    frozen2 = constrained._freeze({"a": [1, {"c": 3}], "b": 2})
    self.assertEqual(frozen1, frozen2)
    # Check that frozen key is hashable
    d = {frozen1: "value"}
    self.assertEqual(d[frozen2], "value")

  def test_bounded_until_token_bounds(self):
    """Test bounded_until helper creates correct {{min,max}} token quantifier

    syntax.
    """
    pattern = constrained.bounded_until(
        "</thought>", min_tokens=2, max_tokens=5
    )
    self.assertEqual(pattern, r"(.{{2,5}}</thought>)")

    # Test error if both char and token bounds are provided
    with self.assertRaises(ValueError):
      constrained.bounded_until("</thought>", min_chars=1, min_tokens=1)

  def test_extract_and_strip_token_quantifiers(self):
    """Test extracting {{min,max}} token quantifiers."""
    pattern = r"(.{{10,50}}</thought>)"
    clean_pat, min_t, max_t = constrained._extract_and_strip_token_quantifiers(
        pattern
    )
    self.assertEqual(clean_pat, r"(.*</thought>)")
    self.assertEqual(min_t, 10)
    self.assertEqual(max_t, 50)

  def test_token_bounds_logits_masking(self):
    """Test constrained_logits mask application for min_tokens and max_tokens using TokenBoundsLoopState."""
    # 2 states: state 0 (non-accept), state 1 (accept)
    # Vocab size 2: token 0 -> state 0, token 1 -> state 1 (accept)
    tt = np.array([[0, 1], [1, 1]], dtype=np.int32)
    active_tokens = np.array([0, 1], dtype=np.int32)
    tables = constrained.ConstraintTables(
        active_tokens=active_tokens,
        token_transitions=tt,
        initial_state=0,
        num_states=2,
        accept_states=frozenset([1]),
        min_tokens=2,
        max_tokens=4,
    )
    tables = constrained.prepare_token_bound_masks(tables)

    logits = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    state = jnp.array([0], dtype=jnp.int32)
    active_tokens_jax = jnp.array(active_tokens, dtype=jnp.int32)

    # Step 0 (< min_tokens=2): token 1 (accept transition) should be masked
    bounds_state_0 = constrained.TokenBoundsLoopState(
        count=jnp.array([0], dtype=jnp.int32),
        accept_mask=jnp.array(tables.accept_mask, dtype=jnp.bool_),
        force_accept_mask=jnp.array(tables.force_accept_mask, dtype=jnp.bool_),
        min_tokens=2,
        max_tokens=4,
    )
    masked_0 = constrained.constrained_logits(
        logits, state, tt, active_tokens_jax, token_bounds_state=bounds_state_0
    )
    self.assertTrue(jnp.isinf(masked_0[0, 0, 1]))
    self.assertEqual(masked_0[0, 0, 0], 0.0)

    # Step 4 (>= max_tokens=4): should force accept transition (token 1)
    bounds_state_4 = constrained.TokenBoundsLoopState(
        count=jnp.array([4], dtype=jnp.int32),
        accept_mask=jnp.array(tables.accept_mask, dtype=jnp.bool_),
        force_accept_mask=jnp.array(tables.force_accept_mask, dtype=jnp.bool_),
        min_tokens=2,
        max_tokens=4,
    )
    masked_4 = constrained.constrained_logits(
        logits, state, tt, active_tokens_jax, token_bounds_state=bounds_state_4
    )
    self.assertTrue(jnp.isinf(masked_4[0, 0, 0]))
    self.assertEqual(masked_4[0, 0, 1], 0.0)

  def test_token_bounds_exhaustive_simulation(self):
    """Exhaustively traverse decoding paths to prove min_tokens and max_tokens bounds."""
    # Pattern: .{{2,4}} followed by terminal '>'
    pattern = r"(.{{2,4}}>)"
    token_map = {0: "a", 1: "b", 2: ">"}
    vocab_size = 3
    tables = constrained.build_regex_constraint(
        pattern, token_map, vocab_size, eos_token_ids=[99]
    )

    self.assertEqual(tables.min_tokens, 2)
    self.assertEqual(tables.max_tokens, 4)

    # Simulate generation paths
    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_tokens_jax = jnp.array(tables.active_tokens, dtype=jnp.int32)
    stack: list[tuple[int, constrained.TokenBoundsLoopState, list[int]]] = [
        (tables.initial_state, init_bounds, [])
    ]
    accepted_histories: list[tuple[int, list[int]]] = []

    while stack:
      curr_state, bounds_st, history = stack.pop()
      if curr_state in tables.accept_states:
        count = int(bounds_st.count[0])
        accepted_histories.append((count, history))
        continue
      if bounds_st.count[0] > 10:  # safety ceiling
        continue

      logits = jnp.zeros((1, 1, vocab_size), dtype=jnp.float32)
      c_state = jnp.array([curr_state], dtype=jnp.int32)

      masked = constrained.constrained_logits(
          logits,
          c_state,
          tt,
          active_tokens_jax,
          token_bounds_state=bounds_st,
      )
      valid_next_tokens = jnp.where(masked[0, 0] > -jnp.inf)[0].tolist()

      next_bounds_st = constrained.advance_token_bounds_state(bounds_st)
      for tok in valid_next_tokens:
        tok_idx = int(np.searchsorted(tables.active_tokens, tok))
        next_state = int(tt[curr_state, tok_idx])
        stack.append((next_state, next_bounds_st, history + [tok]))

    # PROOF 1: Every accepted path has at least min_tokens (2)
    for count, hist in accepted_histories:
      self.assertGreaterEqual(
          count, 2, f"Path {hist} accepted early at step {count} < min_tokens=2"
      )

    # PROOF 2: Every accepted path has at most max_tokens + terminal length (4 + 1 = 5)
    for count, hist in accepted_histories:
      self.assertLessEqual(
          count, 5, f"Path {hist} exceeded max_tokens bounds at step {count}"
      )

    # PROOF 3: A 6-token sequence is NEVER accepted (max_tokens=4 forces completion by step 5)
    six_token_paths = [h for count, h in accepted_histories if count >= 6]
    self.assertEqual(
        len(six_token_paths),
        0,
        f"Found accepted 6-token paths: {six_token_paths}",
    )

    # PROOF 4: A 1-token sequence is NEVER accepted (min_tokens=2 forbids completion at step 1)
    one_token_paths = [h for count, h in accepted_histories if count < 2]
    self.assertEqual(
        len(one_token_paths),
        0,
        f"Found accepted 1-token paths: {one_token_paths}",
    )

  def test_chain_bounded_until_with_json_schema(self):
    """Test chaining bounded_until with a JSON Schema array connects stages."""
    vocab_map = {
        0: "a",
        1: "b",
        2: "<thought>",
        3: "</thought>",
        4: "\n",
        5: "[",
        6: "]",
        7: '"',
        8: ",",
    }
    token_id_to_str = {i: vocab_map.get(i, "") for i in range(10)}
    thought_pattern = (
        "<thought>"
        + constrained.bounded_until("</thought>", max_tokens=5)
        + "\n"
    )
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["a", "b"]},
    }
    tables = constrained.chain_constraints(
        [thought_pattern, schema],
        token_id_to_str=token_id_to_str,
        vocab_size=10,
        eos_token_ids=[9],
    )
    self.assertIsNotNone(tables.force_accept_mask)
    # Ensure no unreachable/dead states in the thought stage have 0 force-accept tokens
    has_force = np.any(tables.force_accept_mask, axis=1)
    # The initial state must have force-accept paths leading toward acceptance
    self.assertTrue(has_force[tables.initial_state])

  def test_compact_constraint_tables_properties(self):
    """Test that ConstraintTables is compact by definition."""
    vocab_map = {
        0: "Safe",
        1: "Unsafe",
        2: "Bullying",
        3: "\n",
        4: "<turn|>",
        5: "random",
        6: "token",
    }
    token_id_to_str = {i: vocab_map.get(i, f"tok_{i}") for i in range(100)}
    tables = constrained.build_unique_items_constraint(
        choices=["Safe", "Unsafe", "Bullying"],
        min_items=1,
        max_items=3,
        token_id_to_str=token_id_to_str,
        vocab_size=100,
        eos_token_ids=[4],
    )

    self.assertIsInstance(tables, constrained.ConstraintTables)
    self.assertLess(len(tables.active_tokens), 100)
    self.assertGreater(len(tables.active_tokens), 0)
    self.assertEqual(
        tables.token_transitions.shape,
        (tables.num_states, len(tables.active_tokens)),
    )
    self.assertIsNotNone(tables.unique_items)
    self.assertEqual(
        tables.unique_items.token_completions.shape,
        (tables.num_states, len(tables.active_tokens)),
    )

  def test_wildcard_compaction_memory_and_defaults(self):
    """Verify that wildcard thinking DFA compactor generates compact tables with default_transitions."""
    vocab_map = {
        0: "<thought>",
        1: "</thought>",
        2: "\n",
        3: '["',
        4: '"]',
        5: "Safe",
        6: "Unsafe",
    }
    # Large vocabulary: 500 tokens, but only 7 are structural
    token_id_to_str = {
        i: vocab_map.get(i, f"irrelevant_tok_{i}") for i in range(500)
    }
    pattern = (
        r"<thought>"
        + constrained.bounded_until(r"</thought>")
        + r"\n\[\"(Safe|Unsafe)\"\]"
    )
    tables = constrained.chain_constraints(
        pattern,
        token_id_to_str=token_id_to_str,
        vocab_size=500,
        eos_token_ids=[499],
    )

    # 1. Compaction assertion: active_tokens is tiny (< 25 tokens), NOT 500
    self.assertIsNotNone(tables.default_transitions)
    self.assertLess(len(tables.active_tokens), 25)
    self.assertEqual(
        tables.token_transitions.shape,
        (tables.num_states, len(tables.active_tokens)),
    )
    self.assertEqual(len(tables.default_transitions), tables.num_states)

    # 2. Wildcard states have non-negative default_transitions
    wildcard_states = np.where(
        tables.default_transitions != constrained.INVALID_STATE
    )[0]
    self.assertGreater(len(wildcard_states), 0)

    # 3. Dense transition expansion correctly overlays defaults and explicit entries
    dense = tables.dense_token_transitions(500)
    self.assertEqual(dense.shape, (tables.num_states, 500))
    for ws in wildcard_states:
      def_target = tables.default_transitions[ws]
      # Inactive random token (e.g. 250) should take the default transition
      self.assertEqual(dense[ws, 250], def_target)
      # Structural closing tag token (1: </thought>) should transition to its explicit target
      self.assertNotEqual(dense[ws, 1], constrained.INVALID_STATE)

  def test_jax_wildcard_logits_and_advance_state(self):
    """Verify JIT-compiled constrained_logits and advance_state with default_transitions."""
    vocab_map = {
        0: "<thought>",
        1: "</thought>",
        2: "\n",
        3: "Safe",
        4: "Unsafe",
    }
    vocab_size = 50
    token_id_to_str = {
        i: vocab_map.get(i, f"tok_{i}") for i in range(vocab_size)
    }
    pattern = (
        r"<thought>"
        + constrained.bounded_until(r"</thought>")
        + r"\n(Safe|Unsafe)"
    )
    tables = constrained.chain_constraints(
        pattern,
        token_id_to_str=token_id_to_str,
        vocab_size=vocab_size,
        eos_token_ids=[49],
    )
    self.assertIsNotNone(tables.default_transitions)

    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    def_trans = jnp.array(tables.default_transitions, dtype=jnp.int32)

    # Step 1: Start at initial state (strict state: only <thought>=0 is allowed)
    s0 = jnp.array([tables.initial_state], dtype=jnp.int32)
    dummy_logits = jnp.zeros((1, 1, vocab_size), dtype=jnp.float32)

    masked_s0 = constrained.constrained_logits(
        dummy_logits, s0, tt, active_toks, default_transitions=def_trans
    )
    valid_s0 = jnp.where(masked_s0[0, 0] > -1e9)[0].tolist()
    self.assertIn(0, valid_s0)  # <thought>
    self.assertNotIn(25, valid_s0)  # random token is blocked at strict state

    # Step 2: Advance via <thought> into the wildcard thinking state
    s_think = constrained.advance_state(
        s0,
        jnp.array([0], dtype=jnp.int32),
        tt,
        active_toks,
        default_transitions=def_trans,
    )
    self.assertGreaterEqual(int(s_think[0]), 0)
    self.assertNotEqual(int(s_think[0]), constrained.INVALID_STATE)

    # Step 3: At wildcard state, ALL 50 tokens must be valid (unmasked)
    masked_think = constrained.constrained_logits(
        dummy_logits, s_think, tt, active_toks, default_transitions=def_trans
    )
    valid_think = jnp.where(masked_think[0, 0] > -1e9)[0].tolist()
    self.assertEqual(len(valid_think), vocab_size)  # all tokens allowed!

    # Step 4: Advance via arbitrary inactive token (25) -> stays in wildcard self-loop
    s_after_rand = constrained.advance_state(
        s_think,
        jnp.array([25], dtype=jnp.int32),
        tt,
        active_toks,
        default_transitions=def_trans,
    )
    self.assertEqual(int(s_after_rand[0]), int(def_trans[int(s_think[0])]))

    # Step 5: Advance via explicit </thought> (token 1) -> moves to after-thought state
    s_after_close = constrained.advance_state(
        s_think,
        jnp.array([1], dtype=jnp.int32),
        tt,
        active_toks,
        default_transitions=def_trans,
    )
    self.assertGreaterEqual(int(s_after_close[0]), 0)
    self.assertNotEqual(int(s_after_close[0]), int(s_after_rand[0]))

  def test_chain_constraints_propagates_default_transitions(self):
    """Verify that multi-stage constraint chaining preserves and correctly offsets default_transitions."""
    vocab_map = {
        0: "<thought>",
        1: "</thought>",
        2: "\n",
        3: '["',
        4: '"]',
        5: "item_a",
        6: "item_b",
    }
    token_id_to_str = {i: vocab_map.get(i, f"tok_{i}") for i in range(100)}
    stage1 = r"<thought>" + constrained.bounded_until(r"</thought>") + r"\n"
    stage2 = {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": ["item_a", "item_b"]},
    }
    tables = constrained.chain_constraints(
        [stage1, stage2],
        token_id_to_str=token_id_to_str,
        vocab_size=100,
        eos_token_ids=[99],
    )

    self.assertIsNotNone(tables.default_transitions)
    self.assertEqual(len(tables.default_transitions), tables.num_states)
    self.assertIsNotNone(tables.unique_items)

    # Wildcard states from stage1 must have valid default transitions
    wildcard_states = np.where(
        tables.default_transitions != constrained.INVALID_STATE
    )[0]
    self.assertGreater(len(wildcard_states), 0)
    for ws in wildcard_states:
      # Target state must be within total state count
      self.assertLess(tables.default_transitions[ws], tables.num_states)

  def test_unique_items_with_wildcard_default_transitions(self):
    """Verify JIT unique_items execution with wildcard default_transitions."""
    vocab_map = {
        0: "<thought>",
        1: "</thought>",
        2: "\n",
        3: "[",
        4: "]",
        5: ", ",
        6: '"apple"',
        7: '"banana"',
    }
    vocab_size = 50
    token_id_to_str = {
        i: vocab_map.get(i, f"tok_{i}") for i in range(vocab_size)
    }
    stage1 = r"<thought>" + constrained.bounded_until(r"</thought>") + r"\n"
    stage2 = {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": ["apple", "banana"]},
    }
    tables = constrained.chain_constraints(
        [stage1, stage2],
        token_id_to_str=token_id_to_str,
        vocab_size=vocab_size,
        eos_token_ids=[49],
    )

    tt = jnp.array(tables.token_transitions, dtype=jnp.int32)
    active_toks = jnp.array(tables.active_tokens, dtype=jnp.int32)
    def_trans = jnp.array(tables.default_transitions, dtype=jnp.int32)
    self.assertIsNotNone(tables.unique_items)
    u_state = constrained.init_unique_items_loop_state(
        tables.unique_items, batch_size=1
    )
    dfa_state = jnp.array([tables.initial_state], dtype=jnp.int32)
    dummy_logits = jnp.zeros((1, 1, vocab_size), dtype=jnp.float32)

    # 1. Advance through <thought> (tok 0)
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([0]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    # 2. In thought block, arbitrary token 25 is unmasked
    masked = constrained.constrained_logits_unique(
        dummy_logits,
        dfa_state,
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    valid_tokens = jnp.where(masked[0, 0] > -1e9)[0].tolist()
    self.assertIn(25, valid_tokens)
    # 3. Advance through random token inside thought -> stays valid
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([25]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    self.assertNotEqual(int(dfa_state[0]), constrained.INVALID_STATE)
    # 4. Advance through </thought> (tok 1) and \n (tok 2)
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([1]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([2]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    # 5. Now entering unique_items array: [ (tok 3)
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([3]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    # 6. First item "apple" (tok 6)
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([6]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    # 7. Separator , (tok 5)
    dfa_state, u_state = constrained.advance_state_unique(
        dfa_state,
        jnp.array([5]),
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    # 8. Check logits: "apple" must be blocked (duplicate), "banana" allowed
    masked_after_apple = constrained.constrained_logits_unique(
        dummy_logits,
        dfa_state,
        tt,
        active_toks,
        u_state,
        default_transitions=def_trans,
    )
    valid_after_apple = jnp.where(masked_after_apple[0, 0] > -1e9)[0].tolist()
    self.assertNotIn(6, valid_after_apple)  # "apple" blocked!
    self.assertIn(7, valid_after_apple)  # "banana" allowed!


if __name__ == "__main__":
  absltest.main()
