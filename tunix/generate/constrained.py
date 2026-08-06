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

r"""Regex and JSON Schema constrained (guided) decoding for LLM generation.

This module provides infrastructure for constraining token-level generation to
match a regular expression or JSON Schema dictionary. The approach is:

  1. Convert JSON Schema dicts to regular expressions via
     ``json_schema_to_regex``.
  2. Parse a regex pattern into an NFA via Thompson's construction.
  3. Convert the NFA to a DFA via the subset (powerset) construction.
  4. Compile the character-level DFA into a token-level transition table by
     simulating each vocabulary token's decoded string through the DFA.
  5. At each decoding step inside the JIT'd ``jax.lax.while_loop``, use the
     transition table to mask out tokens that would leave the DFA in an
     invalid state.

Key Public APIs:
  - ``json_schema_to_regex(schema)``: Converts a JSON Schema dictionary to a
    regex string.
  - ``bounded_until(terminal, min_chars, max_chars)``: Matches free-text up to a
    terminal tag.
  - ``build_regex_constraint(pattern, ...)``: Compiles a regex string to token
    transition tables.

Supported regex features:
  - Literal characters
  - Escaped characters: ``\[``, ``\]``, ``\(``, ``\)``, ``\"``, ``\\``
  - Character classes: ``[abc]``, ``[a-z]``, ``[^abc]``
  - Alternation: ``a|b``
  - Grouping: ``(ab)``
  - Repetition: ``*``, ``+``, ``?``, ``{n}``, ``{n,m}``
  - Wildcard: ``.`` (matches any character except newline)

Not supported (intentionally): backreferences, lookahead/lookbehind, anchors.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping, Sequence
import dataclasses
import functools
import json
import re
from typing import Any, Literal, Optional, TypeAlias, TypedDict, cast, overload

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

# Sentinel value indicating that a (state, token) transition is invalid.
INVALID_STATE: int = -1


class _NfaNode:
  """A single state in a Thompson NFA.

  Each node has at most two epsilon transitions and at most one character
  transition (a set of accepted characters → target node).

  Attributes:
      epsilon: List of epsilon-transition targets (at most 2).
      char_transitions: List of ``(charset, target)`` tuples where *charset* is
        a ``frozenset[str]`` of accepted single characters (or the special
        sentinel ``_ANY_CHAR``).
  """

  def __init__(self) -> None:
    self.epsilon: list[_NfaNode] = []
    self.char_transitions: list[tuple[frozenset[str] | str, _NfaNode]] = []

  def add_epsilon(self, target: _NfaNode) -> None:
    """Add an epsilon transition to *target*."""
    self.epsilon.append(target)

  def add_char(self, charset: frozenset[str] | str, target: _NfaNode) -> None:
    """Add a character transition on *charset* to *target*."""
    self.char_transitions.append((charset, target))


# Sentinel used by ``.`` (match any character except newline).
_ANY_CHAR: str = "__ANY__"


def _parse_char_class(pattern: str, pos: int) -> tuple[frozenset[str], int]:
  """Parse a character class ``[...]`` starting *after* the opening ``[``.

  Args:
      pattern: The full regex pattern string.
      pos: Index immediately after the ``[``.

  Returns:
      A tuple ``(charset, new_pos)`` where *charset* is a ``frozenset`` of
      matched characters and *new_pos* is the index after the closing ``]``.

  Raises:
      ValueError: If the character class is malformed.
  """
  negate = False
  if pos < len(pattern) and pattern[pos] == "^":
    negate = True
    pos += 1

  chars: set[str] = set()
  while pos < len(pattern) and pattern[pos] != "]":
    if pattern[pos] == "\\" and pos + 1 < len(pattern):
      chars.add(pattern[pos + 1])
      pos += 2
    elif (
        pos + 2 < len(pattern)
        and pattern[pos + 1] == "-"
        and pattern[pos + 2] != "]"
    ):
      start_char = pattern[pos]
      end_char = pattern[pos + 2]
      for c in range(ord(start_char), ord(end_char) + 1):
        chars.add(chr(c))
      pos += 3
    else:
      chars.add(pattern[pos])
      pos += 1

  if pos >= len(pattern):
    raise ValueError("Unterminated character class in pattern")
  pos += 1  # skip closing ']'

  if negate:
    # Negate against printable ASCII (space through tilde) minus newline.
    all_chars = {chr(c) for c in range(32, 127)}
    chars = all_chars - chars

  return frozenset(chars), pos


def _regex_to_nfa(pattern: str) -> tuple[_NfaNode, _NfaNode]:
  """Convert a regex *pattern* to a Thompson NFA.

  Uses a recursive-descent parser with the following grammar::

      regex   → term ('|' term)*
      term    → factor+
      factor  → atom ('*' | '+' | '?')?
      atom    → '(' regex ')' | char_class | escaped_char | '.' | literal

  Args:
      pattern: The regex pattern string.

  Returns:
      A tuple ``(start, accept)`` representing the NFA's start and sole
      accept state.

  Raises:
      ValueError: If the pattern is malformed.
  """
  pos = 0

  def peek() -> Optional[str]:
    nonlocal pos
    return pattern[pos] if pos < len(pattern) else None

  def advance() -> str:
    nonlocal pos
    ch = pattern[pos]
    pos += 1
    return ch

  def parse_regex() -> tuple[_NfaNode, _NfaNode]:
    """Parse: regex → term ('|' term)*."""
    start, accept = parse_term()
    while peek() == "|":
      advance()  # consume '|'
      t_start, t_accept = parse_term()
      new_start = _NfaNode()
      new_accept = _NfaNode()
      new_start.add_epsilon(start)
      new_start.add_epsilon(t_start)
      accept.add_epsilon(new_accept)
      t_accept.add_epsilon(new_accept)
      start, accept = new_start, new_accept
    return start, accept

  def parse_term() -> tuple[_NfaNode, _NfaNode]:
    """Parse: term → factor+."""
    start, accept = parse_factor()
    while peek() is not None and peek() not in ("|", ")"):
      f_start, f_accept = parse_factor()
      accept.add_epsilon(f_start)
      accept = f_accept
    return start, accept

  def parse_factor() -> tuple[_NfaNode, _NfaNode]:
    """Parse: factor → atom ('*' | '+' | '?' | '{min,max}') '?'?."""
    nonlocal pos
    atom_pos = pos
    a_start, a_accept = parse_atom()
    p = peek()
    if p in ("*", "+", "?"):
      advance()
      if peek() == "?":
        advance()  # consume non-greedy modifier '?'
      if p == "*":
        new_start, new_accept = _NfaNode(), _NfaNode()
        new_start.add_epsilon(a_start)
        new_start.add_epsilon(new_accept)
        a_accept.add_epsilon(a_start)
        a_accept.add_epsilon(new_accept)
        return new_start, new_accept
      elif p == "+":
        new_start, new_accept = _NfaNode(), _NfaNode()
        new_start.add_epsilon(a_start)
        a_accept.add_epsilon(a_start)
        a_accept.add_epsilon(new_accept)
        return new_start, new_accept
      elif p == "?":
        new_start, new_accept = _NfaNode(), _NfaNode()
        new_start.add_epsilon(a_start)
        new_start.add_epsilon(new_accept)
        a_accept.add_epsilon(new_accept)
        return new_start, new_accept
    elif p == "{" and re.match(r"^\{\d+(,\d*)?\}", pattern[pos:]):
      advance()  # consume '{'
      spec = ""
      while peek() is not None and peek() != "}":
        spec += advance()
      advance()  # consume '}'
      if peek() == "?":
        advance()  # consume non-greedy modifier '?'

      # Parse range specification: {n}, {n,m}, {n,}
      parts = spec.split(",")
      if len(parts) == 1:
        min_c = max_c = int(parts[0].strip())
      elif len(parts) == 2:
        min_c = int(parts[0].strip()) if parts[0].strip() else 0
        max_c = int(parts[1].strip()) if parts[1].strip() else None
      else:
        raise ValueError(f"Invalid range quantifier: {{{spec}}}")

      # Re-parse atom to build unrolled NFA fragments
      cur_start, cur_accept = a_start, a_accept
      # Mandatory repetitions (beyond the 1st already parsed)
      for _ in range(min_c - 1):
        pos_backup = pos
        pos = atom_pos
        next_s, next_a = parse_atom()
        pos = pos_backup
        cur_accept.add_epsilon(next_s)
        cur_accept = next_a

      if min_c == 0:
        # 1st atom was optional
        opt_start, opt_accept = _NfaNode(), _NfaNode()
        opt_start.add_epsilon(cur_start)
        opt_start.add_epsilon(opt_accept)
        cur_accept.add_epsilon(opt_accept)
        cur_start, cur_accept = opt_start, opt_accept

      if max_c is None:
        # Unbounded upper limit ({min,}) -> append a '*' fragment
        pos_backup = pos
        pos = atom_pos
        star_s, star_a = parse_atom()
        pos = pos_backup
        loop_s, loop_a = _NfaNode(), _NfaNode()
        loop_s.add_epsilon(star_s)
        loop_s.add_epsilon(loop_a)
        star_a.add_epsilon(star_s)
        star_a.add_epsilon(loop_a)
        cur_accept.add_epsilon(loop_s)
        cur_accept = loop_a
      else:
        # Optional repetitions between min_c and max_c
        for _ in range(max_c - max(min_c, 1)):
          pos_backup = pos
          pos = atom_pos
          opt_s, opt_a = parse_atom()
          pos = pos_backup
          new_a = _NfaNode()
          cur_accept.add_epsilon(opt_s)
          cur_accept.add_epsilon(new_a)
          opt_a.add_epsilon(new_a)
          cur_accept = new_a

      return cur_start, cur_accept

    return a_start, a_accept

  def parse_atom() -> tuple[_NfaNode, _NfaNode]:
    """Parse: atom → '(' regex ')' | '[' class ']' | '.' | escape | lit."""
    nonlocal pos
    ch = peek()
    if ch is None:
      raise ValueError("Unexpected end of pattern")

    if ch == "(":
      advance()  # consume '('
      # Skip non-capturing group prefix '?:' — functionally identical
      # to a regular group since the NFA does not capture.
      if peek() == "?" and pos + 1 < len(pattern) and pattern[pos + 1] == ":":
        advance()  # consume '?'
        advance()  # consume ':'
      start, accept = parse_regex()
      if peek() != ")":
        raise ValueError("Missing closing parenthesis")
      advance()  # consume ')'
      return start, accept

    if ch == "[":
      advance()  # consume '['
      charset, pos = _parse_char_class(pattern, pos)
      start = _NfaNode()
      accept = _NfaNode()
      start.add_char(charset, accept)
      return start, accept

    if ch == ".":
      advance()
      start = _NfaNode()
      accept = _NfaNode()
      start.add_char(_ANY_CHAR, accept)
      return start, accept

    if ch == "\\":
      advance()  # consume '\\'
      escaped = peek()
      if escaped is None:
        raise ValueError("Trailing backslash in pattern")
      advance()

      # Shorthand character classes.
      if escaped == "d":
        charset = frozenset(chr(c) for c in range(ord("0"), ord("9") + 1))
      elif escaped == "w":
        charset = frozenset(
            chr(c)
            for r in (
                range(ord("a"), ord("z") + 1),
                range(ord("A"), ord("Z") + 1),
                range(ord("0"), ord("9") + 1),
                [ord("_")],
            )
            for c in r
        )
      elif escaped == "s":
        charset = frozenset(" \t\n\r\f\v")
      elif escaped == "n":
        charset = frozenset("\n")
      elif escaped == "t":
        charset = frozenset("\t")
      elif escaped == "r":
        charset = frozenset("\r")
      else:
        # Literal escaped character (e.g. \[, \], \", \\, \(, \)).
        charset = frozenset([escaped])

      start = _NfaNode()
      accept = _NfaNode()
      start.add_char(charset, accept)
      return start, accept

    # Plain literal character.
    advance()
    start = _NfaNode()
    accept = _NfaNode()
    start.add_char(frozenset([ch]), accept)
    return start, accept

  start, accept = parse_regex()
  if pos != len(pattern):
    raise ValueError(f"Unexpected character '{pattern[pos]}' at position {pos}")
  return start, accept


def _epsilon_closure(nodes: frozenset[_NfaNode]) -> frozenset[_NfaNode]:
  """Compute the epsilon closure of a set of NFA nodes.

  Args:
      nodes: The seed set of NFA nodes.

  Returns:
      The epsilon closure as a ``frozenset`` of NFA nodes reachable from
      *nodes* via zero or more epsilon transitions.
  """
  stack = list(nodes)
  closure = set(nodes)
  while stack:
    node = stack.pop()
    for target in node.epsilon:
      if target not in closure:
        closure.add(target)
        stack.append(target)
  return frozenset(closure)


def _nfa_to_dfa(
    start: _NfaNode, accept: _NfaNode
) -> tuple[dict[tuple[int, str], int], int, frozenset[int], int]:
  """Convert an NFA (start, accept) to a DFA via subset construction.

  Args:
      start: The NFA start node.
      accept: The NFA accept node.

  Returns:
      A tuple ``(char_transitions, initial_state, accept_states, num_states)``
      where:
        - *char_transitions* is a dict mapping ``(state_id, char)`` to the
          next DFA state id.
        - *initial_state* is the DFA start state id.
        - *accept_states* is a ``frozenset`` of accepting DFA state ids.
        - *num_states* is the total number of DFA states.
  """
  initial_closure = _epsilon_closure(frozenset([start]))

  # Map frozenset[_NfaNode] → int (DFA state id).
  state_map: dict[frozenset[_NfaNode], int] = {initial_closure: 0}
  next_id = 1

  queue: deque[frozenset[_NfaNode]] = deque([initial_closure])
  dfa_transitions: dict[tuple[int, str], int] = {}
  accept_states: set[int] = set()

  if accept in initial_closure:
    accept_states.add(0)

  while queue:
    current = queue.popleft()
    current_id = state_map[current]

    # Gather all possible transitions from this DFA state.
    # Group by character → set of target NFA nodes.
    move: dict[str, set[_NfaNode]] = {}
    for node in current:
      for charset, target in node.char_transitions:
        if charset is _ANY_CHAR:
          # Expand _ANY_CHAR lazily: we defer to individual char
          # matching at DFA lookup time.  Store as sentinel.
          move.setdefault(_ANY_CHAR, set()).add(target)
        elif isinstance(charset, frozenset):
          for ch in charset:
            move.setdefault(ch, set()).add(target)

    # If there is an _ANY_CHAR move, merge its targets into every explicit
    # char move and also keep _ANY_CHAR as a fallback.
    any_targets = move.pop(_ANY_CHAR, None)

    if any_targets is not None:
      # Collect all chars that have explicit moves.
      explicit_chars = set(move.keys())
      for ch in explicit_chars:
        move[ch] = move[ch] | any_targets

      # Register _ANY_CHAR as a catch-all for chars with no explicit
      # entry.  We store it under the sentinel key.
      any_closure = _epsilon_closure(frozenset(any_targets))
      if any_closure not in state_map:
        state_map[any_closure] = next_id
        next_id += 1
        queue.append(any_closure)
        if accept in any_closure:
          accept_states.add(state_map[any_closure])
      dfa_transitions[(current_id, _ANY_CHAR)] = state_map[any_closure]

    for ch, targets in move.items():
      target_closure = _epsilon_closure(frozenset(targets))
      if target_closure not in state_map:
        state_map[target_closure] = next_id
        next_id += 1
        queue.append(target_closure)
        if accept in target_closure:
          accept_states.add(state_map[target_closure])
      dfa_transitions[(current_id, ch)] = state_map[target_closure]

  return dfa_transitions, 0, frozenset(accept_states), next_id


def _minimize_dfa(
    char_transitions: dict[tuple[int, str], int],
    initial_state: int,
    accept_states: frozenset[int],
    num_states: int,
) -> tuple[dict[tuple[int, str], int], int, frozenset[int], int]:
  """Hopcroft's DFA Minimization Algorithm.

  Merges equivalent states in the character-level DFA, dramatically reducing
  state count when unrolled repetitions or wildcard quantifiers exist.

  Args:
      char_transitions: Dict mapping (state, char) -> next_state.
      initial_state: Initial state ID.
      accept_states: Frozenset of accept state IDs.
      num_states: Total number of states.

  Returns:
      Tuple of (min_transitions, min_initial, min_accepts, min_num_states).
  """
  if num_states <= 1:
    return char_transitions, initial_state, accept_states, num_states

  # 1. Partition initial set into Accept and Non-Accept states
  all_states = set(range(num_states))
  non_accepts = all_states - accept_states

  P: list[set[int]] = []
  if accept_states:
    P.append(set(accept_states))
  if non_accepts:
    P.append(non_accepts)

  # Worklist W of sets to split by
  W: list[set[int]] = [set(s) for s in P]

  # Collect all alphabet characters used across all transitions
  alphabet = set(c for _, c in char_transitions.keys())

  # Build inverse transition map: (target_state, char) -> set of source_states
  inv_trans: dict[tuple[int, str], set[int]] = {}
  for (src, c), tgt in char_transitions.items():
    inv_trans.setdefault((tgt, c), set()).add(src)

  while W:
    A = W.pop()
    for c in alphabet:
      # X = set of states that transition into A on character c
      X: set[int] = set()
      for tgt in A:
        X.update(inv_trans.get((tgt, c), ()))

      if not X:
        continue

      new_P: list[set[int]] = []
      for Y in P:
        inter = Y & X
        diff = Y - X
        if inter and diff:
          new_P.append(inter)
          new_P.append(diff)
          if Y in W:
            W.remove(Y)
            W.append(inter)
            W.append(diff)
          else:
            if len(inter) <= len(diff):
              W.append(inter)
            else:
              W.append(diff)
        else:
          new_P.append(Y)
      P = new_P

  # 2. Re-index merged partition blocks to new minimal state IDs
  state_to_min_id: dict[int, int] = {}
  min_initial = 0
  min_accepts: set[int] = set()

  for new_id, block in enumerate(P):
    for s in block:
      state_to_min_id[s] = new_id
    if initial_state in block:
      min_initial = new_id
    if block & accept_states:
      min_accepts.add(new_id)

  # Ensure start state is state 0 by swapping if needed
  if min_initial != 0:
    # Swap mapping so initial is 0
    old_0_block = P[0]
    init_block = P[min_initial]
    P[0], P[min_initial] = init_block, old_0_block
    for s in init_block:
      state_to_min_id[s] = 0
    for s in old_0_block:
      state_to_min_id[s] = min_initial
    min_accepts = {
        0 if s == min_initial else (min_initial if s == 0 else s)
        for s in min_accepts
    }
    min_initial = 0

  min_transitions: dict[tuple[int, str], int] = {}
  for (src, c), tgt in char_transitions.items():
    new_src = state_to_min_id[src]
    new_tgt = state_to_min_id[tgt]
    min_transitions[(new_src, c)] = new_tgt

  return min_transitions, min_initial, frozenset(min_accepts), len(P)


def _dfa_step(
    char_transitions: dict[tuple[int, str], int],
    state: int,
    ch: str,
) -> int:
  """Advance a DFA by one character, returning the next state.

  First checks for an explicit transition on *ch*, then falls back to the
  ``_ANY_CHAR`` wildcard transition. Returns ``INVALID_STATE`` if no
  transition exists.

  Args:
      char_transitions: The DFA transition dict.
      state: The current DFA state.
      ch: The input character.

  Returns:
      The next DFA state, or ``INVALID_STATE`` if the transition is invalid.
  """
  if (state, ch) in char_transitions:
    return char_transitions[(state, ch)]
  if (state, _ANY_CHAR) in char_transitions:
    return char_transitions[(state, _ANY_CHAR)]
  return INVALID_STATE


def _compile_token_transitions(
    char_transitions: dict[tuple[int, str], int],
    num_states: int,
    accept_states: frozenset[int],
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
) -> np.ndarray:
  """Compile character-level DFA transitions into a token-level table.

  Uses vectorized NumPy lookup across states to compile all (state, token)
  transitions in <0.2s even for 256k vocabularies.
  """
  table = np.full((num_states, vocab_size), INVALID_STATE, dtype=np.int32)
  eos_set = set(eos_token_ids)

  # 1. Pre-build dense byte lookup table: char_table[state, byte_ord] -> next_state
  # Handles ASCII/UTF-8 byte values 0..255.
  char_table = np.full((num_states, 256), INVALID_STATE, dtype=np.int32)
  for state in range(num_states):
    for b in range(256):
      ch = chr(b)
      if (state, ch) in char_transitions:
        char_table[state, b] = char_transitions[(state, ch)]
      elif (state, _ANY_CHAR) in char_transitions:
        char_table[state, b] = char_transitions[(state, _ANY_CHAR)]

  # Handle EOS tokens: allowed from accept states
  for state in accept_states:
    for eos_id in eos_set:
      if 0 <= eos_id < vocab_size:
        table[state, eos_id] = state

  # 2. Vectorized simulation across states per token
  all_states = np.arange(num_states, dtype=np.int32)

  for token_id in range(vocab_size):
    if token_id in eos_set:
      continue

    token_str = token_id_to_str.get(token_id)
    if not token_str:
      continue

    token_bytes = token_str.encode("utf-8", errors="replace")
    states = all_states.copy()

    for b in token_bytes:
      # Clamp invalid states to 0 for safe indexing, then restore.
      # Without this, INVALID_STATE (-1) wraps via NumPy negative
      # indexing to char_table[num_states - 1, ...], silently
      # producing bogus transitions for multi-byte tokens.
      invalid_mask = states == INVALID_STATE
      safe_states = np.where(invalid_mask, 0, states)
      states = char_table[safe_states, b]
      states[invalid_mask] = INVALID_STATE
      if (states == INVALID_STATE).all():
        break

    table[:, token_id] = states

  return table


@dataclasses.dataclass(frozen=True, eq=False)
class ConstraintTables:
  """Pre-compiled constraint tables for regex-guided decoding.

  Attributes:
      token_transitions: Array of shape ``(num_states, vocab_size)`` with dtype
        ``int32``. Entry ``[s, t]`` is the DFA state after processing token
        ``t`` from state ``s``, or ``INVALID_STATE`` if the token is forbidden.
      initial_state: The DFA start state.
      num_states: Total number of DFA states.
      accept_states: Set of accepting DFA state ids.
      unique_items: Optional ``UniqueItemsConstraint`` auxiliary metadata when
        enforcing uniqueItems.
      min_tokens: Minimum tokens to generate before accepting state is allowed.
      max_tokens: Maximum tokens allowed after which accepting state is forced.
      accept_mask: Optional boolean array [num_states, vocab_size] indicating
        transitions that enter an accept state.
      force_accept_mask: Optional boolean array [num_states, vocab_size]
        indicating transitions that move closer to an accept state.
  """

  token_transitions: np.ndarray  # [num_states, vocab_size], int32
  initial_state: int
  num_states: int
  accept_states: frozenset[int]
  unique_items: UniqueItemsConstraint | None = None
  min_tokens: int = 0
  max_tokens: int | None = None
  accept_mask: np.ndarray | None = None
  force_accept_mask: np.ndarray | None = None


_TOKEN_QUANTIFIER_RE = re.compile(r"\{\{(\d*)(?:,(\d*))?\}\}")


def _extract_and_strip_token_quantifiers(
    pattern: str,
) -> tuple[str, int, int | None]:
  """Parse token quantifier {{min,max}} from pattern, strip it, and return

  bounds.
  """
  min_tokens = 0
  max_tokens = None

  def _repl(match: re.Match[str]) -> str:
    nonlocal min_tokens, max_tokens
    g1, g2 = match.group(1), match.group(2)
    if g2 is None:  # {{n}}
      val = int(g1) if g1 else 0
      min_tokens = val
      max_tokens = val
    else:  # {{min,max}}, {{min,}}, or {{,max}}
      min_tokens = int(g1) if g1 else 0
      max_tokens = int(g2) if g2 else None
    return "*"

  clean_pattern = _TOKEN_QUANTIFIER_RE.sub(_repl, pattern)
  return clean_pattern, min_tokens, max_tokens


def prepare_token_bound_masks(tables: ConstraintTables) -> ConstraintTables:
  """Precompute accept_mask and force_accept_mask for token bounded decoding."""
  if tables.min_tokens <= 0 and tables.max_tokens is None:
    return tables

  num_states = tables.num_states
  vocab_size = tables.token_transitions.shape[1]
  tt = tables.token_transitions

  # Convert accept_states to a validated 1D NumPy index array
  accept_indices = np.array(list(tables.accept_states), dtype=np.int32)
  if accept_indices.size > 0:
    accept_indices = accept_indices[
        (accept_indices >= 0) & (accept_indices < num_states)
    ]

  # 1. accept_mask: True for (state, token) transitions where target state is an accept_state
  valid_mask = tt != INVALID_STATE
  is_accept = np.zeros(num_states, dtype=bool)
  is_accept[accept_indices] = True
  accept_mask = np.where(
      valid_mask, is_accept[np.clip(tt, 0, num_states - 1)], False
  )

  # 2. Vectorized Bellman-Ford for shortest distance to accept_state
  INF = 999999
  dist = np.full(num_states, INF, dtype=np.int32)
  dist[accept_indices] = 0

  # Extend dist table by 1 to map INVALID_STATE (-1) to index num_states with
  # INF
  dist_with_invalid = np.append(dist, INF)

  changed = True
  while changed:
    dist_next = dist_with_invalid[tt]
    min_step_dist = np.min(dist_next, axis=1) + 1
    min_step_dist[accept_indices] = 0
    new_dist = np.minimum(dist, min_step_dist)
    changed = not np.array_equal(dist, new_dist)
    dist = new_dist
    dist_with_invalid[:num_states] = dist

  # 3. force_accept_mask: True for transitions that decrease distance to
  # accept_state
  target_dist = dist_with_invalid[tt]
  curr_dist = dist[:, None]
  force_accept_mask = valid_mask & (
      (target_dist < curr_dist) | (curr_dist == 0)
  )

  return dataclasses.replace(
      tables,
      accept_mask=accept_mask,
      force_accept_mask=force_accept_mask,
  )


def validate_string(
    s: str,
    char_transitions: dict[tuple[int, str], int],
    initial_state: int,
    accept_states: frozenset[int],
) -> bool:
  """Test whether the character-level DFA accepts a string.

  Useful for verifying the DFA independently of the token compilation.

  Args:
      s: The string to validate.
      char_transitions: The DFA transition dict.
      initial_state: The DFA start state.
      accept_states: Set of accepting DFA states.

  Returns:
      True if the DFA accepts *s*.
  """
  state = initial_state
  for ch in s:
    state = _dfa_step(char_transitions, state, ch)
    if state == INVALID_STATE:
      return False
  return state in accept_states


def diagnose_constraint_tables(
    tables: ConstraintTables,
    token_id_to_str: Mapping[int, str] | None = None,
    max_examples: int = 10,
) -> dict:
  """Diagnose a compiled constraint table for correctness issues.

  Checks for dead-end states, reports transition statistics, and
  identifies which tokens are valid from each state.

  Args:
      tables: The compiled constraint tables.
      token_id_to_str: Optional token-to-string mapping for diagnostics.
      max_examples: Max example tokens to list per state.

  Returns:
      A dict with diagnostic information:
        - ``num_states``: Total DFA states.
        - ``accept_states``: Set of accept states.
        - ``dead_end_states``: States with zero valid tokens (excluding
          accept states where only EOS is valid).
        - ``per_state``: List of dicts with per-state info.
  """
  tt = tables.token_transitions
  num_states, vocab_size = tt.shape

  per_state = []
  dead_end_states = []

  for state in range(num_states):
    row = tt[state]
    valid_mask = row != INVALID_STATE
    valid_count = int(valid_mask.sum())
    valid_tids = list(np.where(valid_mask)[0])

    examples = []
    if token_id_to_str is not None:
      for tid in valid_tids[:max_examples]:
        examples.append((int(tid), token_id_to_str.get(int(tid), "?")))

    is_accept = state in tables.accept_states
    is_dead_end = valid_count == 0 and not is_accept

    if is_dead_end:
      dead_end_states.append(state)

    per_state.append({
        "state": state,
        "valid_tokens": valid_count,
        "is_accept": is_accept,
        "is_dead_end": is_dead_end,
        "examples": examples,
    })

  return {
      "num_states": num_states,
      "vocab_size": vocab_size,
      "accept_states": sorted(tables.accept_states),
      "dead_end_states": dead_end_states,
      "per_state": per_state,
  }


def bounded_until(
    terminal: str,
    min_chars: int = 0,
    max_chars: int | None = None,
    min_tokens: int = 0,
    max_tokens: int | None = None,
) -> str:
  """Build a regex pattern matching arbitrary text until ``terminal``.

  Produces a DFA-compatible pattern. DFAs have no lazy/greedy
  distinction, so we cannot use ``.{0,}?X`` (it behaves identically
  to ``.{0,}X`` in a DFA). Instead:

  * **Single-character terminals** (e.g. ``"``): uses ``[^X]{n}X`` so
    the DFA deterministically stops at the *first* occurrence of the
    terminal character.

  * **Multi-character terminals** (e.g. ``</thought>``): uses
    ``.{n}T`` (dot wildcard). The DFA naturally tracks the terminal
    prefix through distinct states.

  Supports character bounds (``min_chars``/``max_chars`` using ``{n,m}``)
  or token bounds (``min_tokens``/``max_tokens`` using ``{{n,m}}``).

  Args:
      terminal: Stop string (e.g. ``"</thought>"``, ``'"'``).
      min_chars: Minimum characters before ``terminal`` is allowed.
      max_chars: Maximum characters after which ``terminal`` is forced.
      min_tokens: Minimum tokens before ``terminal`` is allowed.
      max_tokens: Maximum tokens after which ``terminal`` is forced.

  Returns:
      Regex pattern string suitable for :func:`build_regex_constraint`.
  """
  if (min_tokens > 0 or max_tokens is not None) and (
      min_chars > 0 or max_chars is not None
  ):
    raise ValueError(
        "Cannot specify both character bounds (min_chars/max_chars) and token"
        " bounds (min_tokens/max_tokens)."
    )

  t_escaped = "".join(
      f"\\{c}" if c in r"[]()*+?.\$^|{}" else c for c in terminal
  )

  if min_tokens > 0 or max_tokens is not None:
    if max_tokens is None:
      quantifier = f"{{{{{min_tokens},}}}}"
    else:
      quantifier = f"{{{{{min_tokens},{max_tokens}}}}}"

    if len(terminal) == 1:
      first_char = terminal[0]
      escaped_first = (
          f"\\{first_char}" if first_char in r"[]()*+?.\$^|{}" else first_char
      )
      return f"([^{escaped_first}]{quantifier}{t_escaped})"
    else:
      return f"(.{quantifier}{t_escaped})"

  if max_chars is None:
    quantifier = f"{{{min_chars},}}"
  else:
    quantifier = f"{{{min_chars},{max_chars}}}"

  if len(terminal) == 1:
    first_char = terminal[0]
    if first_char in r"[]()*+?.\$^|{}":
      escaped_first = f"\\{first_char}"
    else:
      escaped_first = first_char
    return f"([^{escaped_first}]{quantifier}{t_escaped})"
  else:
    return f"(.{quantifier}{t_escaped})"


class JsonSchema(TypedDict, total=False):
  """Supported JSON Schema keywords for regex constrained decoding.

  This TypedDict documents the subset of JSON Schema that
  ``json_schema_to_regex`` can translate into a DFA-compatible regex.
  Unsupported keywords are silently ignored.
  """

  type: str | list[str]
  enum: list[Any]
  const: Any
  properties: dict[str, JsonSchema]
  required: list[str]
  additionalProperties: bool
  items: JsonSchema
  prefixItems: list[JsonSchema]
  minItems: int
  maxItems: int
  uniqueItems: bool
  minLength: int
  maxLength: int
  minTokens: int
  maxTokens: int
  pattern: str
  format: str
  anyOf: list[JsonSchema]
  oneOf: list[JsonSchema]
  allOf: list[JsonSchema]


# Built-in format patterns for {"type": "string", "format": ...}.
_STRING_FORMAT_PATTERNS: dict[str, str] = {
    "date": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
    "time": r"[0-9]{2}:[0-9]{2}:[0-9]{2}",
    "date-time": (
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(Z|[+-][0-9]{2}:[0-9]{2})"
    ),
    "uuid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}" r"-[0-9a-f]{4}-[0-9a-f]{12}",
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
}


def _regex_escape(s: str) -> str:
  """Escape regex metacharacters in a literal string."""
  return "".join(f"\\{ch}" if ch in r"[]()*+?.\$^|{}" else ch for ch in s)


def _value_to_regex(value: Any) -> str:
  """Regex matching the JSON serialization of a single Python value."""
  if value is None:
    return "null"
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, int):
    return _regex_escape(str(value))
  if isinstance(value, float):
    return _regex_escape(json.dumps(value))
  if isinstance(value, str):
    return f'"{_regex_escape(value)}"'
  return _regex_escape(json.dumps(value, separators=(",", ":")))


def _merge_allof(schemas: list[JsonSchema]) -> JsonSchema:
  """Merge a list of schemas for ``allOf`` (intersection of constraints)."""
  merged: JsonSchema = {}
  for schema in schemas:
    schema_copy = schema.copy()
    if properties := schema_copy.pop("properties", None):
      merged.setdefault("properties", {}).update(properties)
    if required := schema_copy.pop("required", None):
      already_required = merged.setdefault("required", [])
      already_required.extend(
          field for field in required if field not in already_required
      )
    merged.update(schema_copy)
  return merged


def _string_schema_to_regex(schema: JsonSchema) -> str:
  """Regex for a ``{"type": "string"}`` schema."""
  # Explicit regex pattern takes priority.
  if "pattern" in schema:
    pat = schema["pattern"]
    pat = pat.lstrip("^").rstrip("$")
    return f'"({pat})"'

  # Known format → built-in pattern.
  if "format" in schema:
    fmt = schema["format"]
    if fmt in _STRING_FORMAT_PATTERNS:
      return f'"({_STRING_FORMAT_PATTERNS[fmt]})"'

  char_pattern = r'([^"\\]|\\.)'

  if "maxTokens" in schema or schema.get("minTokens", 0) > 0:
    min_tok = schema.get("minTokens", 0)
    max_tok = schema.get("maxTokens")
    if max_tok is None:
      quantifier = f"{{{{{min_tok},}}}}"
    else:
      quantifier = f"{{{{{min_tok},{max_tok}}}}}"
    return f'"{char_pattern}{quantifier}"'

  min_len = schema.get("minLength", 0)
  if "maxLength" in schema:
    max_len = schema["maxLength"]
    return f'"{char_pattern}{{{min_len},{max_len}}}"'
  elif min_len > 0:
    return f'"{char_pattern}{{{min_len},}}"'
  else:
    return f'"{char_pattern}*"'


def _object_schema_to_regex(
    schema: JsonSchema,
    indent: int,
    depth: int,
) -> str:
  """Regex for a ``{"type": "object"}`` schema.

  All properties listed in the schema are emitted in deterministic order.
  Truly optional (absent) keys are not supported — optional semantics
  should be expressed via nullable values (``anyOf`` with ``null``).
  """
  properties = schema.get("properties", {})
  if not properties:
    return r"\{\}"

  ind = " " * indent
  inner = ind * (depth + 1)
  outer = ind * depth

  field_lines = []
  for name, prop_schema in properties.items():
    val_pat = _schema_to_regex(prop_schema, indent, depth + 1)
    field_lines.append(f'"{_regex_escape(name)}": {val_pat}')

  body = (",\n" + inner).join(field_lines)
  return "{\n" + inner + body + "\n" + outer + "}"


def _unique_choices_array_regex(
    choices: list[str], min_items: int, max_items: int
) -> str:
  """Build regex matching a JSON array of unique choices from discrete enum

  values.

  The resulting DFA has O(2^n) states where n = len(choices), which is
  inherent to enforcing uniqueness in a regular language.  The regex string
  itself can grow factorially because shared sub-patterns are inlined by
  value.  For large choice sets, prefer ``build_unique_items_constraint``
  which uses a factored DFA + bitmask approach instead.
  """
  n = len(choices)
  max_items = min(max_items, n)
  min_items = min(min_items, n)

  if max_items == 0:
    if min_items > 0:
      raise ValueError(f"Cannot satisfy minItems={min_items} with maxItems=0.")
    return r"\[\]"

  # Memoize by the *set* of remaining choices.  Since min_items and max_items
  # are fixed for the whole call, and items_picked = n - len(rem), the values
  # of ``still_needed`` and ``still_allowed`` are fully determined by the
  # size of ``rem``.  Using frozenset (rather than tuple) avoids spurious
  # key duplication from different element-removal orderings.
  memo: dict[frozenset[str], str] = {}

  def _rec(rem: frozenset[str]) -> str:
    if rem in memo:
      return memo[rem]

    picked = n - len(rem)
    still_needed = max(0, min_items - picked)
    still_allowed = max_items - picked

    if still_allowed <= 0 or not rem:
      memo[rem] = ""
      return ""

    branches: list[str] = []
    for choice in sorted(rem):  # sorted for deterministic regex output
      sub = _rec(rem - {choice})
      if sub:
        if still_needed > 1:
          # Must pick more items after this one to satisfy minItems.
          branches.append(f"{choice}, (?:{sub})")
        else:
          # Already at or past minItems; remaining picks are optional.
          branches.append(f"{choice}(?:, (?:{sub}))?")
      else:
        if still_needed > 1:
          # Can't use this choice as terminal — not enough items yet.
          continue
        branches.append(choice)

    if not branches:
      memo[rem] = ""
      return ""

    res = f"(?:{'|'.join(branches)})"
    memo[rem] = res
    return res

  inner = _rec(frozenset(choices))

  if not inner:
    if min_items == 0:
      return r"\[\]"
    raise ValueError(
        f"Cannot build unique-items regex: minItems={min_items} cannot be "
        f"satisfied with {n} choices and maxItems={max_items}."
    )

  if min_items == 0:
    return f"(\\[\\]|\\[{inner}\\])"
  return f"\\[{inner}\\]"


# ---------------------------------------------------------------------------
# Unique-items constraint: factored DFA + bitmask approach
# ---------------------------------------------------------------------------
# Instead of encoding uniqueness into the DFA (which requires O(2^n) states),
# we split the constraint into:
#   1. A small structural DFA (O(n×L) states) that matches [item, item, ...]
#      without enforcing uniqueness, but tracks WHICH item is being matched.
#   2. A bitmask side-channel (int32 per batch element) tracking which items
#      have been generated, enforced at the sampler level.
#
# This reduces the DFA from O(2^n) to O(n×L) states while correctly
# enforcing uniqueness, minItems, and maxItems.


@dataclasses.dataclass(frozen=True)
class UniqueItemsConstraint:
  """Auxiliary tables for enforcing uniqueItems at the sampler level.

  Used alongside a structural ``ConstraintTables`` that does NOT encode
  uniqueness.  The sampler carries a ``seen_mask`` (int32 bitmask per batch
  element) and uses these tables to apply additional token blocking.

  All token-level tables are indexed by ``(state, token)`` to correctly
  handle multi-character tokens that span item boundaries.  Earlier versions
  used state-level metadata (``is_item_boundary[state]``,
  ``is_after_item[state]``, ``item_completion_map[state]``) which failed when
  subword tokens (e.g. ``\",\"``) jumped over completion or boundary states.

  Attributes:
      token_completions: Shape ``[S, V]``, int32.  Bitmask of items completed
        during the character-by-character simulation of token ``v`` from state
        ``s``.  Captures intermediate completions that are invisible to the
        final-state-only ``item_completion_map``.
      can_lead_to_items: Shape ``[S, V]``, int32.  For each ``(state, token)``
        pair, a bitmask of which items that token could lead to matching from
        that state.  Zero means the token doesn't start any item from that
        state.
      leads_to_close: Shape ``[S, V]``, bool.  True if token ``v`` from state
        ``s`` transitions through or toward closing the array (``]``).
      leads_to_continue: Shape ``[S, V]``, bool.  True if token ``v`` from state
        ``s`` transitions through or toward the separator (``,``).
      min_items: Minimum items required.
      max_items: Maximum items allowed.
      num_items: Number of distinct enum choices.
  """

  token_completions: np.ndarray
  can_lead_to_items: np.ndarray
  leads_to_close: np.ndarray
  leads_to_continue: np.ndarray
  min_items: int
  max_items: int
  num_items: int


@flax.struct.dataclass
class UniqueItemsLoopState:
  """JAX-compatible state for unique-items enforcement inside while_loop.

  Bundles the dynamic ``seen_mask`` with the static lookup tables so they
  can be carried as a single field in the sampling state.

  Attributes:
      seen_mask: Shape ``[B]``, int32 bitmask of consumed items per batch.
      token_completions: Shape ``[S, V]``, int32 bitmask of items completed
        during intermediate character states for each (state, token) pair.
      can_lead_to_items: Shape ``[S, V]``, int32 item-reachability bitmask.
      leads_to_close: Shape ``[S, V]``, bool.
      leads_to_continue: Shape ``[S, V]``, bool.
      min_items: Minimum items required.
      max_items: Maximum items allowed.
  """

  seen_mask: jnp.ndarray  # [B] int32 — dynamic
  token_completions: jnp.ndarray  # [S, V] int32 — static
  can_lead_to_items: jnp.ndarray  # [S, V] int32 — static
  leads_to_close: jnp.ndarray  # [S, V] bool — static
  leads_to_continue: jnp.ndarray  # [S, V] bool — static
  min_items: int = flax.struct.field(pytree_node=False, default=0)
  max_items: int = flax.struct.field(pytree_node=False, default=0)


def init_unique_items_loop_state(
    info: UniqueItemsConstraint,
    batch_size: int,
) -> UniqueItemsLoopState:
  """Create a ``UniqueItemsLoopState`` from compiled constraint metadata.

  Converts NumPy arrays in *info* to JAX arrays and initialises the
  ``seen_mask`` to zeros.

  Args:
      info: Compiled unique-items metadata from
        ``build_unique_items_constraint``.
      batch_size: Number of sequences in the batch.

  Returns:
      A ``UniqueItemsLoopState`` ready for use in the decode loop.
  """
  return UniqueItemsLoopState(
      seen_mask=jnp.zeros((batch_size,), dtype=jnp.int32),
      token_completions=jnp.array(info.token_completions, dtype=jnp.int32),
      can_lead_to_items=jnp.array(info.can_lead_to_items, dtype=jnp.int32),
      leads_to_close=jnp.array(info.leads_to_close, dtype=jnp.bool_),
      leads_to_continue=jnp.array(info.leads_to_continue, dtype=jnp.bool_),
      min_items=info.min_items,
      max_items=info.max_items,
  )


@flax.struct.dataclass
class TokenBoundsLoopState:
  """JAX-compatible state for token-level min/max bounds inside while_loop.

  Bundles the dynamic ``count`` with the static lookup masks and limits so they
  can be carried as a single field in the sampling state.

  Attributes:
      count: Shape ``[B]``, int32 token step count per batch element.
      accept_mask: Shape ``[S, V]``, bool mask for accept-entering transitions.
      force_accept_mask: Shape ``[S, V]``, bool mask for force-accept
        transitions.
      min_tokens: Minimum required tokens.
      max_tokens: Maximum allowed tokens.
  """

  count: jnp.ndarray  # [B] int32 — dynamic
  accept_mask: jnp.ndarray  # [S, V] bool — static
  force_accept_mask: jnp.ndarray  # [S, V] bool — static
  min_tokens: int = flax.struct.field(pytree_node=False, default=0)
  max_tokens: int | None = flax.struct.field(pytree_node=False, default=None)


def init_token_bounds_loop_state(
    tables: ConstraintTables,
    batch_size: int,
) -> TokenBoundsLoopState | None:
  """Create a ``TokenBoundsLoopState`` from compiled ConstraintTables metadata."""
  if tables.min_tokens <= 0 and tables.max_tokens is None:
    return None

  num_states = tables.num_states
  vocab_size = tables.token_transitions.shape[1]
  accept_mask = (
      jnp.array(tables.accept_mask, dtype=jnp.bool_)
      if tables.accept_mask is not None
      else jnp.zeros((num_states, vocab_size), dtype=jnp.bool_)
  )
  force_accept_mask = (
      jnp.array(tables.force_accept_mask, dtype=jnp.bool_)
      if tables.force_accept_mask is not None
      else jnp.zeros((num_states, vocab_size), dtype=jnp.bool_)
  )
  return TokenBoundsLoopState(
      count=jnp.zeros((batch_size,), dtype=jnp.int32),
      accept_mask=accept_mask,
      force_accept_mask=force_accept_mask,
      min_tokens=tables.min_tokens,
      max_tokens=tables.max_tokens,
  )


def advance_token_bounds_state(
    state: TokenBoundsLoopState,
) -> TokenBoundsLoopState:
  """Increment token step count in TokenBoundsLoopState."""
  return TokenBoundsLoopState(
      count=state.count + 1,
      accept_mask=state.accept_mask,
      force_accept_mask=state.force_accept_mask,
      min_tokens=state.min_tokens,
      max_tokens=state.max_tokens,
  )


def _build_unique_items_char_dfa(
    choice_strings: list[str],
    min_items: int,
    max_items: int,
) -> tuple[
    dict[tuple[int, str], int],
    int,
    frozenset[int],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, dict[str, int]],
    dict[int, int],
    int,
    int,
]:
  """Build a character-level DFA for a JSON array of enum items (no uniqueness).

  The DFA matches ``[item(, item)*]`` where ``item`` is any of the choice
  strings, respecting the array structure but NOT enforcing uniqueness or
  min/max bounds.  Those are enforced externally via a bitmask.

  Internally builds a trie over the choice strings to handle shared prefixes
  (all JSON strings start with ``"``).

  Args:
      choice_strings: JSON-serialized choice strings (e.g. ``['"apple"',
        ...]``).
      min_items: Minimum number of items (used only for empty-array handling).
      max_items: Maximum number of items.

  Returns:
      A tuple of DFA components and trie metadata.
  """
  # --- Build trie over choice strings ---
  # Each trie node is an int id.  Children are stored in trie_children.
  # Terminal nodes are recorded in trie_terminal.
  trie_children: dict[int, dict[str, int]] = {}  # node_id -> {char -> child_id}
  trie_terminal: dict[int, int] = {}  # node_id -> item_index
  next_trie_id = 0

  def new_trie_node() -> int:
    nonlocal next_trie_id
    nid = next_trie_id
    trie_children[nid] = {}
    next_trie_id += 1
    return nid

  trie_root = new_trie_node()

  for item_idx, cs in enumerate(choice_strings):
    node = trie_root
    for ch in cs:
      if ch not in trie_children[node]:
        trie_children[node][ch] = new_trie_node()
      node = trie_children[node][ch]
    trie_terminal[node] = item_idx

  trie_size = next_trie_id

  # --- Assign DFA state ids ---
  n = len(choice_strings)
  st_before = 0
  st_boundary = 1
  st_trie_base = 2
  st_after_base = st_trie_base + trie_size
  st_sep_comma = st_after_base + n
  st_done = st_sep_comma + 1
  num_states = st_done + 1

  char_transitions: dict[tuple[int, str], int] = {}
  char_transitions[(st_before, "[")] = st_boundary

  # ITEM_BOUNDARY: transitions from trie root
  for ch, child_trie_id in trie_children[trie_root].items():
    char_transitions[(st_boundary, ch)] = st_trie_base + child_trie_id

  # If min_items == 0, allow ']' from ITEM_BOUNDARY (empty array)
  if min_items == 0:
    char_transitions[(st_boundary, "]")] = st_done

  # IN_ITEM (trie nodes): follow trie transitions
  for trie_node_id in range(trie_size):
    dfa_state = st_trie_base + trie_node_id
    for ch, child_id in trie_children[trie_node_id].items():
      char_transitions[(dfa_state, ch)] = st_trie_base + child_id

  # Redirect transitions that land on terminal trie nodes to AFTER_ITEM_i.
  for src_state_char, dst_state in list(char_transitions.items()):
    if st_trie_base <= dst_state < st_trie_base + trie_size:
      trie_node_id = dst_state - st_trie_base
      if trie_node_id in trie_terminal and not trie_children[trie_node_id]:
        item_idx = trie_terminal[trie_node_id]
        char_transitions[src_state_char] = st_after_base + item_idx

  # Redirect terminal nodes with children (prefix of another item)
  for trie_node_id, item_idx in trie_terminal.items():
    if trie_children[trie_node_id]:
      dfa_state = st_trie_base + trie_node_id
      char_transitions[(dfa_state, ",")] = st_sep_comma
      char_transitions[(dfa_state, "]")] = st_done

  # AFTER_ITEM_i: expect ',' or ']'
  for i in range(n):
    after_state = st_after_base + i
    char_transitions[(after_state, ",")] = st_sep_comma
    char_transitions[(after_state, "]")] = st_done

  # IN_SEP_COMMA: expect ' '
  char_transitions[(st_sep_comma, " ")] = st_boundary

  # --- Build annotation arrays ---
  item_completion_map = np.full(num_states, -1, dtype=np.int32)
  for i in range(n):
    item_completion_map[st_after_base + i] = i
  for trie_node_id, item_idx in trie_terminal.items():
    if trie_children[trie_node_id]:
      item_completion_map[st_trie_base + trie_node_id] = item_idx

  is_item_boundary = np.zeros(num_states, dtype=bool)
  is_item_boundary[st_boundary] = True

  is_after_item = np.zeros(num_states, dtype=bool)
  for i in range(n):
    is_after_item[st_after_base + i] = True
  for trie_node_id in trie_terminal:
    if trie_children[trie_node_id]:
      is_after_item[st_trie_base + trie_node_id] = True

  is_done = np.zeros(num_states, dtype=bool)
  is_done[st_done] = True

  accept_states = frozenset([st_done])

  return (
      char_transitions,
      st_before,
      accept_states,
      num_states,
      item_completion_map,
      is_item_boundary,
      is_after_item,
      is_done,
      trie_children,
      trie_terminal,
      next_trie_id,
      st_trie_base,
  )


def build_unique_items_constraint(
    choices: list[Any],
    min_items: int,
    max_items: int,
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
) -> ConstraintTables:
  """Build constraint tables for a uniqueItems enum array.

  Uses the factored DFA + bitmask approach: a small structural DFA
  (O(n×L) states) handles syntax, while a bitmask side-channel enforces
  uniqueness.  This avoids the O(2^n) state explosion of encoding
  uniqueness into the DFA.

  Args:
      choices: Raw Python enum values.
      min_items: Minimum items in the array.
      max_items: Maximum items in the array.
      token_id_to_str: Mapping from token id to decoded string.
      vocab_size: Vocabulary size.
      eos_token_ids: End-of-sequence token ids.

  Returns:
      A ``ConstraintTables`` object with ``unique_items`` set.
  """
  # Serialize choices to JSON strings.
  choice_strings: list[str] = []
  for v in choices:
    if v is None:
      choice_strings.append("null")
    elif isinstance(v, bool):
      choice_strings.append("true" if v else "false")
    elif isinstance(v, (int, float)):
      choice_strings.append(json.dumps(v))
    elif isinstance(v, str):
      choice_strings.append(f'"{v}"')
    else:
      choice_strings.append(json.dumps(v, separators=(",", ":")))

  n = len(choice_strings)
  max_items = min(max_items, n)
  min_items = min(min_items, n)

  # Build character-level DFA.
  (
      char_transitions,
      initial_state,
      accept_states,
      num_states,
      item_completion_map,
      is_item_boundary,
      is_after_item,
      is_done,
      trie_children,
      trie_terminal,
      next_trie_id,
      st_trie_base,
  ) = _build_unique_items_char_dfa(
      choice_strings,
      min_items,
      max_items,
  )

  # Compile token-level transitions (reuse existing compiler).
  token_transitions = _compile_token_transitions(
      char_transitions,
      num_states,
      accept_states,
      token_id_to_str,
      vocab_size,
      eos_token_ids,
  )

  # Build can_lead_to_items[state, token] -> bitmask of reachable items.
  # For each (state, token) pair, simulate the token from the state and
  # check which items the resulting state is "on the path" of.
  #
  # An item i is "reachable" from a DFA state s if there exists a sequence
  # of characters starting from s that completes item i.  We compute this
  # from the trie structure: for each DFA state corresponding to a trie
  # node, the reachable items are those in the trie subtree.
  #
  # We compute reachable_items[dfa_state] as a bitmask.
  reachable_items = np.zeros(num_states, dtype=np.int32)

  # For AFTER_ITEM_i states, the item is already completed — mark it.
  for s in range(num_states):
    if item_completion_map[s] >= 0:
      reachable_items[s] |= 1 << item_completion_map[s]

  # For trie-based states, compute reachable items bottom-up.
  # We need the trie structure — rebuild the subtree reachability.
  # Process trie nodes in reverse order (children before parents).
  # (This works because child IDs are always greater than parent IDs
  # due to the order we created them.)
  trie_reachable = np.zeros(next_trie_id, dtype=np.int32)
  for trie_node_id, item_idx in trie_terminal.items():
    trie_reachable[trie_node_id] |= 1 << item_idx
  for trie_node_id in range(next_trie_id - 1, -1, -1):
    for child_id in trie_children[trie_node_id].values():
      trie_reachable[trie_node_id] |= trie_reachable[child_id]
    # Map to DFA state (only if not redirected to AFTER_ITEM)
    dfa_state = st_trie_base + trie_node_id
    if dfa_state < num_states:
      reachable_items[dfa_state] |= trie_reachable[trie_node_id]

  # Now build can_lead_to_items[state, token] by looking up reachable_items
  # of the next state.
  can_lead_to_items = np.zeros((num_states, vocab_size), dtype=np.int32)
  for s in range(num_states):
    for v in range(vocab_size):
      next_s = token_transitions[s, v]
      if next_s != INVALID_STATE:
        can_lead_to_items[s, v] = reachable_items[next_s]

  # Build token_completions[state, token] -> bitmask of items completed
  # during intermediate character states when processing token v from state s.
  # This correctly handles multi-character tokens like '","' that span
  # item-completion states.  Without this, completions at intermediate
  # character positions are invisible to the final-state-only lookup.
  char_table = np.full((num_states, 256), INVALID_STATE, dtype=np.int32)
  for state in range(num_states):
    for b in range(256):
      ch = chr(b)
      if (state, ch) in char_transitions:
        char_table[state, b] = char_transitions[(state, ch)]
      elif (state, _ANY_CHAR) in char_transitions:
        char_table[state, b] = char_transitions[(state, _ANY_CHAR)]

  eos_set = set(eos_token_ids)
  token_completions = np.zeros((num_states, vocab_size), dtype=np.int32)
  for token_id in range(vocab_size):
    if token_id in eos_set:
      continue
    token_str = token_id_to_str.get(token_id)
    if not token_str:
      continue
    token_bytes = token_str.encode("utf-8", errors="replace")
    # Simulate character-by-character from each starting state
    for s in range(num_states):
      cur = s
      completions = np.int32(0)
      for byte_val in token_bytes:
        if cur == INVALID_STATE:
          break
        next_cur = int(char_table[cur, byte_val])
        if next_cur != INVALID_STATE and item_completion_map[next_cur] >= 0:
          completions |= np.int32(1) << np.int32(item_completion_map[next_cur])
        cur = next_cur
      if cur != INVALID_STATE:
        token_completions[s, token_id] = completions

  # Build leads_to_close and leads_to_continue at token level.
  # These are computed for ALL states, not just after-item states, because
  # multi-character tokens can span from mid-item through completion to the
  # separator or closing bracket.
  leads_to_close = np.zeros((num_states, vocab_size), dtype=bool)
  leads_to_continue = np.zeros((num_states, vocab_size), dtype=bool)
  for s in range(num_states):
    for v in range(vocab_size):
      next_s = token_transitions[s, v]
      if next_s != INVALID_STATE:
        # A token needs min/max enforcement if:
        # (a) The current state is an after-item state (item was completed
        #     by a previous token, seen_mask already recorded it), OR
        # (b) This token itself completes an item during its character
        #     simulation (token_completions captures this).
        needs_enforcement = is_after_item[s] or token_completions[s, v] != 0
        if needs_enforcement:
          if is_done[next_s]:
            leads_to_close[s, v] = True
          elif next_s != s:
            leads_to_continue[s, v] = True

  unique_info = UniqueItemsConstraint(
      token_completions=token_completions,
      can_lead_to_items=can_lead_to_items,
      leads_to_close=leads_to_close,
      leads_to_continue=leads_to_continue,
      min_items=min_items,
      max_items=max_items,
      num_items=n,
  )

  return ConstraintTables(
      token_transitions=token_transitions,
      initial_state=initial_state,
      num_states=num_states,
      accept_states=accept_states,
      unique_items=unique_info,
  )


def build_constraint_from_schema(
    schema: JsonSchema,
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
    *,
    indent: int = 2,
) -> ConstraintTables:
  """Build constraint tables from a JSON Schema dict."""
  if (
      schema.get("type") == "array"
      and schema.get("uniqueItems", False)
      and "items" in schema
      and "enum" in (items_schema := schema["items"])
  ):
    return build_unique_items_constraint(
        choices=items_schema["enum"],
        min_items=schema.get("minItems", 0),
        max_items=schema.get("maxItems", len(items_schema["enum"])),
        token_id_to_str=token_id_to_str,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
    )
  pattern = json_schema_to_regex(
      schema,
      indent=indent,
  )
  return build_regex_constraint(
      pattern=pattern,
      token_id_to_str=token_id_to_str,
      vocab_size=vocab_size,
      eos_token_ids=eos_token_ids,
  )


def _flatten_constraints(constraints: Constraint) -> list[ConstraintItem]:
  """Recursively flatten nested constraint sequences into a flat list, merging adjacent string patterns."""
  if isinstance(constraints, (dict, str, ConstraintTables)):
    return [constraints]
  if isinstance(constraints, Sequence):
    flat: list[ConstraintItem] = []
    for item in constraints:
      for sub in _flatten_constraints(item):
        if isinstance(sub, str) and flat and isinstance(flat[-1], str):
          flat[-1] = flat[-1] + sub
        else:
          flat.append(sub)
    return flat
  raise TypeError(f"Unsupported constraint type: {type(constraints)}")


def _chain_constraints_uncached(
    constraints: Constraint,
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
    *,
    indent: int = 2,
) -> ConstraintTables:
  """Uncached implementation of chain_constraints."""
  constraint_items = _flatten_constraints(constraints)

  if not constraint_items:
    raise ValueError("constraints list cannot be empty.")

  compiled_stages: list[ConstraintTables] = []
  for constraint_item in constraint_items:
    if isinstance(constraint_item, ConstraintTables):
      compiled_stages.append(constraint_item)
    elif isinstance(constraint_item, dict):
      compiled_stages.append(
          build_constraint_from_schema(
              constraint_item,
              token_id_to_str,
              vocab_size,
              eos_token_ids,
              indent=indent,
          )
      )
    elif isinstance(constraint_item, str):
      compiled_stages.append(
          build_regex_constraint(
              constraint_item, token_id_to_str, vocab_size, eos_token_ids
          )
      )
    else:
      raise TypeError(
          f"Unsupported constraint element type: {type(constraint_item)}"
      )

  if len(compiled_stages) == 1:
    return compiled_stages[0]

  total_states = sum(stage.num_states for stage in compiled_stages)
  offsets = []
  current_offset = 0
  for stage in compiled_stages:
    offsets.append(current_offset)
    current_offset += stage.num_states

  combined_token_transitions = np.full(
      (total_states, vocab_size), INVALID_STATE, dtype=np.int32
  )

  for stage_index, stage in enumerate(compiled_stages):
    offset = offsets[stage_index]
    for state_id in range(stage.num_states):
      for token_id in range(vocab_size):
        next_state = stage.token_transitions[state_id, token_id]
        if next_state != INVALID_STATE:
          combined_token_transitions[offset + state_id, token_id] = (
              offset + next_state
          )

  for stage_index in range(len(compiled_stages) - 1):
    current_stage = compiled_stages[stage_index]
    next_stage = compiled_stages[stage_index + 1]
    current_offset = offsets[stage_index]
    next_offset = offsets[stage_index + 1]

    for accept_state in current_stage.accept_states:
      global_accept_state = current_offset + accept_state
      for token_id in range(vocab_size):
        next_target = next_stage.token_transitions[
            next_stage.initial_state, token_id
        ]
        if next_target != INVALID_STATE:
          combined_token_transitions[global_accept_state, token_id] = (
              next_offset + next_target
          )

  initial_state = offsets[0] + compiled_stages[0].initial_state
  last_stage = compiled_stages[-1]
  last_offset = offsets[-1]
  accept_states = frozenset(
      [last_offset + accept_state for accept_state in last_stage.accept_states]
  )

  unique_info = None
  for stage_index, stage in enumerate(compiled_stages):
    if stage.unique_items is not None:
      unique_items = stage.unique_items
      offset = offsets[stage_index]
      if unique_info is None:
        chained_token_completions = np.zeros(
            (total_states, vocab_size), dtype=np.int32
        )
        chained_can_lead_to_items = np.zeros(
            (total_states, vocab_size), dtype=np.int32
        )
        chained_leads_to_close = np.zeros(
            (total_states, vocab_size), dtype=bool
        )
        chained_leads_to_continue = np.zeros(
            (total_states, vocab_size), dtype=bool
        )
        unique_min_items = unique_items.min_items
        unique_max_items = unique_items.max_items
        unique_num_items = unique_items.num_items
      else:
        chained_token_completions = unique_info.token_completions
        chained_can_lead_to_items = unique_info.can_lead_to_items
        chained_leads_to_close = unique_info.leads_to_close
        chained_leads_to_continue = unique_info.leads_to_continue
        unique_min_items = unique_items.min_items
        unique_max_items = unique_items.max_items
        unique_num_items = unique_items.num_items

      chained_token_completions[offset : offset + stage.num_states, :] = (
          unique_items.token_completions
      )
      chained_can_lead_to_items[offset : offset + stage.num_states, :] = (
          unique_items.can_lead_to_items
      )
      chained_leads_to_close[offset : offset + stage.num_states, :] = (
          unique_items.leads_to_close
      )
      chained_leads_to_continue[offset : offset + stage.num_states, :] = (
          unique_items.leads_to_continue
      )

      unique_info = UniqueItemsConstraint(
          token_completions=chained_token_completions,
          can_lead_to_items=chained_can_lead_to_items,
          leads_to_close=chained_leads_to_close,
          leads_to_continue=chained_leads_to_continue,
          min_items=unique_min_items,
          max_items=unique_max_items,
          num_items=unique_num_items,
      )

  min_tokens = max((stage.min_tokens for stage in compiled_stages), default=0)
  max_tokens_list = [
      stage.max_tokens
      for stage in compiled_stages
      if stage.max_tokens is not None
  ]
  max_tokens = min(max_tokens_list) if max_tokens_list else None

  tables = ConstraintTables(
      token_transitions=combined_token_transitions,
      initial_state=initial_state,
      num_states=total_states,
      accept_states=accept_states,
      unique_items=unique_info,
      min_tokens=min_tokens,
      max_tokens=max_tokens,
  )
  return prepare_token_bound_masks(tables)


ConstraintItem: TypeAlias = str | JsonSchema | ConstraintTables
Constraint: TypeAlias = ConstraintItem | Sequence["Constraint"]

FrozenHashable: TypeAlias = tuple[Literal["hashable"], Hashable]
FrozenDictPair: TypeAlias = tuple[str, "FrozenKey"]
FrozenDict: TypeAlias = tuple[Literal["dict"], tuple[FrozenDictPair, ...]]
FrozenSeq: TypeAlias = tuple[Literal["seq"], tuple["FrozenKey", ...]]
FrozenKey: TypeAlias = FrozenHashable | FrozenDict | FrozenSeq


@overload
def _freeze(obj: Mapping[Any, Any]) -> FrozenDict:
  ...


@overload
def _freeze(obj: list[ConstraintItem]) -> FrozenSeq:
  ...


@overload
def _freeze(obj: Hashable) -> FrozenHashable:
  ...


def _freeze(obj: Any) -> FrozenKey:
  """Recursively converts dicts, lists, and specs into hashable tuples for caching."""
  try:
    hash(obj)
    return ("hashable", obj)
  except TypeError:
    pass

  if isinstance(obj, Mapping):
    return ("dict", tuple(sorted((k, _freeze(v)) for k, v in obj.items())))
  elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
    return ("seq", tuple(_freeze(x) for x in obj))
  raise TypeError(f"Unsupported type for freezing: {type(obj)}")


@overload
def _unfreeze(frozen: FrozenHashable) -> Hashable:
  ...


@overload
def _unfreeze(frozen: FrozenSeq) -> Sequence[ConstraintItem]:
  ...


@overload
def _unfreeze(frozen: FrozenDict) -> JsonSchema:
  ...


def _unfreeze(frozen: FrozenKey) -> Any:
  """Reconstructs original objects from frozen representations."""
  if frozen[0] == "hashable":
    return frozen[1]
  elif frozen[0] == "dict":
    return {k: _unfreeze(v) for k, v in frozen[1]}
  elif frozen[0] == "seq":
    return [_unfreeze(item) for item in frozen[1]]
  return frozen[1]


@functools.lru_cache(maxsize=128)
def _cached_chain_constraints(
    frozen_constraints: FrozenKey,
    frozen_tok_map: FrozenKey,
    vocab_size: int,
    eos_token_ids: tuple[int, ...],
    indent: int,
) -> ConstraintTables:
  constraints = cast(Constraint, _unfreeze(frozen_constraints))
  token_id_to_str = cast(Mapping[int, str], _unfreeze(frozen_tok_map))
  return _chain_constraints_uncached(
      constraints,
      token_id_to_str=token_id_to_str,
      vocab_size=vocab_size,
      eos_token_ids=eos_token_ids,
      indent=indent,
  )


def chain_constraints(
    constraints: Constraint,
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
    *,
    indent: int = 2,
) -> ConstraintTables:
  """Sequentially chains one or multiple schemas, regex patterns, or ConstraintTables together.

  Results are cached with an LRU cache (maxsize=128) keyed on the frozen
  representation of the inputs.  Passing an already-compiled
  ``ConstraintTables`` instance is a no-op passthrough.

  Transitions from the accept state(s) of constraint i directly into the initial
  state of constraint i+1.  Preserves uniqueItems side-channel metadata for any
  stages that have it.

  Args:
      constraints: A single constraint or sequence of JSON Schemas (dicts),
        regex patterns (strs), or pre-compiled ConstraintTables to execute
        sequentially.
      token_id_to_str: Mapping from token id to decoded string.
      vocab_size: Vocabulary size.
      eos_token_ids: End-of-sequence token ids.
      indent: JSON formatting indent level for schema conversion.

  Returns:
      A single chained ConstraintTables object.
  """
  if isinstance(constraints, ConstraintTables):
    return constraints

  frozen_constraints = _freeze(constraints)
  frozen_tok_map = _freeze(token_id_to_str)
  eos_tuple = tuple(eos_token_ids)
  return _cached_chain_constraints(
      frozen_constraints,
      frozen_tok_map,
      vocab_size,
      eos_tuple,
      indent,
  )


def constrained_logits_unique(
    logits: jnp.ndarray,
    constraint_state: jnp.ndarray,
    token_transitions: jnp.ndarray,
    unique_state: UniqueItemsLoopState,
    token_bounds_state: TokenBoundsLoopState | None = None,
) -> jnp.ndarray:
  """Mask logits enforcing both structural DFA and uniqueness constraints.

  All uniqueness metadata is token-level ``[S, V]`` to correctly handle
  multi-character tokens that span item boundaries.  No state-level gates
  are used.

  Pure JAX, safe for ``jax.lax.while_loop``.

  Args:
      logits: Shape ``[B, 1, V]``.
      constraint_state: Shape ``[B]``, current DFA state.
      token_transitions: Shape ``[S, V]``, structural DFA transitions.
      unique_state: Bundled unique-items loop state.
      token_bounds_state: Optional TokenBoundsLoopState carrying dynamic step
        counts and static masks.

  Returns:
      Masked logits, same shape as input.
  """
  # 1. Structural DFA constraint (same as constrained_logits).
  allowed_next = token_transitions[constraint_state]  # [B, V]
  structural_mask = allowed_next != INVALID_STATE  # [B, V]

  if token_bounds_state is not None:
    if token_bounds_state.min_tokens > 0:
      acc_mask_curr = token_bounds_state.accept_mask[constraint_state]
      min_forbid = (
          token_bounds_state.count[:, None] < token_bounds_state.min_tokens
      ) & acc_mask_curr
      structural_mask = structural_mask & (~min_forbid)

    if token_bounds_state.max_tokens is not None:
      force_mask_curr = token_bounds_state.force_accept_mask[constraint_state]
      max_force = (
          token_bounds_state.count[:, None] >= token_bounds_state.max_tokens
      )
      structural_mask = jnp.where(
          max_force, structural_mask & force_mask_curr, structural_mask
      )

  # 2. Uniqueness constraint — token-level, no state-level gates.
  # Block tokens that can ONLY lead to already-seen items.
  cli = unique_state.can_lead_to_items[constraint_state]  # [B, V] bitmasks
  # remaining = items this token can lead to that are NOT yet seen
  remaining = cli & ~unique_state.seen_mask[:, None]  # [B, V]
  # Block if: token leads to some item AND no unseen item reachable
  uniqueness_block = (cli != 0) & (remaining == 0)

  # 3. Min/max enforcement — token-level, no state-level gates.
  count = jax.lax.population_count(unique_state.seen_mask)  # [B]
  ltc = unique_state.leads_to_close[constraint_state]  # [B, V]
  ltk = unique_state.leads_to_continue[constraint_state]  # [B, V]
  # Block close-path tokens if count < min_items
  close_block = ltc & (count < unique_state.min_items)[:, None]
  # Block continue-path tokens if count >= max_items
  continue_block = ltk & (count >= unique_state.max_items)[:, None]

  # Combine all masks.
  final_mask = (
      structural_mask & ~uniqueness_block & ~close_block & ~continue_block
  )
  return jnp.where(final_mask[:, None, :], logits, -jnp.inf)


def advance_state_unique(
    constraint_state: jnp.ndarray,
    next_token: jnp.ndarray,
    token_transitions: jnp.ndarray,
    unique_state: UniqueItemsLoopState,
) -> tuple[jnp.ndarray, UniqueItemsLoopState]:
  """Advance the DFA state and update the unique-items seen_mask.

  Uses ``token_completions[state, token]`` to capture item completions
  that occur at intermediate character positions within multi-character
  tokens.

  Pure JAX, safe for ``jax.lax.while_loop``.

  Args:
      constraint_state: Shape ``[B]``, current DFA state.
      next_token: Shape ``[B]``, selected token id.
      token_transitions: Shape ``[S, V]``, DFA transition table.
      unique_state: Bundled unique-items loop state.

  Returns:
      Tuple of (new_constraint_state, updated_unique_state).
  """
  new_state = token_transitions[constraint_state, next_token]  # [B]

  # Look up the bitmask of items completed during this token's character
  # simulation (including intermediate states).
  completed_bitmask = unique_state.token_completions[
      constraint_state, next_token
  ]  # [B]
  new_seen = unique_state.seen_mask | completed_bitmask

  return new_state, dataclasses.replace(unique_state, seen_mask=new_seen)


def _array_schema_to_regex(
    schema: JsonSchema,
    indent: int,
    depth: int,
) -> str:
  """Regex for a ``{"type": "array"}`` schema (compact single-line format)."""
  # Tuple validation: prefixItems.
  if "prefixItems" in schema:
    item_pats = [
        _schema_to_regex(s, indent, depth) for s in schema["prefixItems"]
    ]
    return r"\[" + ", ".join(item_pats) + r"\]"

  items_schema = schema.get("items", {"type": "string"})
  min_items = schema.get("minItems", 0)

  if schema.get("uniqueItems", False):
    if "enum" in items_schema:
      choices = [_value_to_regex(v) for v in items_schema["enum"]]
      max_items = schema.get("maxItems", len(items_schema["enum"]))
      return _unique_choices_array_regex(choices, min_items, max_items)
    raise ValueError(
        "uniqueItems: True in json_schema_to_regex is currently only "
        "supported when items is an enum of discrete values."
    )

  item_pat = _schema_to_regex(items_schema, indent, depth)

  if "maxItems" in schema:
    max_items = schema["maxItems"]
    if min_items == 0:
      if max_items == 0:
        return r"\[\]"
      # Can be empty or have 1..max_items elements.
      inner = f"{item_pat}(, {item_pat}){{0,{max_items - 1}}}"
      return f"(\\[\\]|\\[{inner}\\])"
    else:
      # First item mandatory, then (min-1) more mandatory, then optional.
      mandatory_extra = min_items - 1
      optional_extra = max_items - min_items
      parts = item_pat
      if mandatory_extra > 0:
        parts += f"(, {item_pat}){{{mandatory_extra}}}"
      if optional_extra > 0:
        parts += f"(, {item_pat}){{0,{optional_extra}}}"
      return f"\\[{parts}\\]"
  else:
    if min_items == 0:
      inner = f"{item_pat}(, {item_pat})*"
      return f"(\\[\\]|\\[{inner}\\])"
    else:
      mandatory_extra = min_items - 1
      parts = item_pat
      if mandatory_extra > 0:
        parts += f"(, {item_pat}){{{mandatory_extra}}}"
      parts += f"(, {item_pat})*"
      return f"\\[{parts}\\]"


def _schema_to_regex(
    schema: JsonSchema,
    indent: int,
    depth: int,
) -> str:
  """Recursive core: convert a JSON Schema dict to a regex pattern."""

  if "const" in schema:
    return _value_to_regex(schema["const"])

  if "enum" in schema:
    alts = [_value_to_regex(v) for v in schema["enum"]]
    return f'({"|".join(alts)})'

  if "allOf" in schema:
    merged = _merge_allof(schema["allOf"])
    # Carry over any top-level keys not in the sub-schemas.
    for k, v in schema.items():
      if k != "allOf" and k not in merged:
        merged[k] = v
    return _schema_to_regex(merged, indent, depth)

  if sub_schema := schema.get("anyOf") or schema.get("oneOf"):
    alts = [_schema_to_regex(s, indent, depth) for s in sub_schema]
    return f'({"|".join(alts)})'

  schema_type = schema.get("type")

  if isinstance(schema_type, list):
    alts = [
        _schema_to_regex({**schema, "type": t}, indent, depth)
        for t in schema_type
    ]
    return f'({"|".join(alts)})'

  if schema_type == "string":
    return _string_schema_to_regex(schema)

  if schema_type == "integer":
    return r"-?[0-9]+"

  if schema_type == "number":
    return r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?"

  if schema_type == "boolean":
    return r"(true|false)"

  if schema_type == "null":
    return "null"

  if schema_type == "object":
    return _object_schema_to_regex(schema, indent, depth)

  if schema_type == "array":
    return _array_schema_to_regex(schema, indent, depth)

  if "properties" in schema:
    return _object_schema_to_regex(schema, indent, depth)

  raise ValueError(f"Cannot convert schema to regex: {schema!r}")


def json_schema_to_regex(
    schema: JsonSchema,
    *,
    indent: int = 2,
) -> str:
  """Convert a JSON Schema dict into a regex for constrained decoding.

  Translates a JSON Schema into a regex pattern that matches valid JSON
  instances conforming to the schema. The resulting pattern can be compiled
  into a DFA via :func:`build_regex_constraint` for token-level guided
  generation.

  Example::

      schema = {
          "type": "object",
          "properties": {
              "label": {"type": "string", "maxLength": 20},
              "severity": {"enum": ["low", "medium", "high"]},
          },
          "required": ["label", "severity"],
      }
      pattern = json_schema_to_regex(schema)
      tables = build_regex_constraint(pattern, token_map, vocab_size, eos)

  Args:
      schema: A JSON Schema dict.  See :class:`JsonSchema` for the supported
        keyword subset.
      indent: Number of spaces per indentation level in the formatted JSON
        output.

  Returns:
      A regex pattern string.

  Raises:
      ValueError: If the schema contains constructs that cannot be
          expressed as a regex.
  """
  return _schema_to_regex(schema, indent, depth=0)


def build_regex_constraint(
    pattern: str,
    token_id_to_str: Mapping[int, str],
    vocab_size: int,
    eos_token_ids: Sequence[int],
) -> ConstraintTables:
  """Build a regex constraint for guided token-level decoding.

  Compiles *pattern* into a character-level DFA, then projects that DFA onto
  the token vocabulary to produce a transition table suitable for use inside
  ``jax.lax.while_loop``.

  Example usage::

      pattern = r'\\["(none|spam|gore)"(, "(none|spam|gore)")*\\]'
      tables = build_regex_constraint(
          pattern, token_id_to_str, vocab_size, eos_token_ids
      )

  Args:
      pattern: A regex pattern string (see module docstring for supported
        features).
      token_id_to_str: Mapping from every token id ``[0, vocab_size)`` to its
        decoded string representation.
      vocab_size: Size of the model's vocabulary.
      eos_token_ids: Token ids that signal end-of-sequence.

  Returns:
      A :class:`ConstraintTables` containing the compiled transition table
      and DFA metadata.

  Raises:
      ValueError: If *pattern* is malformed.
  """
  clean_pattern, min_tokens, max_tokens = _extract_and_strip_token_quantifiers(
      pattern
  )
  nfa_start, nfa_accept = _regex_to_nfa(clean_pattern)
  char_transitions, initial_state, accept_states, num_states = _nfa_to_dfa(
      nfa_start, nfa_accept
  )
  char_transitions, initial_state, accept_states, num_states = _minimize_dfa(
      char_transitions, initial_state, accept_states, num_states
  )
  token_transitions = _compile_token_transitions(
      char_transitions,
      num_states,
      accept_states,
      token_id_to_str,
      vocab_size,
      eos_token_ids,
  )
  tables = ConstraintTables(
      token_transitions=token_transitions,
      initial_state=initial_state,
      num_states=num_states,
      accept_states=accept_states,
      min_tokens=min_tokens,
      max_tokens=max_tokens,
  )
  return prepare_token_bound_masks(tables)


def constrained_logits(
    logits: jnp.ndarray,
    constraint_state: jnp.ndarray,
    token_transitions: jnp.ndarray,
    token_bounds_state: TokenBoundsLoopState | None = None,
) -> jnp.ndarray:
  """Mask logits to enforce a regex constraint.

  Sets logits for invalid next tokens to ``-inf`` so that they have zero
  probability after softmax.

  This function is pure JAX and safe for use inside
  ``jax.lax.while_loop``.

  Args:
      logits: Logit array of shape ``[B, 1, V]``.
      constraint_state: Current DFA state per batch element, shape ``[B]``.
      token_transitions: Token-level transition table of shape ``[S, V]`` where
        ``S`` is the number of DFA states and ``V`` is vocab size.
      token_bounds_state: Optional TokenBoundsLoopState carrying dynamic step
        counts, static masks, and bounds.

  Returns:
      Masked logits of the same shape as *logits*.
  """
  # allowed_next: [B, V] — look up the transition row for each batch item.
  allowed_next = token_transitions[constraint_state]  # [B, V]
  mask = allowed_next != INVALID_STATE  # [B, V]

  if token_bounds_state is not None:
    if token_bounds_state.min_tokens > 0:
      acc_mask_curr = token_bounds_state.accept_mask[constraint_state]
      min_forbid = (
          token_bounds_state.count[:, None] < token_bounds_state.min_tokens
      ) & acc_mask_curr
      mask = mask & (~min_forbid)

    if token_bounds_state.max_tokens is not None:
      force_mask_curr = token_bounds_state.force_accept_mask[constraint_state]
      max_force = (
          token_bounds_state.count[:, None] >= token_bounds_state.max_tokens
      )
      mask = jnp.where(max_force, mask & force_mask_curr, mask)

  return jnp.where(mask[:, None, :], logits, -jnp.inf)


def advance_state(
    constraint_state: jnp.ndarray,
    next_token: jnp.ndarray,
    token_transitions: jnp.ndarray,
) -> jnp.ndarray:
  """Advance the constraint DFA state after selecting a token.

  This function is pure JAX and safe for use inside
  ``jax.lax.while_loop``.

  Args:
      constraint_state: Current DFA state per batch element, shape ``[B]``.
      next_token: Selected token id per batch element, shape ``[B]``.
      token_transitions: Token-level transition table of shape ``[S, V]``.

  Returns:
      Updated constraint state of shape ``[B]``.
  """
  return token_transitions[constraint_state, next_token]
