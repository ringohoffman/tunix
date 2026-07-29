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
from collections.abc import Mapping, Sequence
import dataclasses
import json
import re
from typing import Any, Optional, TypedDict

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


@dataclasses.dataclass(frozen=True)
class ConstraintTables:
  """Pre-compiled constraint tables for regex-guided decoding.

  Attributes:
      token_transitions: Array of shape ``(num_states, vocab_size)`` with dtype
        ``int32``.  Entry ``[s, t]`` is the DFA state after processing token
        ``t`` from state ``s``, or ``INVALID_STATE`` if the token is forbidden.
      initial_state: The DFA start state.
      num_states: Total number of DFA states.
      accept_states: Set of accepting DFA state ids.
  """

  token_transitions: np.ndarray  # [num_states, vocab_size], int32
  initial_state: int
  num_states: int
  accept_states: frozenset[int]


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

  Args:
      terminal: Stop string (e.g. ``"</thought>"``, ``'"'``).
      min_chars: Minimum characters before ``terminal`` is allowed.
      max_chars: Maximum characters after which ``terminal`` is forced.

  Returns:
      Regex pattern string suitable for :func:`build_regex_constraint`.
  """
  t_escaped = "".join(
      f"\\{c}" if c in r"[]()*+?.\$^|{}" else c for c in terminal
  )
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
  minLength: int
  maxLength: int
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
  merged: dict[str, Any] = {}
  for s in schemas:
    for k, v in s.items():
      if k == "properties":
        merged.setdefault("properties", {}).update(v)
      elif k == "required":
        existing = merged.setdefault("required", [])
        existing.extend(r for r in v if r not in existing)
      else:
        merged[k] = v
  return merged  # type: ignore[return-value]


def _string_schema_to_regex(schema: JsonSchema, max_string_chars: int) -> str:
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

  min_len = schema.get("minLength", 0)
  max_len = schema.get("maxLength", max_string_chars)
  return f'"([^"]{{{min_len},{max_len}}})"'


def _object_schema_to_regex(
    schema: JsonSchema,
    indent: int,
    max_string_chars: int,
    max_array_items: int,
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
    val_pat = _schema_to_regex(
        prop_schema, indent, max_string_chars, max_array_items, depth + 1
    )
    field_lines.append(f'"{_regex_escape(name)}": {val_pat}')

  body = (",\n" + inner).join(field_lines)
  return "{\n" + inner + body + "\n" + outer + "}"


def _array_schema_to_regex(
    schema: JsonSchema,
    indent: int,
    max_string_chars: int,
    max_array_items: int,
    depth: int,
) -> str:
  """Regex for a ``{"type": "array"}`` schema (compact single-line format)."""
  # Tuple validation: prefixItems.
  if "prefixItems" in schema:
    item_pats = [
        _schema_to_regex(s, indent, max_string_chars, max_array_items, depth)
        for s in schema["prefixItems"]
    ]
    return r"\[" + ", ".join(item_pats) + r"\]"

  # List validation: items.
  items_schema = schema.get("items", {"type": "string"})
  min_items = schema.get("minItems", 0)
  max_items = schema.get("maxItems", max_array_items)

  item_pat = _schema_to_regex(
      items_schema, indent, max_string_chars, max_array_items, depth
  )

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


def _schema_to_regex(
    schema: JsonSchema,
    indent: int,
    max_string_chars: int,
    max_array_items: int,
    depth: int,
) -> str:
  """Recursive core: convert a JSON Schema dict to a regex pattern."""

  # --- const: single allowed value ---
  if "const" in schema:
    return _value_to_regex(schema["const"])

  # --- enum: alternation of allowed values ---
  if "enum" in schema:
    alts = [_value_to_regex(v) for v in schema["enum"]]
    return f'({"|".join(alts)})'

  # --- allOf: merge schemas, then recurse ---
  if "allOf" in schema:
    merged = _merge_allof(schema["allOf"])
    # Carry over any top-level keys not in the sub-schemas.
    for k, v in schema.items():
      if k != "allOf" and k not in merged:
        merged[k] = v  # type: ignore[literal-required]
    return _schema_to_regex(
        merged, indent, max_string_chars, max_array_items, depth
    )

  # --- anyOf / oneOf: alternation ---
  for keyword in ("anyOf", "oneOf"):
    if keyword in schema:
      alts = [
          _schema_to_regex(s, indent, max_string_chars, max_array_items, depth)
          for s in schema[keyword]
      ]
      return f'({"|".join(alts)})'

  # --- type dispatch ---
  schema_type = schema.get("type")

  # Multi-type: {"type": ["string", "null"]} → anyOf
  if isinstance(schema_type, list):
    alts = [
        _schema_to_regex(
            {**{k: v for k, v in schema.items() if k != "type"}, "type": t},
            indent,
            max_string_chars,
            max_array_items,
            depth,
        )
        for t in schema_type
    ]
    return f'({"|".join(alts)})'

  if schema_type == "string":
    return _string_schema_to_regex(schema, max_string_chars)

  if schema_type == "integer":
    return r"-?[0-9]+"

  if schema_type == "number":
    return r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?"

  if schema_type == "boolean":
    return r"(true|false)"

  if schema_type == "null":
    return "null"

  if schema_type == "object":
    return _object_schema_to_regex(
        schema, indent, max_string_chars, max_array_items, depth
    )

  if schema_type == "array":
    return _array_schema_to_regex(
        schema, indent, max_string_chars, max_array_items, depth
    )

  # No type but has properties → treat as object.
  if "properties" in schema:
    return _object_schema_to_regex(
        schema, indent, max_string_chars, max_array_items, depth
    )

  raise ValueError(f"Cannot convert schema to regex: {schema!r}")


def json_schema_to_regex(
    schema: JsonSchema,
    *,
    indent: int = 2,
    max_string_chars: int = 200,
    max_array_items: int = 20,
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
      max_string_chars: Default upper bound on string character length when the
        schema omits ``maxLength``.
      max_array_items: Default upper bound on array length when the schema omits
        ``maxItems``.

  Returns:
      A regex pattern string.

  Raises:
      ValueError: If the schema contains constructs that cannot be
          expressed as a regex.
  """
  return _schema_to_regex(
      schema, indent, max_string_chars, max_array_items, depth=0
  )


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
  nfa_start, nfa_accept = _regex_to_nfa(pattern)
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
  return ConstraintTables(
      token_transitions=token_transitions,
      initial_state=initial_state,
      num_states=num_states,
      accept_states=accept_states,
  )


def constrained_logits(
    logits: jnp.ndarray,
    constraint_state: jnp.ndarray,
    token_transitions: jnp.ndarray,
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

  Returns:
      Masked logits of the same shape as *logits*.
  """
  # allowed_next: [B, V] — look up the transition row for each batch item.
  allowed_next = token_transitions[constraint_state]  # [B, V]
  mask = allowed_next != INVALID_STATE  # [B, V]
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
