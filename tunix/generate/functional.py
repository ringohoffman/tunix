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

"""Functional stateless autoregressive generation for LLMs.

Pure JAX generation functions suitable for use inside ``jax.jit`` and
``jax.grad`` (via ``stop_gradient``).  No host-side I/O, no string
tokenization, no buffer donation — all of that belongs in the
convenience ``Sampler`` wrapper.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TypeAlias

import flax
from flax import nnx
import flax.struct
import jax
import jax.numpy as jnp
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import constrained
from tunix.generate import utils

LayerCache: TypeAlias = dict[str, jax.Array]
Cache: TypeAlias = dict[str, LayerCache]
StackedCache: TypeAlias = tuple[LayerCache | None, ...]
KVCache: TypeAlias = Cache | StackedCache | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrefixCache:
  """Cached KV state and metadata for a pre-computed token prefix."""

  # Cached key/value state populated for prefix tokens.
  cache: KVCache

  # Prefix token array of shape [P].
  prefix_tokens: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GenerateOutput:
  """Typed output from ``generate``."""

  tokens: jax.Array
  """Generated token IDs, shape ``[B, max_new_tokens]``."""

  logits: jax.Array | None = None
  """Per-step logits, shape ``[B, max_new_tokens, V]``, or ``None``."""

  logprobs: jax.Array | None = None
  """Per-step log-probabilities of sampled tokens, shape ``[B, max_new_tokens]``, or ``None``."""

  cache: KVCache = None
  """Final KV cache after generation.  Populated when ``return_cache=True``.

  Callers can explicitly manage this cache's lifecycle:
  - ``del output.cache``  — free HBM immediately
  - ``jax.device_put(output.cache, cpu)``  — offload to host RAM
  - Pass to downstream kernels that accept a warm cache
  """


@flax.struct.dataclass
class _DecodeState:
  """Carry state for the autoregressive decode while_loop."""

  step: jnp.int32
  token_buffer: jax.Array  # [B, total_len]
  positions: jax.Array  # [B, total_len]
  cache: KVCache
  done: jax.Array  # [B]
  key: jax.Array

  # Optional buffers (None when not requested)
  logits_buffer: jax.Array | None = None  # [B, total_len, V]
  logprobs_buffer: jax.Array | None = None  # [B, total_len]

  # Constraint state (None when unconstrained)
  constraint_state: jax.Array | None = None  # [B]
  unique_state: constrained.UniqueItemsLoopState | None = None
  token_bounds_state: constrained.TokenBoundsLoopState | None = None

  # Beam search state (None when not using beam search)
  beam_search_state: beam_search_lib._BeamSearchSamplingState | None = None


def _unpack_model_output(
    output: tuple[jax.Array, KVCache] | object,
) -> tuple[jax.Array, KVCache]:
  """Unpacks model output into ``(logits, cache)`` tuple."""
  if hasattr(output, "logits") and hasattr(output, "cache"):
    return output.logits, output.cache  # type: ignore[union-attr]
  if isinstance(output, (tuple, list)) and len(output) >= 2:
    return output[0], output[1]
  raise TypeError(
      f"Unsupported model output type: {type(output)}. Expected tuple (logits,"
      " cache) or object with .logits and .cache attributes."
  )


def _apply_constraints_and_forbidden_tokens(
    logits: jax.Array,
    forbidden_token_ids: jax.Array | None = None,
    constraint_state: jax.Array | None = None,
    constraint_transitions: jax.Array | None = None,
    constraint_active_tokens: jax.Array | None = None,
    unique_state: constrained.UniqueItemsLoopState | None = None,
    token_bounds_state: constrained.TokenBoundsLoopState | None = None,
    constraint_default_transitions: jax.Array | None = None,
) -> jax.Array:
  """Masks logits via forbidden-token and DFA constraints.

  Args:
    logits: Shape ``[B, 1, V]``.
    forbidden_token_ids: 1-D int32 array of forbidden vocab indices, or None.
    constraint_state: Per-batch DFA state ``[B]``, or None.
    constraint_transitions: DFA transition table ``[S, K]``, or None.
    constraint_active_tokens: Active token IDs ``[K]``, or None.
    unique_state: Unique-items loop state, or None.
    token_bounds_state: Token-bounds loop state, or None.
    constraint_default_transitions: Default successor per state ``[S]``, or
      None.

  Returns:
    Masked logits, same shape as input.
  """
  if logits.ndim == 2:
    logits = logits[:, None, :]

  if forbidden_token_ids is not None:
    logits = logits.at[:, :, forbidden_token_ids].set(-jnp.inf)

  if (
      constraint_transitions is not None
      and constraint_state is not None
      and constraint_active_tokens is not None
  ):
    if unique_state is not None:
      logits = constrained.constrained_logits_unique(
          logits,
          constraint_state,
          constraint_transitions,
          constraint_active_tokens,
          unique_state,
          token_bounds_state=token_bounds_state,
          default_transitions=constraint_default_transitions,
      )
    else:
      logits = constrained.constrained_logits(
          logits,
          constraint_state,
          constraint_transitions,
          constraint_active_tokens,
          token_bounds_state=token_bounds_state,
          default_transitions=constraint_default_transitions,
      )
  return logits


def _advance_constraint_state(
    next_token: jax.Array,
    constraint_state: jax.Array | None,
    constraint_transitions: jax.Array | None,
    constraint_active_tokens: jax.Array | None,
    unique_state: constrained.UniqueItemsLoopState | None,
    token_bounds_state: constrained.TokenBoundsLoopState | None,
    constraint_default_transitions: jax.Array | None = None,
) -> tuple[
    jax.Array | None,
    constrained.UniqueItemsLoopState | None,
    constrained.TokenBoundsLoopState | None,
]:
  """Advances DFA + unique + bounds state after selecting ``next_token``."""
  if (
      constraint_transitions is None
      or constraint_state is None
      or constraint_active_tokens is None
  ):
    return constraint_state, unique_state, token_bounds_state

  if token_bounds_state is not None:
    token_bounds_state = constrained.advance_token_bounds_state(
        token_bounds_state
    )

  if unique_state is not None:
    constraint_state, unique_state = constrained.advance_state_unique(
        constraint_state,
        next_token,
        constraint_transitions,
        constraint_active_tokens,
        unique_state,
        default_transitions=constraint_default_transitions,
    )
  else:
    constraint_state = constrained.advance_state(
        constraint_state,
        next_token,
        constraint_transitions,
        constraint_active_tokens,
        default_transitions=constraint_default_transitions,
    )
  return constraint_state, unique_state, token_bounds_state


def _sample_token(
    logits: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    key: jax.Array,
    step: int | jax.Array,
    return_logprobs: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Sample a single token from logits.

  Args:
    logits: Shape ``[B, 1, V]`` or ``[B, V]``.
    temperature: Sampling temperature.  0.0 → greedy.
    top_p: Nucleus sampling threshold (1.0 = no filtering).
    top_k: Top-k filtering (None = no filtering).
    key: PRNG key.
    step: Current decode step (folded into ``key``).
    return_logprobs: Whether to return log-probability of the sampled token.

  Returns:
    ``(next_token, logp)`` where ``next_token`` has shape ``[B]``
    and ``logp`` has shape ``[B]`` or is ``None``.
  """
  if logits.ndim == 2:
    logits = logits[:, None, :]
  else:
    logits = logits[:, -1:]

  from tunix.generate import sampler as sampler_lib

  if temperature == 0.0:
    return sampler_lib.sample_best(logits, return_logprobs=return_logprobs)

  step_key = jax.random.fold_in(key, step)
  return sampler_lib.sample_top_p(
      logits,
      step_key,
      temperature,
      top_p=top_p,
      top_k=top_k,
      return_logprobs=return_logprobs,
  )


def prefill_prefix(
    model: nnx.Module,
    prefix_ids: jax.Array | Sequence[int],
    cache_size: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> PrefixCache:
  """Prefill a constant prefix with batch size 1 and return a PrefixCache.

  Args:
    model: Transformer model NNX module.
    prefix_ids: 1-D array or sequence of prefix token IDs.
    cache_size: KV cache capacity.
    dtype: KV cache data type.

  Returns:
    A ``PrefixCache`` dataclass containing the prefilled KV cache, prefix
    length,
    and prefix token array.
  """
  prefix_arr = jnp.asarray(prefix_ids, dtype=jnp.int32)
  if prefix_arr.ndim == 1:
    prefix_arr = prefix_arr[None, :]
  pfx_batch, pfx_len = prefix_arr.shape
  if pfx_batch != 1:
    raise ValueError(f"prefix_ids must have batch size 1, got {pfx_batch}")

  cache = None
  if hasattr(model, "init_cache"):
    cache = model.init_cache(1, cache_size, dtype=dtype)

  positions = jnp.arange(pfx_len, dtype=jnp.int32)[None, :]
  prefix_mask = prefix_arr != 0

  if hasattr(model, "get_attention_mask"):
    attn_mask = model.get_attention_mask(prefix_arr, inputs_mask=prefix_mask)
    seq_len = attn_mask.shape[-1]
    padding = cache_size - seq_len
    attn_mask = jnp.pad(
        attn_mask,
        (*((0, 0) for _ in range(attn_mask.ndim - 1)), (0, padding)),
    )
  else:
    attn_mask = utils.make_causal_attn_mask(prefix_mask, cache_size)

  prefill_kwargs: dict[str, object] = {}
  try:
    if "decode_only_last_token" in inspect.signature(model.__call__).parameters:
      prefill_kwargs["decode_only_last_token"] = True
  except (ValueError, TypeError):
    pass

  _, cache = _unpack_model_output(
      model(
          prefix_arr,
          positions,
          cache,
          attn_mask,
          **prefill_kwargs,
      )
  )
  return PrefixCache(
      cache=cache,
      prefix_tokens=prefix_arr[0],
  )


def generate(
    model: nnx.Module,
    input_ids: jax.Array,
    *,
    max_new_tokens: int,
    pad_id: int,
    eos_id: int,
    cache_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    key: jax.Array | None = None,
    forbidden_token_ids: jax.Array | None = None,
    constraint_tables: constrained.ConstraintTables | None = None,
    beam_size: int | None = None,
    return_logits: bool = False,
    return_logprobs: bool = False,
    return_cache: bool = False,
    prefix_cache: PrefixCache | KVCache = None,
) -> GenerateOutput:
  """Autoregressive generation as a pure JAX function.

  Designed to be called inside ``jax.jit`` / ``jax.grad`` (via
  ``stop_gradient``).  Does NOT materialise to host or decode strings.

  Args:
    model: An NNX module implementing the standard transformer interface
      ``model(tokens, positions, cache, attention_mask) -> (logits, cache)``.
    input_ids: Left-padded prompt token IDs, shape ``[B, prompt_len]`` (or
      suffix-only ``[B, S]`` when ``prefix_cache`` is provided).
    max_new_tokens: Maximum number of new tokens to generate.
    pad_id: Padding token ID.
    eos_id: End-of-sequence token ID.
    cache_size: KV cache length (must be ``>= prompt_len + max_new_tokens``).
    temperature: Sampling temperature.  ``0.0`` → greedy.
    top_p: Nucleus sampling threshold (``1.0`` = disabled).
    top_k: Top-k filtering (``None`` = disabled).
    key: PRNG key for stochastic sampling.  Defaults to ``PRNGKey(0)``.
    forbidden_token_ids: 1-D int32 array of forbidden vocab IDs, or ``None``.
    constraint_tables: Pre-compiled DFA tables for grammar-constrained decoding,
      or ``None``.
    beam_size: Beam width for beam search.  ``None`` → sampling mode.
    return_logits: If ``True``, accumulate per-step logits.
    return_logprobs: If ``True``, accumulate per-step log-probabilities.
    return_cache: If ``True``, include the final KV cache in the output. Allows
      callers to explicitly manage cache lifecycle (keep, delete, or offload to
      host) rather than having it implicitly freed.
    prefix_cache: Optional ``PrefixCache`` containing precomputed KV cache for a
      shared prefix. When provided, prefill only executes for suffix tokens.

  Returns:
    A ``GenerateOutput`` containing generated tokens and optional
    logits / log-probabilities / KV cache.
  """
  batch_size = input_ids.shape[0]
  in_len = input_ids.shape[1]

  prefix_length = 0
  prefix_tokens_arr = None
  raw_cache = None
  if prefix_cache is not None:
    if isinstance(prefix_cache, PrefixCache):
      prefix_tokens_arr = prefix_cache.prefix_tokens
      prefix_length = int(prefix_tokens_arr.shape[-1])
      raw_cache = prefix_cache.cache
    elif isinstance(prefix_cache, dict) and "v" in prefix_cache:
      raw_cache = prefix_cache
      prefix_length = int(prefix_cache.get("end_index", [0])[0])
    else:
      raw_cache = prefix_cache

  if (
      prefix_length > 0
      and in_len <= prefix_length
      and prefix_tokens_arr is not None
  ):
    # Suffix-only tokens passed as input_ids: [B, S]
    suffix_len = in_len
    prompt_len = prefix_length + suffix_len
    total_len = prompt_len + max_new_tokens
    token_buffer = jnp.full((batch_size, total_len), pad_id, dtype=jnp.int32)
    token_buffer = token_buffer.at[:, :prefix_length].set(
        prefix_tokens_arr[None, :]
    )
    token_buffer = token_buffer.at[:, prefix_length:prompt_len].set(input_ids)
  else:
    # Full prompt passed as input_ids: [B, prompt_len] (or no prefix cache)
    prompt_len = in_len
    total_len = prompt_len + max_new_tokens
    token_buffer = jnp.full((batch_size, total_len), pad_id, dtype=jnp.int32)
    token_buffer = token_buffer.at[:, :prompt_len].set(input_ids)

  if max_new_tokens <= 0:
    return GenerateOutput(
        tokens=jnp.empty((batch_size, 0), dtype=input_ids.dtype),
    )

  if cache_size < total_len:
    raise ValueError(
        f"cache_size ({cache_size}) must be >= prompt_len + max_new_tokens"
        f" ({total_len})"
    )

  if key is None:
    key = jax.random.key(0)

  cache: KVCache = None
  if raw_cache is not None:

    def _broadcast_cache(leaf):
      if not isinstance(leaf, (jax.Array, jnp.ndarray)):
        return leaf
      if leaf.ndim >= 2 and leaf.shape[1] == 1 and leaf.shape[0] > 1:
        return jnp.broadcast_to(
            leaf, (leaf.shape[0], batch_size, *leaf.shape[2:])
        )
      elif leaf.shape[0] == 1 and batch_size > 1:
        return jnp.broadcast_to(leaf, (batch_size, *leaf.shape[1:]))
      return leaf

    cache = jax.tree.map(_broadcast_cache, raw_cache)
  elif hasattr(model, "init_cache"):
    model_dtype = (
        model.config.dtype
        if hasattr(model, "config") and hasattr(model.config, "dtype")
        else jnp.bfloat16
    )
    cache = model.init_cache(
        batch_size,
        cache_size,
        dtype=model_dtype,
    )

  constraint_state: jax.Array | None = None
  constraint_active_tokens: jax.Array | None = None
  constraint_transitions: jax.Array | None = None
  constraint_default_transitions: jax.Array | None = None
  unique_state: constrained.UniqueItemsLoopState | None = None
  token_bounds_state: constrained.TokenBoundsLoopState | None = None
  if constraint_tables is not None:
    constraint_state = jnp.full(
        (batch_size,), constraint_tables.initial_state, dtype=jnp.int32
    )
    constraint_active_tokens = jnp.array(
        constraint_tables.active_tokens, dtype=jnp.int32
    )
    constraint_transitions = jnp.array(
        constraint_tables.token_transitions, dtype=jnp.int32
    )
    if constraint_tables.default_transitions is not None:
      constraint_default_transitions = jnp.array(
          constraint_tables.default_transitions, dtype=jnp.int32
      )
    token_bounds_state = constrained.init_token_bounds_loop_state(
        constraint_tables, batch_size
    )
    if constraint_tables.unique_items is not None:
      unique_state = constrained.init_unique_items_loop_state(
          constraint_tables.unique_items, batch_size
      )

  positions = utils.build_positions_from_mask(token_buffer != pad_id)

  prefill_kwargs: dict[str, object] = {}
  try:
    if "decode_only_last_token" in inspect.signature(model.__call__).parameters:
      prefill_kwargs["decode_only_last_token"] = True
  except (ValueError, TypeError):
    pass

  if prefix_length > 0:
    suffix_len = prompt_len - prefix_length
    suffix_tokens = token_buffer[:, prefix_length:prompt_len]
    suffix_positions = positions[:, prefix_length:prompt_len]
    suffix_mask = suffix_tokens != pad_id

    attention_mask = jnp.zeros(
        (batch_size, suffix_len, cache_size), dtype=jnp.bool_
    )
    attention_mask = attention_mask.at[:, :, :prefix_length].set(True)
    causal_suffix = jnp.tril(
        jnp.ones((suffix_len, suffix_len), dtype=jnp.bool_)
    )[None, ...]
    attention_mask = attention_mask.at[:, :, prefix_length:prompt_len].set(
        causal_suffix & suffix_mask[:, None, :]
    )

    out = model(
        suffix_tokens,
        suffix_positions,
        cache,
        attention_mask,
        **prefill_kwargs,
    )
    logits, cache = _unpack_model_output(out)
  else:
    input_mask = input_ids != pad_id
    if hasattr(model, "get_attention_mask"):
      attention_mask = model.get_attention_mask(
          input_ids, inputs_mask=input_mask
      )
      seq_len = attention_mask.shape[-1]
      padding = cache_size - seq_len
      attention_mask = jnp.pad(
          attention_mask,
          (*((0, 0) for _ in range(attention_mask.ndim - 1)), (0, padding)),
      )
    else:
      attention_mask = utils.make_causal_attn_mask(input_mask, cache_size)

    out = model(
        input_ids,
        positions[:, :prompt_len],
        cache,
        attention_mask,
        **prefill_kwargs,
    )
    logits, cache = _unpack_model_output(out)

  # Initialise logits/logprobs buffers now that we know the vocab size.
  actual_vocab_size = logits.shape[-1]
  logits_buffer: jax.Array | None = None
  logprobs_buffer: jax.Array | None = None
  if return_logits:
    logits_buffer = jnp.zeros(
        (batch_size, total_len, actual_vocab_size),
        dtype=jnp.float32,
    )
    # Store prefill logits (last position only, matching Sampler behaviour).
    logits_buffer = logits_buffer.at[:, prompt_len - 1, :].set(
        logits[:, -1].astype(jnp.float32)
    )

  if return_logprobs:
    logprobs_buffer = jnp.zeros(
        (batch_size, total_len),
        dtype=jnp.float32,
    )

  # beam search init (expand batch after prefill)
  beam_search_state: beam_search_lib._BeamSearchSamplingState | None = None
  if beam_size is not None and beam_size > 1:
    beam_search_state, updated = beam_search_lib.init_batched_beam_state(
        logits=logits,
        input_token_buffer=token_buffer,
        initial_cache=cache,
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        positions=positions,
        logits_buffer=logits_buffer,
        beam_size=beam_size,
    )
    logits = updated["logits"]
    cache = updated["cache"]
    token_buffer = updated["token_buffer"]
    positions = updated["positions"]
    logits_buffer = updated["logits_buffer"]
    done_expanded = updated["done"]
    batch_size = batch_size * beam_size  # effective batch is now B*K
    key_expanded = key  # same key, fold_in handles per-step randomness
  else:
    done_expanded = jnp.zeros((batch_size,), dtype=jnp.bool_)
    key_expanded = key

  last_logits = logits[:, -1:]
  last_logits = _apply_constraints_and_forbidden_tokens(
      last_logits,
      forbidden_token_ids=forbidden_token_ids,
      constraint_state=constraint_state,
      constraint_transitions=constraint_transitions,
      constraint_active_tokens=constraint_active_tokens,
      unique_state=unique_state,
      token_bounds_state=token_bounds_state,
      constraint_default_transitions=constraint_default_transitions,
  )

  if beam_size is not None and beam_size > 1:
    # Beam search handles its own token selection
    beam_search_state, bs_updated = beam_search_lib.beam_search_step(
        logits=last_logits,
        done=done_expanded,
        token_buffer=token_buffer,
        cache=cache,
        logits_buffer=logits_buffer,
        state=beam_search_state,
        pad_token_id=pad_id,
        decoding_step=prompt_len - 1,
        logprobs_buffer=logprobs_buffer,
    )
    cache = bs_updated["cache"]
    token_buffer = bs_updated["token_buffer"]
    done_expanded = bs_updated["done"]
    logits_buffer = bs_updated["logits_buffer"]
    logprobs_buffer = bs_updated.get("logprobs_buffer", logprobs_buffer)
    first_token = token_buffer[:, prompt_len]
  else:
    first_token, logp = _sample_token(
        last_logits,
        temperature,
        top_p,
        top_k,
        key_expanded,
        step=0,
        return_logprobs=return_logprobs,
    )
    token_buffer = token_buffer.at[:, prompt_len].set(first_token)
    if logprobs_buffer is not None and logp is not None:
      logprobs_buffer = logprobs_buffer.at[:, prompt_len].set(logp)

  # Store logits for the first generated position.
  if logits_buffer is not None:
    logits_buffer = logits_buffer.at[:, prompt_len, :].set(
        jnp.squeeze(last_logits, 1).astype(jnp.float32)
    )

  constraint_state, unique_state, token_bounds_state = (
      _advance_constraint_state(
          first_token,
          constraint_state=constraint_state,
          constraint_transitions=constraint_transitions,
          constraint_active_tokens=constraint_active_tokens,
          unique_state=unique_state,
          token_bounds_state=token_bounds_state,
          constraint_default_transitions=constraint_default_transitions,
      )
  )

  last_prompt_pos = jnp.max(positions[:, :prompt_len], axis=-1)
  first_decode_pos = last_prompt_pos + 1
  positions = positions.at[:, prompt_len].set(first_decode_pos)

  done = done_expanded | (first_token == eos_id)

  init_state = _DecodeState(
      step=jnp.int32(1),
      token_buffer=token_buffer,
      positions=positions,
      cache=cache,
      done=done,
      key=key_expanded,
      logits_buffer=logits_buffer,
      logprobs_buffer=logprobs_buffer,
      constraint_state=constraint_state,
      unique_state=unique_state,
      token_bounds_state=token_bounds_state,
      beam_search_state=beam_search_state,
  )

  def cond_fn(carry: tuple[nnx.Module, _DecodeState]) -> jax.Array:
    _, state = carry
    return (state.step < max_new_tokens) & jnp.any(~state.done)

  def body_fn(
      carry: tuple[nnx.Module, _DecodeState],
  ) -> tuple[nnx.Module, _DecodeState]:
    loop_model, state = carry
    current_idx = prompt_len + state.step - 1

    last_token = state.token_buffer[:, current_idx][:, None]
    step_position = state.positions[:, current_idx][:, None]

    attn_mask = utils.compute_attention_masks(
        current_idx, cache_size, state.token_buffer == pad_id
    )

    loop_out = loop_model(
        last_token,
        step_position,
        state.cache,
        attn_mask,
    )
    loop_logits, new_cache = _unpack_model_output(loop_out)

    loop_logits = _apply_constraints_and_forbidden_tokens(
        loop_logits,
        forbidden_token_ids=forbidden_token_ids,
        constraint_state=state.constraint_state,
        constraint_transitions=constraint_transitions,
        constraint_active_tokens=constraint_active_tokens,
        unique_state=state.unique_state,
        token_bounds_state=state.token_bounds_state,
        constraint_default_transitions=constraint_default_transitions,
    )

    # Store logits before beam search potentially reshuffles buffers.
    new_logits_buffer = state.logits_buffer
    if new_logits_buffer is not None:
      write_logits_idx = prompt_len + state.step
      new_logits_buffer = new_logits_buffer.at[:, write_logits_idx, :].set(
          jnp.squeeze(loop_logits, 1).astype(jnp.float32)
      )

    new_logprobs_buffer = state.logprobs_buffer
    new_beam_search_state = state.beam_search_state
    new_token_buffer = state.token_buffer
    new_done = state.done

    if beam_size is not None and beam_size > 1:
      new_beam_search_state, bs_upd = beam_search_lib.beam_search_step(
          logits=loop_logits,
          done=state.done,
          token_buffer=state.token_buffer,
          cache=new_cache,
          logits_buffer=new_logits_buffer,
          state=state.beam_search_state,
          pad_token_id=pad_id,
          decoding_step=current_idx,
          logprobs_buffer=new_logprobs_buffer,
      )
      new_cache = bs_upd["cache"]
      new_token_buffer = bs_upd["token_buffer"]
      new_done = bs_upd["done"]
      new_logits_buffer = bs_upd["logits_buffer"]
      new_logprobs_buffer = bs_upd.get("logprobs_buffer", new_logprobs_buffer)
      next_token = new_token_buffer[:, prompt_len + state.step]
    else:
      next_token, logp = _sample_token(
          loop_logits,
          temperature,
          top_p,
          top_k,
          state.key,
          step=state.step,
          return_logprobs=return_logprobs,
      )
      next_token_masked = jnp.where(state.done, pad_id, next_token)
      write_idx = prompt_len + state.step
      new_token_buffer = state.token_buffer.at[:, write_idx].set(
          next_token_masked
      )
      if new_logprobs_buffer is not None and logp is not None:
        new_logprobs_buffer = new_logprobs_buffer.at[:, write_idx].set(logp)

    new_constraint_state, new_unique_state, new_token_bounds_state = (
        _advance_constraint_state(
            next_token,
            constraint_state=state.constraint_state,
            constraint_transitions=constraint_transitions,
            constraint_active_tokens=constraint_active_tokens,
            unique_state=state.unique_state,
            token_bounds_state=state.token_bounds_state,
            constraint_default_transitions=constraint_default_transitions,
        )
    )

    next_position = step_position[:, 0] + 1
    write_pos_idx = prompt_len + state.step
    new_positions = state.positions.at[:, write_pos_idx].set(next_position)
    new_done = new_done | (next_token == eos_id)

    return loop_model, _DecodeState(
        step=state.step + 1,
        token_buffer=new_token_buffer,
        positions=new_positions,
        cache=new_cache,
        done=new_done,
        key=state.key,
        logits_buffer=new_logits_buffer,
        logprobs_buffer=new_logprobs_buffer,
        constraint_state=new_constraint_state,
        unique_state=new_unique_state,
        token_bounds_state=new_token_bounds_state,
        beam_search_state=new_beam_search_state,
    )

  _, final_state = nnx.while_loop(cond_fn, body_fn, (model, init_state))

  out_token_buffer = final_state.token_buffer
  out_logits_buffer = final_state.logits_buffer
  out_logprobs_buffer = final_state.logprobs_buffer

  # finalize beam search: select best beam per batch element
  if beam_size is not None and beam_size > 1:
    bs_final = beam_search_lib.finalize_beam_search_state(
        final_state.beam_search_state,
        out_token_buffer,
        out_logits_buffer,
        out_logprobs_buffer,
    )
    out_token_buffer = bs_final["token_buffer"]
    out_logits_buffer = bs_final["logits_buffer"]
    out_logprobs_buffer = bs_final["logprobs_buffer"]

  gen_tokens = out_token_buffer[:, prompt_len : prompt_len + max_new_tokens]

  gen_logits: jax.Array | None = None
  if out_logits_buffer is not None:
    gen_logits = out_logits_buffer[:, prompt_len : prompt_len + max_new_tokens]

  gen_logprobs: jax.Array | None = None
  if out_logprobs_buffer is not None:
    gen_logprobs = out_logprobs_buffer[
        :, prompt_len : prompt_len + max_new_tokens
    ]

  return GenerateOutput(
      tokens=gen_tokens,
      logits=gen_logits,
      logprobs=gen_logprobs,
      cache=final_state.cache if return_cache else None,
  )
