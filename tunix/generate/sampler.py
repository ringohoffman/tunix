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

"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import inspect
import time
from typing import Any, Literal, TypeGuard, TypedDict, overload
import warnings

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import flax.typing
import jax
import jax.numpy as jnp
import jax.typing
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import constrained
from tunix.generate import utils
import tunix.generate.beam_search as beam_search_lib
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.processors import image_processor as image_processor_lib
from tunix.utils import sharding_utils

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


def is_variable(x: Any) -> TypeGuard[nnx.Variable[jax.Array]]:
  """TypeGuard for nnx.Variable leaves in jax.tree flattening."""
  return isinstance(x, nnx.Variable)


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, LayerCache]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # Fixed-size buffer for accumulating the output logprobs.
  logprobs_buffer: jnp.ndarray | None  # [B, L]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: tuple[int, ...] | None

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Sampling parameters.
  # For top_p, it contains "top_p" and "top_k".
  # For beam search, it contains "beam_size"
  sampling_parameters: utils.SamplingParameters = flax.struct.field(
      pytree_node=False
  )

  # Only present when sampling_mode is "beam_search".
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None

  # Constraint DFA state per batch element, shape [B], int32.
  # None when no constraint is active.
  constraint_state: jnp.ndarray | None = None

  # Token-level constraint transition table, shape [S, V], int32.
  # Constant across decode steps; carried in state so it is visible
  # inside the jax.lax.while_loop.
  constraint_transitions: jnp.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
  # Upcast to float32 for numerical stability of softmax and subsequent cumsum.
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  # top_k=0 or None both mean "no top-k filtering" — use full vocabulary.
  _no_topk = top_k is None or top_k <= 0
  # Skip softmax and sorting if top_p is 1.0 and top_k is full vocab.
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not return_logprobs:
      return next_token, None
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    return next_token, logp_sampled

  k = next_token_logits.shape[-1] if _no_topk else top_k
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)

  probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > top_p
  logits_sorted = jnp.where(mask, -jnp.inf, logits_sorted)

  next_token_idx = jax.random.categorical(key, logits=logits_sorted)
  next_token = jnp.take_along_axis(indices, next_token_idx[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)

  if return_logprobs:
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  else:
    logp_sampled = None

  return next_token, logp_sampled


def sample_best(
    logits, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


@nnx.jit(
    static_argnames=(
        'n_layers',
        'cache_size',
        'batch_size',
        'num_kv_heads',
        'head_dim',
        'dtype',
        'batch_sharding',
        'data_sharding',
    )
)
def _init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
    batch_sharding: jax.sharding.NamedSharding,
    data_sharding: jax.sharding.NamedSharding,
) -> Cache:
  """Create KV cache for the transformer.

  Args:
    n_layers: The number of attention layers.
    cache_size: The size of the cache.
    batch_size: The batch size.
    num_kv_heads: The number of KV attention heads.
    head_dim: The dimension of the KV attention head.
    dtype: The data type of the cache.
    batch_sharding: NamedSharding for 1D batch-sliced arrays of shape [B].
    data_sharding: NamedSharding for 4D KV cache tensors of shape [B, S, H, D].

  Returns:
    The KV cache for one attention block.
  """

  shape = (batch_size, cache_size, num_kv_heads, head_dim)
  return {
      f'layer_{i}': {
          'k': jax.device_put(jnp.zeros(shape, dtype=dtype), data_sharding),
          'v': jax.device_put(jnp.zeros(shape, dtype=dtype), data_sharding),
          'end_index': jax.device_put(
              jnp.zeros((batch_size,), dtype=jnp.int32), batch_sharding
          ),
      }
      for i in range(n_layers)
  }


class Sampler(base_sampler.BaseSampler):
  """Sampler for transformer model.

  Args:
    transformer: An instance of the transformer model.
    tokenizer: A tokenizer for the given model.
    cache_config: Configuration for the KV cache.
    image_processor: Optional image processor for vision-language models.
    eos_tokens: End-of-sequence token IDs. Defaults to tokenizer's eos_id.
  """

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
      image_processor: image_processor_lib.ImageProcessor | None = None,
      eos_tokens: Sequence[int] | None = None,
  ) -> None:
    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self.image_processor = image_processor
    self.eos_tokens = jnp.array(
        eos_tokens if eos_tokens is not None else [self.tokenizer.eos_id()]
    )

    mesh = jax.sharding.get_mesh()
    if mesh.empty:
      mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty:
      shd_config = getattr(
          getattr(transformer, 'config', None), 'shd_config', None
      )
      if shd_config is not None and hasattr(shd_config, 'input_pspec'):
        # Preferred: explicit input sharding from model config.
        input_pspec = shd_config.input_pspec
      elif shd_config is not None and hasattr(shd_config, 'act_btd'):
        # Legacy fallback: infer batch axis from activation spec.
        batch_axis = shd_config.act_btd[0]
        input_pspec = jax.sharding.PartitionSpec(batch_axis, None)
      else:
        batch_axis = 'fsdp' if 'fsdp' in mesh.shape else None
        input_pspec = jax.sharding.PartitionSpec(batch_axis, None)
      self.data_sharding = jax.sharding.NamedSharding(mesh, input_pspec)
    else:
      active_mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('dev',))
      self.data_sharding = jax.sharding.NamedSharding(
          active_mesh, jax.sharding.PartitionSpec()
      )
    batch_axis = (
        self.data_sharding.spec[0] if len(self.data_sharding.spec) > 0 else None
    )
    self.batch_sharding = jax.sharding.NamedSharding(
        self.data_sharding.mesh,
        jax.sharding.PartitionSpec(batch_axis),
    )
    self.logits_sharding = jax.sharding.NamedSharding(
        self.data_sharding.mesh,
        jax.sharding.PartitionSpec(batch_axis, None, None),
    )

    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)
    self._transformer_state: nnx.State[
        flax.typing.PathParts, nnx.Variable[jax.Array]
    ] = nnx.variables(transformer)
    self._flattened_transformer_state: list[nnx.Variable[jax.Array]] = (
        jax.tree.leaves(self._transformer_state, is_leaf=is_variable)
    )
    # We separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time.
    #
    # We donate the sampling_state (argnum 1) containing the KV cache arrays.
    # JAX arrays are immutable, so updating the cache at each decoding step
    # would normally force JAX to allocate a new memory buffer and copy old
    # contents. Since the KV cache memory footprint scales with batch size and
    # prompt+decoding length (reaching gigabytes), this continuous reallocation
    # and copying triggers massive memory overhead and OOMs. Donating the input
    # state allows the XLA compiler to reuse the memory buffer in-place,
    # completely avoiding allocation/copy overhead.
    self._compiled_decode_fn = jax.jit(self._decode_fn, donate_argnums=(1,))
    self._compiled_prefill_fn = jax.jit(
        self._prefill_fn,
        donate_argnums=(1,),
        static_argnames=('echo',),
    )
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

  def compile_constraint(
      self,
      pattern: str | None = None,
      schema: dict | None = None,
      **kwargs,
  ) -> constrained.ConstraintTables:
    """Compiles a regex pattern or JSON Schema into token-level constraint tables.

    Args:
      pattern: Regex pattern string.
      schema: JSON Schema dictionary. Mutually exclusive with pattern.
      **kwargs: Keyword arguments passed to json_schema_to_regex if schema is
        provided.

    Returns:
      ConstraintTables containing token transition tables.
    """
    if pattern is None and schema is None:
      raise ValueError('Either pattern or schema must be provided.')
    if pattern is not None and schema is not None:
      raise ValueError('Only one of pattern or schema should be provided.')
    if schema is not None:
      pattern = constrained.json_schema_to_regex(schema, **kwargs)

    eos_ids = (
        self.eos_tokens.tolist()
        if hasattr(self.eos_tokens, 'tolist')
        else list(self.eos_tokens)
    )
    return constrained.build_regex_constraint(
        pattern=pattern,
        token_id_to_str=self.tokenizer.token_id_to_str,
        vocab_size=self.tokenizer.vocab_size,
        eos_token_ids=eos_ids,
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, statelib.State]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:

    def get_all_param_types(tree):
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=is_variable,
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):

        def equivalent_sharding(x, y):
          # Lift the condition on memory_kind due to offloading.
          # Besides it seems jax.jit might change some shardings of the params
          # to equivalent representation so here we check if the specs are
          # equivalent instead of checking the identity.
          if isinstance(
              x.sharding, jax.sharding.SingleDeviceSharding
          ) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
            return x.sharding.device_set == y.sharding.device_set
          if not (
              isinstance(x.sharding, jax.sharding.NamedSharding)
              and isinstance(y.sharding, jax.sharding.NamedSharding)
          ):
            return False
          if x.sharding.mesh != y.sharding.mesh:
            return False
          mesh = x.sharding.mesh
          diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
          for spec in diff_spec:
            if spec and mesh.shape[spec] != 1:
              return False
          return True

        return (
            jnp.shape(x) == jnp.shape(y)
            and x.dtype == y.dtype
            and equivalent_sharding(x, y)
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`: '
            f'{param_types}'
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=is_variable,
    )

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: tuple[int, ...] | None,
      temperature: float,
      top_p: float | None,
      top_k: int | None,
      seed: jax.Array,
      beam_size: int | None,
      include_logprobs: bool = False,
      constraint_tables: constrained.ConstraintTables | None = None,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size, num_input_tokens, *_ = all_input_ids.shape

    if seed is None:
      seed = jax.random.key(0)
    elif not hasattr(seed, 'dtype'):
      seed = jax.random.key(seed)

    token_buffer = jnp.full(
        (batch_size, total_sampling_steps),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    token_buffer = jax.device_put(token_buffer, self.data_sharding)

    positions = jax.device_put(
        utils.build_positions_from_mask(
            token_buffer != self.tokenizer.pad_id()
        ),
        self.data_sharding,
    )
    done = jax.device_put(
        jnp.zeros((batch_size,), dtype=jnp.bool_), self.batch_sharding
    )

    if hasattr(self.transformer, 'init_cache'):
      cache = self.transformer.init_cache(
          batch_size, self.cache_config.cache_size, self.dtype
      )
    else:
      warnings.warn(
          'Using deprecated _init_cache in Tunix sampler. Models are now'
          ' required to have their own init_cache attribute.',
          DeprecationWarning,
      )
      cache = _init_cache(
          n_layers=self.cache_config.num_layers,
          cache_size=self.cache_config.cache_size,
          batch_size=batch_size,
          num_kv_heads=self.cache_config.num_kv_heads,
          head_dim=self.cache_config.head_dim,
          dtype=self.dtype,
          batch_sharding=self.batch_sharding,
          data_sharding=self.data_sharding,
      )

    logits_buffer = (
        jax.device_put(
            jnp.zeros(
                (batch_size, total_sampling_steps, self.transformer.num_embed),
                dtype=jnp.float32,
            ),
            self.logits_sharding,
        )
        if include_logits
        else None
    )

    logprobs_buffer = (
        jax.device_put(
            jnp.zeros((batch_size, total_sampling_steps), dtype=jnp.float32),
            self.data_sharding,
        )
        if include_logprobs
        else None
    )

    sampling_mode, sampling_parameters = utils.resolve_sampling_config(
        beam_size=beam_size, top_p=top_p, top_k=top_k
    )

    constraint_state = None
    constraint_transitions = None
    if constraint_tables is not None:
      constraint_state = jnp.full(
          (batch_size,), constraint_tables.initial_state, dtype=jnp.int32
      )
      constraint_transitions = jnp.array(
          constraint_tables.token_transitions, dtype=jnp.int32
      )

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=int(num_input_tokens),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        seed=seed,
        sampling_mode=sampling_mode,
        beam_search_sampling_state=None,
        constraint_state=constraint_state,
        constraint_transitions=constraint_transitions,
    )

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _sample(
      self,
      logits: jax.Array,
      cache: dict[str, LayerCache],
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""

    logits = logits[:, -1][:, None, :]  # B, 1, V
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.logits_buffer
    logprobs_buffer = sampler_state.logprobs_buffer
    beam_search_state = sampler_state.beam_search_sampling_state
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.constraint_transitions is not None:
      logits = constrained.constrained_logits(
          logits,
          sampler_state.constraint_state,
          sampler_state.constraint_transitions,
      )

    if sampler_state.sampling_mode == 'beam_search':
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,
          pad_token_id=self.tokenizer.pad_id(),
          decoding_step=decoding_step,
          logprobs_buffer=logprobs_buffer,
      )
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      logits_buffer = updated_args['logits_buffer']
      logprobs_buffer = updated_args['logprobs_buffer']
    else:
      if sampler_state.sampling_mode == 'greedy':
        next_token_candidate, logp = sample_best(
            logits, return_logprobs=(logprobs_buffer is not None)
        )
      elif sampler_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampler_state.seed, decoding_step)
        next_token_candidate, logp = sample_top_p(
            logits,
            key,
            sampler_state.temperature,
            sampler_state.sampling_parameters['top_p'],
            sampler_state.sampling_parameters['top_k'],
            return_logprobs=(logprobs_buffer is not None),
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampler_state.sampling_mode
        )
      token_buffer = token_buffer.at[:, decoding_step + 1].set(
          next_token_candidate
      )
      if logprobs_buffer is not None:
        logprobs_buffer = logprobs_buffer.at[:, decoding_step + 1].set(logp)

    constraint_state = sampler_state.constraint_state
    if sampler_state.constraint_transitions is not None:
      constraint_state = constrained.advance_state(
          sampler_state.constraint_state,
          next_token_candidate,
          sampler_state.constraint_transitions,
      )

    done = done | jnp.isin(token_buffer[:, decoding_step + 1], self.eos_tokens)
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_state,
        constraint_state=constraint_state,
        constraint_transitions=sampler_state.constraint_transitions,
    )

  def _prefill_fn(
      self,
      params: statelib.State,
      sampler_state: _SamplingState,
      images: jnp.ndarray | None = None,
      echo: bool = True,
  ) -> _SamplingState:
    """Performs prefill."""
    batch_size = sampler_state.token_buffer.shape[0]

    tokens = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )
    step_positions = jax.lax.dynamic_slice(
        sampler_state.positions,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    input_mask = tokens != self.tokenizer.pad_id()

    if hasattr(self.transformer, 'get_attention_mask'):
      attention_mask = self.transformer.get_attention_mask(
          tokens, inputs_mask=input_mask
      )
      seq_len = attention_mask.shape[-1]
      padding = self.cache_config.cache_size - seq_len
      attention_mask = jnp.pad(
          attention_mask,
          (*((0, 0) for _ in range(attention_mask.ndim - 1)), (0, padding)),
      )
    else:
      attention_mask = utils.make_causal_attn_mask(
          input_mask, self.cache_config.cache_size
      )

    # Merge once at the JIT boundary, outside any traced control flow.
    transformer = nnx.merge(self._transformer_graphdef, params)
    kwargs = {} if images is None else {'images': images}
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
        **kwargs,
    )
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.positions
    beam_search_sampling_state = None
    if sampler_state.logits_buffer is not None:
      start_idx = (
          sampler_state.num_input_tokens if decode_only_last_token else 1
      )
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, start_idx, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.sampling_mode == 'beam_search':
      # init beam state in prefill instead of init as one minor optimization
      # to avoid running unnecessary prefill for
      # duplicated input prompt per beam.
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.positions,
          logits_buffer=sampler_state.logits_buffer,
          beam_size=int(sampler_state.sampling_parameters['beam_size']),
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args['logits']
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      positions = updated_args['positions']
      logits_buffer = updated_args['logits_buffer']

    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=sampler_state.logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_sampling_state,
        constraint_state=sampler_state.constraint_state,
        constraint_transitions=sampler_state.constraint_transitions,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        sampler_state=updated_sampling_state,
    )
    return updated_sampler_state

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted).

    Uses nnx.while_loop instead of jax.lax.while_loop so that NNX graph
    operations (merge/split on the transformer module) happen at the
    correct trace level.  This makes the function compatible with
    jax.jit(...).lower() for programmatic HLO inspection.
    """
    # Merge once at the JIT boundary, outside the while_loop.
    transformer = nnx.merge(self._transformer_graphdef, params)

    def sample_with_transformer(
        carry: tuple[nnx.Module, _SamplingState],
    ) -> tuple[nnx.Module, _SamplingState]:
      transformer, sampler_state = carry
      return transformer, self._sample_step(transformer, sampler_state)

    def cond_fn(
        carry: tuple[nnx.Module, _SamplingState],
    ) -> jax.Array:
      _, sampler_state = carry
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    _, sampling_state = nnx.while_loop(
        cond_fn, sample_with_transformer, (transformer, sampling_state)
    )
    return sampling_state

  def _sample_step(
      self,
      transformer: nnx.Module,
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Performs a single sampling step.

    Args:
      transformer: An already-merged NNX module. Must NOT be re-merged from
        (graphdef, state) inside this function, because it is called inside an
        nnx.while_loop body where NNX handles the graph state serialization
        automatically.
      sampler_state: The current sampling state.

    Returns:
      Updated sampling state after one decode step.
    """
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = sampler_state.decoding_step

    last_token = sampler_state.token_buffer[:, decoding_step]
    last_token = last_token.reshape((batch_size, 1))
    step_positions = jnp.expand_dims(
        sampler_state.positions[:, decoding_step], -1
    )

    input_mask = sampler_state.token_buffer == self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )

    logits, cache = transformer(
        last_token,
        positions=step_positions,
        cache=sampler_state.cache,
        attention_mask=attention_mask,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        sampler_state=sampler_state,
    )

    if updated_sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = updated_sampler_state.logits_buffer.at[
          :, decoding_step + 1
      ].set(next_logits)
    else:
      logits_buffer = None

    updated_sampler_state = dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )
    return updated_sampler_state

  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      forbidden_tokens: Iterable[int] | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: int | None = None,
      pad_output: bool = False,
      images: (
          str
          | np.ndarray
          | list[str | np.ndarray | list[str | np.ndarray] | None]
          | jnp.ndarray
          | None
      ) = None,
      constraint_tables: constrained.ConstraintTables | None = None,
      constraint_pattern: str | None = None,
      constraint_schema: dict | None = None,
  ) -> base_sampler.SamplerOutput:
    """Samples a completion of the input string.

    If top_p is provided, the sampling mode will be top_p.
    If beam_size is provided, the sampling mode will be beam_search.
    If None of them are provided, the sampling mode will be greedy.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      max_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: Optional Iterable of token IDs that are disallowed.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      beam_size: beam size for beam search.
      seed: random seed for sampling.
      pad_output: whether to pad the output to maximum length. If this set as
        True, the output len will be max_generation_steps if echo is False,
        otherwise it will be max_generation_steps + max_prompt_length. The
        padding now only supports right padding. Can modify to support left
        padding if needed.
      images: input images to process. Can be a string/array, list of
        strings/arrays, or list of list of strings/arrays depending on whether
        there is one, multiple, or varying number of images per batch.
      constraint_tables: Pre-compiled ConstraintTables for guided decoding.
      constraint_pattern: Regex pattern string to constrain output.
      constraint_schema: JSON Schema dictionary to constrain output.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )
    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None
    tokens = [self.tokenize(x) for x in input_strings]

    processed_images = images
    if images is not None and self.image_processor is not None:
      processed_images = self.image_processor(images)
      processed_images = jnp.array(processed_images)

    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    all_input_ids = np.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])

    if constraint_tables is None and (
        constraint_pattern is not None or constraint_schema is not None
    ):
      constraint_tables = self.compile_constraint(
          pattern=constraint_pattern, schema=constraint_schema
      )

    return self._generate_impl(
        all_input_ids=all_input_ids,
        max_prompt_length=max_prompt_length,
        max_generation_steps=max_generation_steps,
        echo=echo,
        return_logits=return_logits,
        return_logprobs=return_logprobs,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        beam_size=beam_size,
        seed=seed,
        pad_output=pad_output,
        processed_images=processed_images,
        constraint_tables=constraint_tables,
    )

  def generate_from_tokens(
      self,
      input_ids: np.ndarray | jnp.ndarray,
      max_generation_steps: int,
      *,
      forbidden_tokens: Iterable[int] | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      pad_output: bool = False,
      constraint_tables: constrained.ConstraintTables | None = None,
      constraint_pattern: str | None = None,
      constraint_schema: dict | None = None,
  ) -> base_sampler.SamplerOutput:
    """Generate from pre-tokenized, pre-padded token arrays.

    Bypasses tokenize() and host-side padding. Callers are responsible for:
      1. Tokenization (e.g. via BatchTokenizer or HuggingFace tokenizer)
      2. Left-padding to uniform prompt length
      3. Optionally transferring to device (arrays will be converted to
         jax arrays if they aren't already)

    This enables efficient pipelines where tokenization and device transfer
    happen on a background thread (e.g. via PrefetchDataLoader), while the
    accelerator is busy decoding the previous batch.

    Args:
      input_ids: Left-padded token IDs of shape [B, prompt_len]. Padding should
        use the tokenizer's pad_id.
      max_generation_steps: Maximum number of tokens to generate.
      forbidden_tokens: Token IDs that are disallowed during generation.
      temperature: Sampling temperature (0.0 = greedy).
      top_p: Nucleus sampling threshold.
      top_k: Top-k sampling threshold.
      beam_size: Beam size for beam search.
      seed: Random seed.
      echo: Whether to include the prompt in the output.
      return_logits: Whether to return per-step logits.
      return_logprobs: Whether to return per-step log probabilities.
      pad_output: Whether to pad output to maximum length.

    Returns:
      SamplerOutput with generated text, tokens, and optional logits/logprobs.
    """
    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

    # Ensure we have a numpy array for padded_prompt_tokens in the output.
    all_input_ids_np = np.asarray(input_ids)

    max_prompt_length = input_ids.shape[1]

    if constraint_tables is None and (
        constraint_pattern is not None or constraint_schema is not None
    ):
      constraint_tables = self.compile_constraint(
          pattern=constraint_pattern, schema=constraint_schema
      )

    return self._generate_impl(
        all_input_ids=all_input_ids_np,
        max_prompt_length=max_prompt_length,
        max_generation_steps=max_generation_steps,
        echo=echo,
        return_logits=return_logits,
        return_logprobs=return_logprobs,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        beam_size=beam_size,
        seed=seed,
        pad_output=pad_output,
        processed_images=None,
        constraint_tables=constraint_tables,
    )

  def _generate_impl(
      self,
      all_input_ids: np.ndarray,
      max_prompt_length: int,
      max_generation_steps: int,
      *,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      forbidden_token_ids: tuple[int, ...] | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: int | jax.Array | None = None,
      pad_output: bool = False,
      processed_images: jnp.ndarray | None = None,
      constraint_tables: constrained.ConstraintTables | None = None,
  ) -> base_sampler.SamplerOutput:
    """Core generation logic shared by __call__ and generate_from_tokens.

    Takes pre-tokenized, pre-padded numpy arrays and runs the full
    prefill → decode → extract output pipeline.

    Args:
      all_input_ids: Left-padded token IDs [B, max_prompt_length].
      max_prompt_length: Prompt length (with padding).
      max_generation_steps: Maximum tokens to generate.
      echo: Include prompt in output.
      return_logits: Return per-step logits.
      return_logprobs: Return per-step log probabilities.
      forbidden_token_ids: Disallowed token IDs.
      temperature: Sampling temperature.
      top_p: Nucleus sampling threshold.
      top_k: Top-k sampling threshold.
      beam_size: Beam search beam size.
      seed: Random seed (int, PRNGKey, or None).
      pad_output: Pad output to max length.
      processed_images: Pre-processed images, or None.

    Returns:
      SamplerOutput.
    """
    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
      raise ValueError(
          f'Total sampling steps {total_sampling_steps} must be less than the'
          f' cache size {self.cache_config.cache_size}.'
      )

    sampling_state = self.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
        include_logprobs=return_logprobs,
        constraint_tables=constraint_tables,
    )
    if constraint_tables is not None:
      compiled_prefill_fn = nnx.jit(self._prefill_fn, static_argnames=('echo',))
      compiled_decode_fn = nnx.jit(self._decode_fn)
    else:
      compiled_prefill_fn = self._compiled_prefill_fn
      compiled_decode_fn = self._compiled_decode_fn

    sampling_state = compiled_prefill_fn(
        self._flattened_transformer_state,
        sampling_state,
        processed_images,
        echo=echo,
    )
    sampling_state = compiled_decode_fn(
        self._flattened_transformer_state,
        sampling_state,
    )
    token_buffers = sampling_state.token_buffer
    logits_buffers = sampling_state.logits_buffer
    final_logprobs_buffer = sampling_state.logprobs_buffer

    if sampling_state.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
          sampling_state.logprobs_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      final_logprobs_buffer = updated_args['logprobs_buffer']
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state
    if pad_output:
      max_len = total_sampling_steps if echo else max_generation_steps
      lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
          token_buffers,
          logits_buffers,
          return_logits,
          echo,
          self.tokenizer.pad_id(),
          self.eos_tokens,
          max_prompt_length,
          max_len,
      )
      out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
      decoded_outputs = [
          self.tokenizer.decode(tokens[:length].tolist())
          for tokens, length in zip(out_tokens, lengths)
      ]
      out_logprobs: list[jax.Array] = []
      if return_logprobs:
        token_buffers = jax.device_get(token_buffers)
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
        for i in range(len(token_buffers)):
          start_idx = (
              utils.np_find_first_non_pad_idx(
                  token_buffers[i], self.tokenizer.pad_id()
              )
              if echo
              else max_prompt_length
          )
          end_idx = (
              utils.np_find_first_eos_idx(
                  token_buffers[i][max_prompt_length:], self.eos_tokens
              )
              + max_prompt_length
          )
          length = end_idx - start_idx
          # Slice logprobs and pad to max_len
          sliced_logprobs = final_logprobs_buffer[i][start_idx:end_idx]
          padded_logprobs = np.pad(
              sliced_logprobs,
              (0, max_len - length),
              mode='constant',
              constant_values=0.0,
          )
          out_logprobs.append(padded_logprobs.tolist())

    else:
      out_tokens: list[jax.Array] = []
      out_logits: list[jax.Array] = []
      out_logprobs: list[jax.Array] = []
      token_buffers = jax.device_get(token_buffers)
      if return_logprobs:
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
      if return_logits:
        logits_buffers = jax.device_get(logits_buffers)
      for i in range(len(token_buffers)):
        token_buffer = token_buffers[i]
        start_idx = (
            utils.np_find_first_non_pad_idx(
                token_buffer, self.tokenizer.pad_id()
            )
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.np_find_first_eos_idx(
                token_buffer[max_prompt_length:], self.eos_tokens
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        if return_logprobs:
          # Extract logprobs for the generated tokens
          out_logprobs.append(
              final_logprobs_buffer[i][start_idx:end_idx].tolist()
          )

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]

    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs if return_logprobs else None,
    )
    return result
