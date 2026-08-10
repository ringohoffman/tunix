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

"""Gemma4 model."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import enum
import functools
import inspect
import itertools
from typing import Literal, Self, TypeVar, TypedDict, overload

import flax
from flax import nnx
import flax.typing
import jax
from jax import checkpoint_policies as cp
from jax import numpy as jnp
from jax import shard_map
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
import numpy as np
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma4 import moe
from tunix.utils import compat, env_utils, sharding_utils
from typing_extensions import NotRequired

# JAX checkpoint policy type — matches nnx.remat's policy parameter.
CheckpointPolicy = Callable[..., bool]

env_utils.setup_sharding_environment()

_REMAT_SUPPORTS_GRAPH_UPDATES = (
    "graph_updates" in inspect.signature(nnx.remat).parameters
)


_F = TypeVar("_F", bound=Callable[..., object])


def _compat_remat(
    fn: _F,
    *,
    graph_updates: bool = True,
    policy: CheckpointPolicy | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
) -> _F:
  """nnx.remat wrapper that drops graph_updates if Flax doesn't support it."""
  kwargs = {}
  if static_argnums is not None:
    kwargs["static_argnums"] = static_argnums
  if _REMAT_SUPPORTS_GRAPH_UPDATES:
    return nnx.remat(fn, graph_updates=graph_updates, policy=policy, **kwargs)
  return nnx.remat(fn, policy=policy, **kwargs)


class LayerKV(TypedDict):
  """Key and value projection tensors for a single layer."""

  k: jaxtyping.Array
  """Key projection tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)."""
  v: jaxtyping.Array
  """Value projection tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)."""


class LayerCache(TypedDict):
  """Stateful pre-allocated KV cache buffer with position index for a single layer."""

  k: jaxtyping.Array
  """Pre-allocated key cache array of shape (batch_size, max_seq_len, num_kv_heads, head_dim)."""
  v: jaxtyping.Array
  """Pre-allocated value cache array of shape (batch_size, max_seq_len, num_kv_heads, head_dim)."""
  end_index: jaxtyping.Array
  """Current sequence length or insertion index array of shape (batch_size,)."""


class OriginKV(TypedDict):
  """Origin key and value projections for global and local layers in KV

  sharing.
  """

  global_origin: LayerKV
  """LayerKV projections from the global origin layer (shape (batch_size, seq_len, num_global_kv_heads, global_key_size))."""
  local_origin: LayerKV
  """LayerKV projections from the local origin layer (shape (batch_size, seq_len, num_kv_heads, head_dim))."""


Cache = dict[str, LayerCache]
StackedCache = tuple[LayerCache | None, ...]
TransientKVs = dict[str, tuple[jaxtyping.Array, jaxtyping.Array]]


class GemmaInput(TypedDict):
  """Input parameters for the Gemma4 model."""

  tokens: jaxtyping.Array
  """Input token IDs array of shape (batch_size, seq_len)."""
  positions: NotRequired[jaxtyping.Array | None]
  """Optional position indices array of shape (batch_size, seq_len)."""
  cache: NotRequired[Cache | StackedCache | None]
  """Optional existing KV cache dictionary or stacked cache tuple."""
  attention_mask: NotRequired[jaxtyping.Array | None]
  """Optional attention mask array."""
  decode_only_last_token: NotRequired[bool]
  """If True, only compute logits for the final token position."""
  segment_ids: NotRequired[jaxtyping.Array | splash.SegmentIds | None]
  """Optional document segment IDs for packed sequence attention masking."""
  target_indices: NotRequired[jaxtyping.Array | None]
  """Optional target token indices for sparse logit computation."""
  return_hidden_states: NotRequired[bool]
  """If True, return post-norm hidden states in GemmaOutput."""


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class GemmaOutput:
  """Output of the Gemma4 model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated KV cache, or None if no cache was provided.
    hidden_states: The hidden states of the model (post final norm, pre decode).
      Only populated when ``return_hidden_states=True``.

  This class supports tuple unpacking for backward compatibility::

      logits, cache = model(tokens, ...)  # still works
      out = model(tokens, ...)  # preferred
      out.logits, out.cache, out.hidden_states
  """

  logits: jax.Array
  cache: Cache | StackedCache | None = None
  hidden_states: jax.Array | None = None

  def tree_flatten(
      self,
  ) -> tuple[
      tuple[jax.Array, Cache | StackedCache | None, jax.Array | None],
      None,
  ]:
    children = (self.logits, self.cache, self.hidden_states)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: None,
      children: tuple[jax.Array, Cache | StackedCache | None, jax.Array | None],
  ) -> Self:
    return cls(*children)

  def __iter__(
      self,
  ) -> Iterator[jax.Array | Cache | StackedCache | None]:
    """Yield (logits, cache) for backward-compatible tuple unpacking."""
    yield self.logits
    yield self.cache

  @overload
  def __getitem__(
      self,
      idx: Literal[0],
  ) -> jax.Array:
    ...

  @overload
  def __getitem__(
      self,
      idx: Literal[1],
  ) -> Cache | StackedCache | None:
    ...

  @overload
  def __getitem__(
      self,
      idx: int,
  ) -> jax.Array | Cache | StackedCache | None:
    ...

  def __getitem__(self, idx: int) -> jax.Array | Cache | StackedCache | None:
    return (self.logits, self.cache)[idx]


class RematConfig(enum.Enum):
  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


@dataclasses.dataclass
class RematStrategy:
  """Unified control over rematerialization and activation offloading.

  Independently configures WHERE to place remat boundaries and WHAT
  to do with each intermediate value (recompute, save on device, or
  offload to host memory).

  Everything inside a remat boundary is recomputed by default.
  ``save_on_device`` and ``offload_to_host`` carve out exceptions.

  Examples::

      # Remat at block level, recompute everything (current BLOCK default)
      RematStrategy(boundary="block")

      # Remat at decoder level, offload matmul outputs to host
      RematStrategy(
        boundary="decoder",
        offload_to_host=["decoder_input"],
        offload_dots=True,
      )
  """

  # ── Boundary: WHERE to recompute ─────────────────────────────
  boundary: Literal["none", "block", "decoder"] = "block"

  # ── Policy: WHAT to save vs offload vs recompute ─────────────
  save_on_device: list[str] = dataclasses.field(default_factory=list[str])
  offload_to_host: list[str] = dataclasses.field(default_factory=list[str])
  offload_dots: bool = False

  # ── Advanced ─────────────────────────────────────────────────
  custom_policy: CheckpointPolicy | None = None
  offload_src: str = "device"
  offload_dst: str = "pinned_host"

  def build_policy(self) -> CheckpointPolicy | None:
    """Build a composed JAX checkpoint policy from this strategy.

    Returns ``None`` when no explicit policy is configured (default
    remat behaviour: recompute everything).
    """
    policies: list[CheckpointPolicy] = []

    if self.save_on_device or self.offload_to_host:
      policies.append(
          cp.save_and_offload_only_these_names(
              names_which_can_be_saved=self.save_on_device,
              names_which_can_be_offloaded=self.offload_to_host,
              offload_src=self.offload_src,
              offload_dst=self.offload_dst,
          )
      )

    if self.offload_dots:
      policies.append(
          cp.offload_dot_with_no_batch_dims(
              offload_src=self.offload_src,
              offload_dst=self.offload_dst,
          )
      )

    if self.custom_policy is not None:
      policies.append(self.custom_policy)

    if not policies:
      return None

    result = policies[0]
    for p in policies[1:]:
      result = cp.save_from_both_policies(result, p)
    return result


def _remat_config_to_strategy(
    config: RematConfig,
) -> RematStrategy:
  """Convert a legacy ``RematConfig`` enum value to a ``RematStrategy``."""
  _MAP: dict[RematConfig, RematStrategy] = {
      RematConfig.NONE: RematStrategy(boundary="none"),
      RematConfig.BLOCK: RematStrategy(boundary="block"),
      RematConfig.DECODER: RematStrategy(boundary="decoder"),
  }
  return _MAP.get(config, RematStrategy(boundary="none"))


def _get_remat_strategy(config: object) -> RematStrategy:
  """Resolve a ``RematStrategy`` from a model config.

  Supports both the new ``remat_strategy`` field and the legacy
  ``remat_config`` enum, with ``remat_strategy`` taking precedence.
  """
  strategy = getattr(config, "remat_strategy", None)
  if strategy is not None:
    return strategy
  remat_config = getattr(config, "remat_config", RematConfig.NONE)
  if isinstance(remat_config, int):
    remat_config = RematConfig(remat_config)
  return _remat_config_to_strategy(remat_config)


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for gemma transformer."""

  emb_vd: tuple[str | None, ...]
  q_weight_ndh: tuple[str | None, ...]
  kv_weight_cndh: tuple[str | None, ...]
  qkv_weight_cndh: tuple[str | None, ...]
  o_weight_nhd: tuple[str | None, ...]
  ffw_weight_df: tuple[str | None, ...]
  ffw_weight_fd: tuple[str | None, ...]
  rms_norm_weight: tuple[str | None, ...]
  act_btd: tuple[str | None, ...]
  act_btf: tuple[str | None, ...]
  act_btnh: tuple[str | None, ...]
  vision_proj: tuple[str | None, ...]
  vision_soft_emb_norm_weight: tuple[str | None, ...]
  # MoE sharding
  exp_weight_edf: tuple[str | None, ...]
  exp_weight_efd: tuple[str | None, ...]
  # PLE sharding
  per_layer_model_projection: tuple[str | None, ...]
  per_layer_input_gate: tuple[str | None, ...]
  per_layer_projection: tuple[str | None, ...]
  per_layer_input_embedding: tuple[str | None, ...]
  # Input data sharding (token IDs, masks, etc. of shape [B, T])
  input_bt: tuple[str | None, ...] = ("fsdp", None)

  @property
  def input_pspec(self) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(*self.input_bt)

  @staticmethod
  def get_default_sharding(is_sampling: bool = False) -> ShardingConfig:
    return ShardingConfig(
        emb_vd=("tp", "fsdp"),
        q_weight_ndh=("tp", "fsdp", None),
        kv_weight_cndh=(None, "tp", "fsdp", None),
        qkv_weight_cndh=(None, "tp", "fsdp", None),
        o_weight_nhd=("tp", None, "fsdp"),
        ffw_weight_df=("fsdp", "tp"),
        ffw_weight_fd=("tp", "fsdp"),
        rms_norm_weight=("tp",),
        act_btd=("fsdp", None, None if is_sampling else "tp"),
        act_btf=("fsdp", None, "tp"),
        act_btnh=("fsdp", None, "tp", None),
        vision_proj=("fsdp", "tp"),
        vision_soft_emb_norm_weight=("tp",),
        exp_weight_edf=("fsdp", None, None, "tp"),
        exp_weight_efd=("fsdp", "tp", None),
        per_layer_model_projection=("fsdp", None, "tp"),
        per_layer_input_gate=("fsdp", "tp"),
        per_layer_projection=("tp", "fsdp"),
        per_layer_input_embedding=("tp", None, "fsdp"),
    )

  @staticmethod
  def no_shard() -> ShardingConfig:
    return ShardingConfig(
        emb_vd=(None, None),
        q_weight_ndh=(None, None, None),
        kv_weight_cndh=(None, None, None, None),
        qkv_weight_cndh=(None, None, None, None),
        o_weight_nhd=(None, None, None),
        ffw_weight_df=(None, None),
        ffw_weight_fd=(None, None),
        rms_norm_weight=(None,),
        act_btd=(None, None, None),
        act_btf=(None, None, None),
        act_btnh=(None, None, None, None),
        vision_proj=(None, None),
        vision_soft_emb_norm_weight=(None,),
        exp_weight_edf=(None, None, None, None),
        exp_weight_efd=(None, None, None),
        per_layer_model_projection=(None, None, None),
        per_layer_input_gate=(None, None),
        per_layer_projection=(None, None),
        per_layer_input_embedding=(None, None, None),
        input_bt=(None, None),
    )

  def with_scan_axis(self) -> Self:
    """Return a copy with a leading ``None`` prepended to all weight specs.

    When ``use_scan_layers=True``, ``nnx.vmap`` stacks per-layer parameters
    along a new leading scan axis.  The stored ``nnx.Param(sharding=...)``
    annotation is applied to the *stacked* tensor, so a 4-axis spec like
    ``P(None, 'tp', 'fsdp', None)`` would be misapplied to a 5-dim tensor,
    causing an ``IndivisibleError`` at optimizer initialisation.

    This method prepends ``None`` to every *weight* sharding tuple (those that
    annotate ``nnx.Param`` inside ``DecoderLayer``, ``Attention``, and
    ``FeedForward``) while leaving activation specs unchanged (those are
    applied at runtime to tensors that are NOT stacked).
    """

    def _prepend(t: tuple[str | None, ...]) -> tuple[str | None, ...]:
      return (None,) + t

    return dataclasses.replace(
        self,
        # per-layer weight specs — gain an extra scan leading axis
        q_weight_ndh=_prepend(self.q_weight_ndh),
        kv_weight_cndh=_prepend(self.kv_weight_cndh),
        qkv_weight_cndh=_prepend(self.qkv_weight_cndh),
        o_weight_nhd=_prepend(self.o_weight_nhd),
        ffw_weight_df=_prepend(self.ffw_weight_df),
        ffw_weight_fd=_prepend(self.ffw_weight_fd),
        rms_norm_weight=_prepend(self.rms_norm_weight),
        vision_proj=_prepend(self.vision_proj),
        vision_soft_emb_norm_weight=_prepend(self.vision_soft_emb_norm_weight),
        exp_weight_edf=_prepend(self.exp_weight_edf),
        exp_weight_efd=_prepend(self.exp_weight_efd),
        per_layer_input_gate=_prepend(self.per_layer_input_gate),
        per_layer_projection=_prepend(self.per_layer_projection),
        # activation specs and embedder-level weights
        # (per_layer_model_projection, per_layer_input_embedding) are NOT
        # inside scan groups — leave them unchanged.
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  """Transformer config."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float = 30.0
  sliding_window_size: int | None = None
  per_layer_input_dim: int = 0
  num_global_kv_heads: int | None = None
  global_key_size: int = 512
  attention_pattern: tuple["AttentionType", ...] | None = None
  frac_shared_layers: float = 0.0
  global_rope_proportion: float = 0.25
  local_rope_proportion: float = 1.0
  k_eq_v_global: bool = False
  override_kv_shared_ffw_hidden: int | None = None

  local_base_frequency: int = 10_000
  global_base_frequency: int = 1_000_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0

  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.float32
  dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024
  use_sliding_window_kv_cache: bool = True

  # MoE config
  enable_moe: bool = False
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  expert_dim: int | None = None
  moe_dense_hidden_dim: int | None = None

  # Scan-over-layers: when True, uses nnx.scan over attention-pattern groups
  # instead of a Python for-loop, producing a while loop in HLO for tiled
  # scheduling and reduced memory fragmentation.
  use_scan_layers: bool = False

  def __post_init__(self) -> None:
    # TODO(tunix-dev): support flash attention with sliding window KV cache
    if self.use_sliding_window_kv_cache and self.use_flash_attention:
      raise ValueError(
          "Flash attention and sliding window KV cache are mutually exclusive."
      )

  @classmethod
  def gemma4_e2b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> Self:
    return cls(
        num_layers=35,
        num_embed=262144,
        embed_dim=1536,
        hidden_dim=1536 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=20.0 / 35,
        override_kv_shared_ffw_hidden=int(1536 * 4 * 2),
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_e4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> Self:
    return cls(
        num_layers=42,
        num_embed=262144,
        embed_dim=2560,
        hidden_dim=2560 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=18.0 / 42,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_31b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> Self:
    return cls(
        num_layers=60,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 4,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        num_global_kv_heads=4,
        sliding_window_size=1024,
        shd_config=sharding_config,
        k_eq_v_global=True,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_26b_a4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> Self:
    return cls(
        num_layers=30,
        num_embed=262144,
        embed_dim=2816,
        hidden_dim=2112,  # Dense shared MLP branch
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_global_kv_heads=2,
        sliding_window_size=1024,
        shd_config=sharding_config,
        enable_moe=True,
        num_experts=128,
        expert_dim=704,
        num_experts_per_tok=8,
        moe_dense_hidden_dim=2112,
        k_eq_v_global=True,
        global_rope_proportion=0.25,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      config: ModelConfig,
      rngs: nnx.Rngs,
  ) -> None:
    self.config = config
    self.vocab_size = config.num_embed
    self.embed_dim = config.embed_dim
    self.param_dtype = config.param_dtype

    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=self.param_dtype)(
            rngs.params(), (self.vocab_size, self.embed_dim)
        ),
        sharding=config.shd_config.emb_vd,
    )

    if config.per_layer_input_dim > 0:
      self.per_layer_model_projection = Einsum(
          einsum_str="BTD,DNP->BTNP",
          shape=(self.embed_dim, config.num_layers, config.per_layer_input_dim),
          sharding=config.shd_config.per_layer_model_projection,
          w_scale=(float(self.embed_dim) ** -0.5),
          rngs=rngs,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )

      self.per_layer_projection_norm = RMSNorm(
          config.per_layer_input_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )
      self.per_layer_input_embedding = nnx.Param(
          nnx.initializers.normal(dtype=self.param_dtype)(
              rngs.params(),
              (self.vocab_size, config.num_layers, config.per_layer_input_dim),
          ),
          sharding=config.shd_config.per_layer_input_embedding,
      )

  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = jnp.astype(x, self.config.dtype)
    x = sharding_utils.shard(x, self.config.shd_config.act_btd)
    return x

  def encode_per_layer_input(
      self, x: jaxtyping.Array, t: jaxtyping.Array
  ) -> jaxtyping.Array:
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
    )
    x = self.per_layer_model_projection(x)
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding[t]
    y *= jnp.sqrt(self.config.per_layer_input_dim).astype(y.dtype)
    return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)

  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.config.dtype)
    w = jnp.astype(self.input_embedding[...], self.config.dtype)
    return jnp.dot(x, w.T)


class Einsum(nnx.Module):
  """Einsum module."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      w_scale: float | None = None,
  ) -> None:
    self.einsum_str = einsum_str
    self.dtype = dtype
    self.w_scale = w_scale

    self.shape = shape
    self.expected_in_ndim = len(einsum_str.split(",")[0].strip())
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    w = self.w[...]
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    if (diff := self.expected_in_ndim - x.ndim) > 0:
      x = jnp.expand_dims(x, axis=tuple(range(1, 1 + diff)))
    return jnp.einsum(self.einsum_str, x, w)


def find_last_one_index(attn_mask: jaxtyping.Array) -> jaxtyping.Array:
  """Finds the index of the last (rightmost) '1' from attn_mask."""
  cache_len = attn_mask.shape[-1]

  # 1. check if the entire row is all zeros.
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)

  # 2. reverse the rows in the attn_mask
  reversed_matrix = attn_mask[:, :, ::-1]

  # 3. find the fist 1 from the right.
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)

  # 4. covert back to the original index
  last_one_index_original = cache_len - 1 - first_one_from_right

  # 5. return the final index, 0 for rows are all zeros.
  final_indices = jnp.where(
      all_zeros_mask,
      0,
      last_one_index_original,
  )

  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jaxtyping.Array,  # [B, seq_len, cache_len] seq_len=1 for decoding
    sliding_window_size: int,
) -> jaxtyping.Array:
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)

  # 1. compute the window start position
  window_start_pos = upper_index - sliding_window_size + 1

  # 2. create window mask
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]

  # 3. create causal mask
  causal_mask = abs_pos[None, :] <= upper_index[:, None]

  # 4. create final mask
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]  # [B, 1, cache_len]


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      sharding: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ) -> None:
    self.scale = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), (dim,)).astype(param_dtype),
        sharding=sharding.rms_norm_weight,
    )
    self.dtype = dtype

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)
    scale = jnp.expand_dims(self.scale[...], axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * scale
    return normed_inputs.astype(self.dtype)


def apply_rope(
    inputs: jaxtyping.Array,
    positions: jaxtyping.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jaxtyping.Array:
  """Applies RoPE.

  Let B denote batch size, L denote sequence length, N denote number of heads,
  and H denote head dimension. Note that H must be divisible by 2.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions:  Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.
    rope_proportion: The proportion of the head dimension to apply RoPE to.

  Returns:
    Array of shape [B, L, N, H].
  """
  head_dim = inputs.shape[-1]
  rope_angles = int(rope_proportion * head_dim // 2)
  nope_angles = head_dim // 2 - rope_angles
  freq_exponents = (2.0 / head_dim) * jnp.arange(
      0, rope_angles, dtype=jnp.float32
  )
  timescale = jnp.pad(
      base_frequency**freq_exponents,
      (0, nope_angles),
      mode="constant",
      constant_values=(0, jnp.inf),
  )

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


K_MASK = -2.3819763e38


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def create_kv_cache_sharing_patterns(
    num_layers: int,
    frac_shared_layers: float,
    share_global: bool,
    share_local: bool,
    attention_types: tuple[AttentionType, ...],
) -> list[int]:
  """Creates a list of layer indices for which KV cache is used."""
  kv_cache_sharing_patterns: list[int] = []
  num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)
  for i in range(num_layers):
    if i < num_unshared_layers:
      kv_cache_sharing_patterns.append(i)
    else:
      if attention_types[i] == AttentionType.GLOBAL and share_global:
        kv_cache_sharing_patterns.append(num_unshared_layers - 1)
      elif attention_types[i] == AttentionType.LOCAL_SLIDING and share_local:
        kv_cache_sharing_patterns.append(num_unshared_layers - 2)
      else:
        kv_cache_sharing_patterns.append(i)
  return kv_cache_sharing_patterns


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ) -> None:
    self.config = config
    self.rope_proportion = (
        config.global_rope_proportion
        if attn_type == AttentionType.GLOBAL
        else config.local_rope_proportion
    )
    self.attn_type = attn_type
    self.rope_base_frequency = (
        config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency
    )
    self.rope_scale_factor = (
        config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor
    )

    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    if attn_type == AttentionType.GLOBAL:
      if config.num_global_kv_heads is not None:
        self.num_kv_heads = config.num_global_kv_heads
      if config.global_key_size is not None:
        self.head_dim = config.global_key_size

    self.attn_vec_einsum = Einsum(
        einsum_str="BTNH,NHD->BTD",
        shape=(config.num_heads, self.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=config.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_einsum = Einsum(
        einsum_str="BTD,NDH->BTNH",
        shape=(config.num_heads, config.embed_dim, self.head_dim),
        rngs=rngs,
        sharding=config.shd_config.q_weight_ndh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    k_eq_v = (
        config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
    )
    if k_eq_v:
      self.k_einsum = Einsum(
          einsum_str="BSD,KDH->BSKH",
          shape=(
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=config.shd_config.q_weight_ndh,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    else:
      kv_sharding = (
          (None, None, "fsdp", None)
          if self.num_kv_heads == 1
          else config.shd_config.kv_weight_cndh
      )

      self.kv_einsum = Einsum(
          einsum_str="BSD,CKDH->CBSKH",
          shape=(
              2,
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=kv_sharding,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self._query_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self._key_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerKV | LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      kv_shared_cache: LayerKV | LayerCache | None = None,
      kv_override: LayerKV | LayerCache | None = None,
      use_kv_override: jaxtyping.Array | bool | None = None,
      skip_kv_projection: bool = False,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> tuple[
      LayerKV | LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    x = checkpoint_name(x, "residual_attn")
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = sharding_utils.shard(
        query_proj, self.config.shd_config.act_btnh
    )
    query_proj = self._query_norm(query_proj)
    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    if kv_shared_cache is not None:
      assert cache is None
      key_proj = kv_shared_cache["k"]
      value_proj = kv_shared_cache["v"]
    elif skip_kv_projection:
      # Shared layer in scan path: skip KV einsum, norms, and RoPE entirely.
      # Use zeros as placeholder — the jnp.where kv_override below will
      # substitute the origin layer's real KV for both attention and cache
      # output.  Avoids wasted decode FLOPs for models with
      # frac_shared_layers > 0 (~0.9% of total for e2b, ~1.2% for e4b).
      assert kv_override is not None, (
          "skip_kv_projection requires kv_override to provide the "
          "origin layer's KV values"
      )
      kv_shape = (x.shape[0], x.shape[1], self.num_kv_heads, self.head_dim)
      key_proj = jnp.zeros(kv_shape, dtype=x.dtype)
      value_proj = jnp.zeros(kv_shape, dtype=x.dtype)
    else:
      if hasattr(self, "k_einsum"):  # case where k_eq_v is True
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)

      key_proj = sharding_utils.shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = sharding_utils.shard(
          value_proj, self.config.shd_config.act_btnh
      )

      # Apply norms to computed KV
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

    new_cache: LayerKV | LayerCache
    if cache is not None:
      assert "end_index" in cache
      assert kv_shared_cache is None
      # Update cache with new kv projections
      cache_len = cache["v"].shape[1]
      end_index = cache["end_index"][0]
      if seq_len > cache_len:
        # Prompt longer than cache size (e.g. sliding window cache test)
        valid_len = cache_len
        latest_indices = jnp.arange(seq_len - valid_len, seq_len) % cache_len
        cache_v = (
            cache["v"]
            .at[:, latest_indices, ...]
            .set(value_proj[:, -valid_len:, ...])
        )
        cache_k = (
            cache["k"]
            .at[:, latest_indices, ...]
            .set(key_proj[:, -valid_len:, ...])
        )
        new_cache = {
            "v": cache_v,
            "k": cache_k,
            "end_index": cache["end_index"] + seq_len,
        }
      elif seq_len > 1:  # prefill (seq_len <= cache_len)
        slice_indices = (0, end_index % cache_len, 0, 0)
        cache_v = jax.lax.dynamic_update_slice(
            cache["v"], value_proj.astype(cache["v"].dtype), slice_indices
        )
        cache_k = jax.lax.dynamic_update_slice(
            cache["k"], key_proj.astype(cache["k"].dtype), slice_indices
        )
        new_cache = {
            "v": cache_v,
            "k": cache_k,
            "end_index": cache["end_index"] + seq_len,
        }
        value_proj = cache_v
        key_proj = cache_k
      else:  # decode (seq_len == 1)
        slice_indices = (0, end_index % cache_len, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache["v"], value_proj.astype(cache["v"].dtype), slice_indices
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache["k"], key_proj.astype(cache["k"].dtype), slice_indices
        )
        new_cache = {
            "v": value_proj,
            "k": key_proj,
            "end_index": cache["end_index"] + seq_len,
        }
    else:
      new_cache = {
          "v": value_proj,
          "k": key_proj,
      }

    # KV override for scan-based cache sharing.  When use_kv_override is
    # True (shared layer), swap K/V with the origin's cache for attention
    # and suppress the cache write so the shared layer's output cache
    # contains the origin's cache rather than its own wasted projections.
    if kv_override is not None and use_kv_override is not None:
      if cache is None or seq_len > cache["v"].shape[1]:
        # prefill without prior cache: key_proj is raw [B, seq_len, H, D]
        key_proj = jnp.where(
            use_kv_override, kv_override["k"][:, :seq_len], key_proj
        )
        value_proj = jnp.where(
            use_kv_override, kv_override["v"][:, :seq_len], value_proj
        )
      else:  # decode or cache-enabled prefill: key_proj is full cache [B, cache_len, H, D]
        key_proj = jnp.where(use_kv_override, kv_override["k"], key_proj)
        value_proj = jnp.where(use_kv_override, kv_override["v"], value_proj)
      if cache is not None:
        assert "end_index" in new_cache
        kv_override_end_index = (
            kv_override["end_index"]
            if kv_override is not None and "end_index" in kv_override
            else new_cache["end_index"]
        )
        new_cache = {
            "k": jnp.where(use_kv_override, kv_override["k"], new_cache["k"]),
            "v": jnp.where(use_kv_override, kv_override["v"], new_cache["v"]),
            "end_index": jnp.where(
                use_kv_override,
                kv_override_end_index,
                new_cache["end_index"],
            ),
        }

    if (
        self.config.use_flash_attention
        and seq_len > 1
        and cache is None
        and (seq_len % self.config.flash_attention_block_size == 0)
    ):
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = shd.get_abstract_mesh()
      if self.attn_type == AttentionType.LOCAL_SLIDING:
        assert self.config.sliding_window_size is not None
        mask = mask_lib.LocalMask(
            (seq_len, seq_len),
            window_size=(self.config.sliding_window_size - 1, 0),
            offset=0,
        )
      else:
        mask = mask_lib.CausalMask((seq_len, seq_len))

      multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
      if (
          mesh is not None
          and shd_b is not None
          and shd_b in mesh.shape
          and b % mesh.shape[shd_b] != 0
      ):
        shd_b = None
      head_shards = (
          mesh.shape[shd_n]
          if mesh is not None and shd_n is not None and shd_n in mesh.shape
          else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t]
          if mesh is not None and shd_t is not None and shd_t in mesh.shape
          else 1
      )

      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
          interpret=(jax.devices()[0].platform == "cpu"),
      )

      shd_spec = P(shd_b, shd_n, shd_t, shd_h)
      shd_n_kv = (
          shd_n
          if mesh is not None
          and shd_n is not None
          and shd_n in mesh.shape
          and kh % mesh.shape[shd_n] == 0
          else None
      )
      unsharded_seq_kv = P(shd_b, shd_n_kv, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      if segment_ids is not None:
        seg_spec = P(shd_b, shd_t)
        unsharded_seg_spec = P(shd_b, None)

        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                kernel_spec,
                shd_spec,
                unsharded_seq_kv,
                unsharded_seq_kv,
                seg_spec,
                unsharded_seg_spec,
            ),
            out_specs=shd_spec,
            check_vma=False,
        )
        def sharded_splash_attn_with_seg(
            kernel: splash.SplashAttentionKernel,
            q_block: jaxtyping.Array,
            k_block: jaxtyping.Array,
            v_block: jaxtyping.Array,
            q_seg_block: jaxtyping.Array,
            kv_seg_block: jaxtyping.Array,
        ) -> jaxtyping.Array:
          def _single_batch_kernel(
              q: jaxtyping.Array,
              k: jaxtyping.Array,
              v: jaxtyping.Array,
              q_seg: jaxtyping.Array,
              kv_seg: jaxtyping.Array,
          ) -> splash.SplashCustomReturnType:
            prefix_id: int | None = (
                int(_prefix_segment_id)
                if _prefix_segment_id is not None
                and isinstance(_prefix_segment_id, (int, np.integer))
                else (0 if _prefix_segment_id is not None else None)
            )
            seg_ids = splash.SegmentIds(
                q=q_seg,
                kv=kv_seg,
                prefix_segment_id=prefix_id,
            )
            return kernel(q, k, v, segment_ids=seg_ids)

          result = jax.vmap(_single_batch_kernel)(
              q_block, k_block, v_block, q_seg_block, kv_seg_block
          )
          # vmap(kernel) returns SplashCustomReturnType (Array | tuple);
          # without residuals it's always a plain Array.
          assert isinstance(result, jax.Array)
          return result

        if isinstance(segment_ids, splash.SegmentIds):
          q_seg = segment_ids.q
          kv_seg = segment_ids.kv
          _prefix_segment_id = segment_ids.prefix_segment_id
        else:
          assert segment_ids is not None
          q_seg = segment_ids
          kv_seg = segment_ids
          _prefix_segment_id = None

        qkv = sharded_splash_attn_with_seg(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
            q_seg,
            kv_seg,
        )
      else:

        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                kernel_spec,
                shd_spec,
                unsharded_seq_kv,
                unsharded_seq_kv,
            ),
            out_specs=shd_spec,
            check_vma=False,
        )
        def sharded_splash_attn(
            kernel: splash.SplashAttentionKernel,
            q_block: jaxtyping.Array,
            k_block: jaxtyping.Array,
            v_block: jaxtyping.Array,
        ) -> jaxtyping.Array:
          result = jax.vmap(kernel)(q_block, k_block, v_block)
          assert isinstance(result, jax.Array)
          return result

        qkv = sharded_splash_attn(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
        )
      encoded = qkv.transpose(0, 2, 1, 3)
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

    else:
      if self.use_gqa:
        b, t, kg, h = query_proj.shape
        n_groups = kg // self.num_kv_heads
        query_reshaped = query_proj.reshape(
            (b, t, self.num_kv_heads, n_groups, h)
        )
        logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_reshaped, key_proj)
        b, t, k, g, s = logits.shape
        logits = logits.reshape((b, t, k * g, s))
      else:
        logits = jnp.einsum("BTNH,BSNH->BTNS", query_proj, key_proj)

      assert attn_mask is not None, "attn_mask required for non-flash path"
      if attn_mask is not None:
        if cache is None or seq_len > cache["v"].shape[1]:
          # Only compute attention scores for the actual sequence length when not
          # using a cache-backed representation.
          attn_mask = attn_mask[..., :seq_len]
        elif attn_mask.shape[-1] < key_proj.shape[1]:
          padding = key_proj.shape[1] - attn_mask.shape[-1]
          attn_mask = jnp.pad(
              attn_mask,
              (*((0, 0) for _ in range(attn_mask.ndim - 1)), (0, padding)),
          )
        elif attn_mask.shape[-1] > key_proj.shape[1]:
          attn_mask = attn_mask[..., : key_proj.shape[1]]

      if self.attn_type == AttentionType.LOCAL_SLIDING:
        if (
            segment_pos.shape[1] == 1
            and self.config.use_sliding_window_kv_cache
        ):
          # for decoding with sliding window cache
          active_cache = cache if cache is not None else kv_shared_cache
          if active_cache is None:
            raise ValueError(
                "Cache or shared cache is required for local sliding attention"
                " in decoding."
            )
          cache_len = key_proj.shape[1]
          assert "end_index" in active_cache
          end_idx = active_cache["end_index"]
          if cache is None and kv_shared_cache is not None:
            # In case of shared KV cache, the origin layer already updated the
            # end index. We need to subtract 1 to get the correct end index of
            # the previous token.
            end_idx = end_idx - 1
          end_idx = end_idx[:, None, None]
          p = jnp.arange(cache_len)[None, None, :]

          # map physical index to logical index
          logical_indices = end_idx - ((end_idx - p) % cache_len)

          # identify uninitialized slots (before the cache fills up)
          valid_physical = logical_indices >= 0
          logical_indices = jnp.maximum(0, logical_indices)

          attn_mask = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
          attn_mask = attn_mask * valid_physical
        elif segment_pos.shape[1] == 1:
          # for decoding without sliding window cache
          assert self.config.sliding_window_size is not None
          sliding_mask = create_sliding_window_mask(
              attn_mask,
              sliding_window_size=self.config.sliding_window_size,
          )
          attn_mask = sliding_mask * attn_mask
        else:  # for prefill
          assert self.config.sliding_window_size is not None
          all_ones = jnp.ones_like(attn_mask)
          sliding_mask = jnp.triu(
              all_ones, -1 * self.config.sliding_window_size + 1
          ) * jnp.tril(all_ones, self.config.sliding_window_size - 1)
          attn_mask = sliding_mask * attn_mask

      attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )

      if self.use_gqa:
        b, t, kg, s = attn.shape
        n_groups = kg // self.num_kv_heads
        probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
        encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs_reshaped, value_proj)
        b, t, k, g, h = encoded.shape
        encoded = encoded.reshape((b, t, k * g, h))
      else:
        encoded = jnp.einsum("BTNS,BSNH->BTNH", attn, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = sharding_utils.shard(
        attn_output, self.config.shd_config.act_btd
    )
    return new_cache, attn_output, (key_proj, value_proj)

  @property
  def use_gqa(self) -> bool:
    return self.num_kv_heads != self.config.num_heads and self.num_kv_heads > 1

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerKV | LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      kv_shared_cache: LayerKV | LayerCache | None = None,
      kv_override: LayerKV | LayerCache | None = None,
      use_kv_override: jaxtyping.Array | bool | None = None,
      skip_kv_projection: bool = False,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> tuple[
      LayerKV | LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    remat_config = self.config.remat_config
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument. graph_updates=False prevents TraceContextError
      # when mutating params across jax transformation trace levels.
      return _compat_remat(
          self.block.__func__, graph_updates=False, static_argnums=(8,)
      )(
          self,
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache,
          kv_override,
          use_kv_override,
          skip_kv_projection,
          segment_ids,
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          kv_override=kv_override,
          use_kv_override=use_kv_override,
          skip_kv_projection=skip_kv_projection,
          segment_ids=segment_ids,
      )

  @nnx.jit(static_argnames=("batch_size", "max_seq_len", "dtype"))
  def init_cache(
      self, batch_size: int, max_seq_len: int, dtype: jnp.dtype
  ) -> LayerCache:
    cache_len = max_seq_len
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
        and self.config.sliding_window_size is not None
    ):
      cache_len = min(max_seq_len, self.config.sliding_window_size)

    cache_shape = (batch_size, cache_len, self.num_kv_heads, self.head_dim)
    k = sharding_utils.shard(
        jnp.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
    )
    v = sharding_utils.shard(
        jnp.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
    )
    end_index = sharding_utils.shard(
        jnp.zeros((batch_size,), jnp.int32),
        self.config.shd_config.act_btnh[:1],
    )
    return {"k": k, "v": v, "end_index": end_index}


class FeedForward(nnx.Module):
  """Feed forward module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ) -> None:
    self.config = config
    h_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
    self.gate_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.up_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        h_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), config.shd_config.ffw_weight_fd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = checkpoint_name(x, "residual_ffw")
    return self.down_proj(nnx.gelu(self.gate_proj(x)) * self.up_proj(x))

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    remat_config = self.config.remat_config
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      return _compat_remat(self.block.__func__, graph_updates=False)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  """Decoder layer."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ) -> None:
    self.config = config
    self.pre_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.attn = Attention(
        config=config,
        attn_type=attn_type,
        rngs=rngs,
    )
    self.post_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.pre_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.mlp = FeedForward(config=config, hidden_dim=hidden_dim, rngs=rngs)

    if config.enable_moe:
      self.moe_pre_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.moe = moe.MoERagged(
          config=config,
          rngs=rngs,
      )
      self.moe_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.dense_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self.post_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    if config.per_layer_input_dim > 0:
      self.per_layer_input_gate = Einsum(
          einsum_str="BTD,DP->BTP",
          shape=(config.embed_dim, config.per_layer_input_dim),
          sharding=config.shd_config.per_layer_input_gate,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.per_layer_projection = Einsum(
          einsum_str="BTP,PD->BTD",
          shape=(config.per_layer_input_dim, config.embed_dim),
          sharding=config.shd_config.per_layer_projection,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.post_per_layer_input_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

    self.skip_scale = nnx.Param(jnp.ones((1,), dtype=config.param_dtype))

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerKV | LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      per_layer_input: jaxtyping.Array | None = None,
      kv_shared_cache: LayerKV | LayerCache | None = None,
      kv_override: LayerKV | LayerCache | None = None,
      use_kv_override: jaxtyping.Array | bool | None = None,
      skip_kv_projection: bool = False,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> tuple[
      LayerKV | LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    x = checkpoint_name(x, "decoder_input")
    norm = self.pre_attention_norm(x)
    cache, attn, kv = self.attn(
        norm,
        segment_pos,
        cache,
        attn_mask,
        kv_shared_cache=kv_shared_cache,
        kv_override=kv_override,
        use_kv_override=use_kv_override,
        skip_kv_projection=skip_kv_projection,
        segment_ids=segment_ids,
    )
    attn = self.post_attention_norm(attn)
    attn += x

    norm_ffw = self.pre_ffw_norm(attn)
    ffw = self.mlp(norm_ffw)
    if self.config.enable_moe:
      ffw = self.dense_post_ffw_norm(ffw)
      moe_norm_ffw = self.moe_pre_ffw_norm(attn)
      moe_out = self.moe(moe_norm_ffw, router_input=attn)
      moe_out = self.moe_post_ffw_norm(moe_out)
      ffw += moe_out
    ffw = self.post_ffw_norm(ffw)

    ffw += attn

    if self.config.per_layer_input_dim > 0 and per_layer_input is not None:
      gating_input = ffw
      mapped = self.per_layer_input_gate(gating_input)
      mapped = jax.nn.gelu(mapped) * per_layer_input
      mapped = self.per_layer_projection(mapped)
      mapped = self.post_per_layer_input_norm(mapped)
      ffw += mapped

    ffw = ffw * self.skip_scale[...]
    return cache, ffw, kv

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerKV | LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      per_layer_input: jaxtyping.Array | None = None,
      kv_shared_cache: LayerKV | LayerCache | None = None,
      kv_override: LayerKV | LayerCache | None = None,
      use_kv_override: jaxtyping.Array | bool | None = None,
      skip_kv_projection: bool = False,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> tuple[
      LayerKV | LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    remat_config = self.config.remat_config
    if (
        remat_config == RematConfig.DECODER
        or remat_config == RematConfig.DECODER.value
    ):
      return _compat_remat(
          self.block.__func__, graph_updates=False, static_argnums=(9,)
      )(
          self,
          x,
          segment_pos,
          cache,
          attn_mask,
          per_layer_input,
          kv_shared_cache,
          kv_override,
          use_kv_override,
          skip_kv_projection,
          segment_ids,
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          per_layer_input,
          kv_shared_cache,
          kv_override=kv_override,
          use_kv_override=use_kv_override,
          skip_kv_projection=skip_kv_projection,
          segment_ids=segment_ids,
      )

  @nnx.jit(static_argnames=("batch_size", "max_seq_len", "dtype"))
  def init_cache(
      self, batch_size: int, max_seq_len: int, dtype: jnp.dtype
  ) -> LayerCache:
    return self.attn.init_cache(batch_size, max_seq_len, dtype)


class ScanLayerGroup(nnx.Module):
  """A group of DecoderLayers matching one full attention pattern cycle.

  For Gemma4 31B with pattern (L, L, L, L, L, G), each group contains
  6 layers. When used with nnx.scan, XLA compiles the group body once
  and executes it N times (where N = num_layers / pattern_length),
  producing a tiled schedule with regular memory behavior.
  """

  def __init__(
      self,
      config: ModelConfig,
      pattern: tuple[AttentionType, ...],
      *,
      hidden_dim: int | None = None,
      skip_kv_projection: bool = False,
      rngs: nnx.Rngs,
  ) -> None:
    self.config = config
    self.pattern = pattern
    self.skip_kv_projection = skip_kv_projection
    self.sub_layers = compat.ModuleList[DecoderLayer]()
    hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
    for attn_type in pattern:
      self.sub_layers.append(
          DecoderLayer(
              config=config,
              attn_type=attn_type,
              hidden_dim=hidden_dim,
              rngs=rngs,
          )
      )

  def __call__(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      attn_mask: jaxtyping.Array | None,
      cache: (
          tuple[LayerCache | None, ...]
          | list[LayerCache | None]
          | StackedCache
          | None
      ) = None,
      origin_kv: tuple[LayerCache, ...] | None = None,
      is_shared: jaxtyping.Array | None = None,
      origin_sub_indices: list[int] | None = None,
      per_layer_inputs: jaxtyping.Array | None = None,
      new_cache: dict[int, LayerCache] | None = None,
      new_group_kvs: dict[int, LayerKV] | None = None,
      origin_kv_global: LayerKV | LayerCache | None = None,
      origin_kv_local: LayerKV | LayerCache | None = None,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> jaxtyping.Array:
    """Run one pattern-group of layers with full scan compatibility."""
    for sub_idx, layer in enumerate(self.sub_layers):
      layer_cache = (
          cache[sub_idx] if cache is not None and sub_idx < len(cache) else None
      )

      # Resolve KV override for scan-based cache sharing.
      kv_override = None
      use_kv_override = None
      if origin_kv is not None and is_shared is not None:
        assert origin_sub_indices is not None
        origin_s = origin_sub_indices[sub_idx]
        kv_override = origin_kv[origin_s]
        use_kv_override = is_shared[sub_idx]
      elif origin_kv_global is not None and origin_kv_local is not None:
        attn_type = self.pattern[sub_idx]
        if attn_type == AttentionType.GLOBAL:
          kv_override = origin_kv_global
        else:
          kv_override = origin_kv_local
        use_kv_override = jnp.array(True)

      pli = (
          per_layer_inputs[:, :, sub_idx, :]
          if per_layer_inputs is not None
          else None
      )

      layer_cache, x, kv = layer(
          x,
          positions,
          layer_cache,
          attn_mask,
          per_layer_input=pli,
          kv_override=kv_override,
          use_kv_override=use_kv_override,
          skip_kv_projection=self.skip_kv_projection,
          segment_ids=segment_ids,
      )
      if new_cache is not None and layer_cache is not None:
        assert "end_index" in layer_cache
        new_cache[sub_idx] = layer_cache
      if new_group_kvs is not None and kv is not None:
        new_group_kvs[sub_idx] = {"k": kv[0], "v": kv[1]}

    return x


class Gemma4(BackendMappingMixin, nnx.Module):
  """Gemma4 model."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
    self.config = config
    self.embedder = Embedder(config, rngs=rngs)

    pattern = (
        config.attention_pattern
        if config.attention_pattern
        else GEMMA4_ATTENTION_PATTERN
    )
    attention_types = [
        attn_type
        for _, attn_type in zip(
            range(config.num_layers), itertools.cycle(pattern)
        )
    ]
    self.kv_cache_sharing_patterns = create_kv_cache_sharing_patterns(
        num_layers=config.num_layers,
        frac_shared_layers=config.frac_shared_layers,
        share_global=True,
        share_local=True,
        attention_types=tuple(attention_types),
    )
    # Layers that shared layers depend on.
    self.shared_layer_origins = {
        j for i, j in enumerate(self.kv_cache_sharing_patterns) if i != j
    }

    if config.use_scan_layers:
      self._init_scan_layers(config, pattern, rngs)
    else:
      self._init_loop_layers(config, attention_types, rngs)

    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _init_loop_layers(
      self,
      config: ModelConfig,
      attention_types: list[AttentionType],
      rngs: nnx.Rngs,
  ) -> None:
    """Original for-loop layer initialization."""
    self.layers = compat.ModuleList[DecoderLayer]()
    for i in range(config.num_layers):
      attn_type = attention_types[i]
      h_dim = config.hidden_dim
      if (
          self.kv_cache_sharing_patterns[i] != i
          and config.override_kv_shared_ffw_hidden is not None
      ):
        h_dim = config.override_kv_shared_ffw_hidden
      self.layers.append(
          DecoderLayer(
              config=config, attn_type=attn_type, hidden_dim=h_dim, rngs=rngs
          )
      )

  def _init_scan_layers(
      self,
      config: ModelConfig,
      pattern: tuple[AttentionType, ...],
      rngs: nnx.Rngs,
  ) -> None:
    """Scan-based layer initialization using nnx.vmap over pattern groups.

    Two-phase approach to get sharding right:

    Phase 1 (inside vmap body): use the **original** config so that
    ``nnx.Param(sharding=spec)`` fires ``with_sharding_constraint`` on the
    *per-instance* value with the correct per-instance rank.  Using a
    scan-axis-prepended spec here would cause::

        ValueError: spec P(None, 'tp') requires rank ≥ 2 but value has rank 1

    Phase 2 (post-vmap): ``nnx.vmap`` stacks all params along a new leading
    axis of size ``num_scan_groups``.  The stored ``out_sharding`` metadata
    still reflects the per-instance rank, so when the optimizer later calls
    ``with_sharding_constraint`` on the *stacked* value it gets::

        IndivisibleError: spec P(None, 'fsdp', None) axis 1 partitioned 32
        times, but stacked dim size is 16 (shape: (10, 2, 16, ...))

    We fix this by walking ``scan_groups`` after vmap and prepending ``None``
    to every tuple-style ``out_sharding`` annotation.  The prepended ``None``
    tells JAX that the scan axis is replicated / unsharded.
    """
    pattern_len = len(pattern)
    num_unshared_layers = int(
        config.num_layers - config.frac_shared_layers * config.num_layers
    )
    num_shared_layers = config.num_layers - num_unshared_layers

    if (
        num_unshared_layers % pattern_len != 0
        or num_shared_layers % pattern_len != 0
    ):
      raise ValueError(
          "use_scan_layers requires num_unshared_layers"
          f" ({num_unshared_layers}) and num_shared_layers"
          f" ({num_shared_layers}) to be divisible by pattern length"
          f" ({pattern_len})."
      )

    self.num_unshared_groups = num_unshared_layers // pattern_len
    self.num_shared_groups = num_shared_layers // pattern_len
    self.num_scan_groups = config.num_layers // pattern_len
    self.scan_pattern = pattern

    # Unshared scan groups
    @nnx.split_rngs(splits=self.num_unshared_groups)
    @nnx.vmap(axis_size=self.num_unshared_groups)
    def create_unshared_group(rngs: nnx.Rngs) -> ScanLayerGroup:
      return ScanLayerGroup(
          config, pattern, hidden_dim=config.hidden_dim, rngs=rngs
      )

    self.unshared_scan_groups = create_unshared_group(rngs)
    if self.num_shared_groups == 0:
      self.scan_groups = self.unshared_scan_groups

    for _, var in nnx.iter_graph(self.unshared_scan_groups):
      if not isinstance(var, nnx.Param):
        continue
      spec = var.get_metadata("out_sharding", None)
      if isinstance(spec, tuple):
        var.set_metadata("out_sharding", (None,) + spec)

    # Shared scan groups (if frac_shared_layers > 0)
    if self.num_shared_groups > 0:
      shared_hdim = (
          config.override_kv_shared_ffw_hidden
          if config.override_kv_shared_ffw_hidden is not None
          else config.hidden_dim
      )

      @nnx.split_rngs(splits=self.num_shared_groups)
      @nnx.vmap(axis_size=self.num_shared_groups)
      def create_shared_group(rngs: nnx.Rngs) -> ScanLayerGroup:
        return ScanLayerGroup(
            config,
            pattern,
            hidden_dim=shared_hdim,
            skip_kv_projection=True,
            rngs=rngs,
        )

      self.shared_scan_groups = create_shared_group(rngs)

      for _, var in nnx.iter_graph(self.shared_scan_groups):
        if not isinstance(var, nnx.Param):
          continue
        spec = var.get_metadata("out_sharding", None)
        if isinstance(spec, tuple):
          var.set_metadata("out_sharding", (None,) + spec)

  def forward_backbone(
      self,
      tokens: jaxtyping.Array,
      positions: jaxtyping.Array | None = None,
      cache: Cache | StackedCache | None = None,
      attention_mask: jaxtyping.Array | None = None,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
  ) -> tuple[jaxtyping.Array, Cache | StackedCache | None]:
    """Forward pass through the backbone only (embed → layers → final norm).

    Returns the post-norm hidden states ``[B, L, D]`` without projecting
    through the vocabulary embedding.  Subclasses can call this to attach
    their own task-specific heads without paying for the LM decode.

    Args:
      tokens: Input token IDs, shape ``[B, L]``.
      positions: RoPE position indices, shape ``[B, L]``. Computed from
        ``tokens`` if not provided.
      cache: KV cache dict, or ``None`` for training / prefill without cache.
      attention_mask: Causal attention mask.
      segment_ids: Accepted for RL pipeline compatibility; currently unused.

    Returns:
      A tuple of ``(hidden_states, cache)`` where ``hidden_states`` has
      shape ``[B, L, D]`` and ``cache`` is the updated KV cache (or
      ``None`` if no cache was provided).
    """
    if positions is None:
      B, T = tokens.shape  # pylint: disable=invalid-name
      positions = jnp.tile(jnp.arange(T)[None, :], (B, 1))

    return_cache = cache is not None
    new_cache: Cache = {}
    x = self.embedder.encode(tokens)

    per_layer_inputs = None
    if self.config.per_layer_input_dim > 0:
      per_layer_inputs = self.embedder.encode_per_layer_input(x, tokens)

    transient_kvs: TransientKVs = {}
    is_prefill = tokens.shape[1] > 1

    forward_fn = (
        self._forward_scan
        if self.config.use_scan_layers
        else self._forward_loop
    )
    x, out_cache = forward_fn(
        x,
        positions,
        cache=cache,
        attention_mask=attention_mask,
        per_layer_inputs=per_layer_inputs,
        new_cache=new_cache,
        transient_kvs=transient_kvs,
        is_prefill=is_prefill,
        segment_ids=segment_ids,
    )
    x = self.final_norm(x)

    return x, out_cache if return_cache else None

  def __call__(
      self,
      tokens: jaxtyping.Array,
      positions: jaxtyping.Array | None = None,
      cache: Cache | StackedCache | None = None,
      attention_mask: jaxtyping.Array | None = None,
      decode_only_last_token: bool = False,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None = None,
      *,
      target_indices: jaxtyping.Array | None = None,
      return_hidden_states: bool = False,
  ) -> GemmaOutput:
    """Gemma4 forward pass.

    Args:
      tokens: Input token IDs, shape ``[B, L]``.
      positions: RoPE position indices, shape ``[B, L]``. Computed from
        ``tokens`` if not provided.
      cache: KV cache dict, or ``None`` for training / prefill without cache.
      attention_mask: Causal attention mask.
      decode_only_last_token: If ``True``, only decode the last sequence
        position. Kept for backward compatibility with the sampler.
      segment_ids: Accepted for RL pipeline compatibility; currently unused.
      target_indices: Optional ``[B, K]`` array of sequence-axis indices. When
        provided, only those hidden states are projected through the embedder
        decode, producing logits of shape ``[B, K, V]`` instead of ``[B, L,
        V]``. This is mutually exclusive with ``decode_only_last_token``.
      return_hidden_states: If ``True``, populate ``GemmaOutput.hidden_states``
        with the post-norm, pre-decode hidden states.

    Returns:
      A ``GemmaOutput`` with ``logits``, ``cache``, and optionally
      ``hidden_states``.
    """
    x, out_cache = self.forward_backbone(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    # Sparse gather: select specific hidden states before the expensive decode.
    if target_indices is not None:
      # target_indices shape: [B, K] — gather K positions per batch element.
      x = jnp.take_along_axis(x, target_indices[..., None], axis=1)
    elif decode_only_last_token:
      x = x[:, -1:, :]

    hidden_states_out = x if return_hidden_states else None
    logits = self.embedder.decode(x).astype(jnp.float32)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return GemmaOutput(
        logits=logits,
        cache=out_cache,
        hidden_states=hidden_states_out,
    )

  def _forward_loop(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: Cache | StackedCache | None,
      attention_mask: jaxtyping.Array | None,
      per_layer_inputs: jaxtyping.Array | None,
      new_cache: Cache,
      transient_kvs: TransientKVs,
      is_prefill: bool,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None,
  ) -> tuple[jaxtyping.Array, Cache]:
    """For-loop forward pass over layers with full feature parity and pre-merged scan layers."""
    num_layers = self.config.num_layers
    unrolled_layers: Sequence[DecoderLayer] = []
    if self.config.use_scan_layers:
      pattern_len = len(self.scan_pattern)
      num_unshared_layers = int(
          num_layers - self.config.frac_shared_layers * num_layers
      )
      unshared_splits = [
          nnx.split(sub_layer)
          for sub_layer in self.unshared_scan_groups.sub_layers
      ]
      shared_splits = (
          [
              nnx.split(sub_layer)
              for sub_layer in self.shared_scan_groups.sub_layers
          ]
          if self.num_shared_groups > 0
          else []
      )
      for i in range(num_layers):
        if i < num_unshared_layers:
          group_idx = i // pattern_len
          sub_idx = i % pattern_len
          graphdef, state = unshared_splits[sub_idx]
        else:
          rel_i = i - num_unshared_layers
          group_idx = rel_i // pattern_len
          sub_idx = rel_i % pattern_len
          graphdef, state = shared_splits[sub_idx]
        layer_state = jax.tree.map(lambda leaf: leaf[group_idx], state)
        unrolled_layers.append(nnx.merge(graphdef, layer_state))
    else:
      unrolled_layers = self.layers

    for i, layer in enumerate(unrolled_layers):
      layer_name = f"layer_{i}"
      shared_idx = self.kv_cache_sharing_patterns[i]
      is_shared = shared_idx != i
      kv_shared_cache: LayerKV | LayerCache | None
      if is_shared:
        assert shared_idx in self.shared_layer_origins
        layer_cache = None
        shared_layer_name = f"layer_{shared_idx}"
        if is_prefill:
          shared_k, shared_v = transient_kvs[shared_layer_name]
          kv_shared_cache = {"k": shared_k, "v": shared_v}
        else:
          kv_shared_cache = new_cache.get(shared_layer_name)
      else:
        layer_cache = cache[layer_name] if isinstance(cache, dict) else None
        kv_shared_cache = None

      layer_cache, x, layers_kvs = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          per_layer_input=per_layer_inputs[:, :, i, :]
          if per_layer_inputs is not None
          else None,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
      )
      if is_prefill and i in self.shared_layer_origins:
        transient_kvs[layer_name] = layers_kvs
      if not is_shared:
        if layer_cache is not None and "end_index" in layer_cache:
          new_cache[layer_name] = layer_cache
    return x, new_cache

  def _forward_scan(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: Cache | StackedCache | None,
      attention_mask: jaxtyping.Array | None,
      per_layer_inputs: jaxtyping.Array | None,
      new_cache: Cache,
      transient_kvs: TransientKVs,
      is_prefill: bool,
      segment_ids: jaxtyping.Array | splash.SegmentIds | None,
  ) -> tuple[jaxtyping.Array, Cache | StackedCache | None]:
    """Scan-based forward pass compiling into a single XLA while_loop in HLO (or unrolled for cache generation)."""
    num_layers = self.config.num_layers

    # When running cache inference (prefill or decode) with scan layers, run
    # through nnx.scan (compiles into a single while_loop HLO with O(1) graph
    # size) instead of unrolling all N layers into a flat HLO graph.
    if cache is not None:
      pattern_len = len(self.scan_pattern)
      num_scan_groups = num_layers // pattern_len
      seq_len = x.shape[1]

      # Reshape per_layer_inputs for scan: (B,T,N,D) -> (groups,B,T,pat,D)
      scan_per_layer_inputs = None
      if per_layer_inputs is not None:
        b, t, _, d = per_layer_inputs.shape
        reshaped = per_layer_inputs.reshape(
            (b, t, num_scan_groups, pattern_len, d)
        )
        shd_b, shd_t, _, _ = self.config.shd_config.act_btnh
        scan_per_layer_inputs = sharding_utils.shard(
            jnp.transpose(reshaped, (2, 0, 1, 3, 4)),
            (None, shd_b, shd_t, None, None),
        )

      is_stacked_cache = isinstance(cache, (tuple, list))
      if is_stacked_cache:
        scan_cache = cache
      else:
        # Stack initial dict layer caches across groups for each sub-layer.
        scan_cache_list: list[LayerCache | None] = []
        for sub_idx in range(pattern_len):
          group_layer_indices = [
              g * pattern_len + sub_idx for g in range(num_scan_groups)
          ]
          proto_i = next(
              (
                  i
                  for i in group_layer_indices
                  if self.kv_cache_sharing_patterns[i] == i
              ),
              None,
          )
          proto_cache = (
              cache[f"layer_{proto_i}"]
              if proto_i is not None and f"layer_{proto_i}" in cache
              else None
          )

          if proto_cache is not None:
            k_shape = proto_cache["k"].shape
            v_shape = proto_cache["v"].shape
            end_idx_shape = proto_cache["end_index"].shape
            k_dtype = proto_cache["k"].dtype
            v_dtype = proto_cache["v"].dtype
            end_idx_dtype = proto_cache["end_index"].dtype
            ks: list[jaxtyping.Array] = []
            vs: list[jaxtyping.Array] = []
            end_indices: list[jaxtyping.Array] = []
            for i in group_layer_indices:
              if (
                  self.kv_cache_sharing_patterns[i] == i
                  and f"layer_{i}" in cache
              ):
                c = cache[f"layer_{i}"]
                ks.append(c["k"])
                vs.append(c["v"])
                end_indices.append(c["end_index"])
              else:
                # Shared-layer slots in the stacked cache are zero-initialized
                # placeholders. The scan body writes to them, but the values
                # are never read because the kv_override path swaps in the
                # origin layer's real cache. Consumers must check
                # kv_cache_sharing_patterns before reading scan cache slots.
                ks.append(jnp.zeros(k_shape, dtype=k_dtype))
                vs.append(jnp.zeros(v_shape, dtype=v_dtype))
                end_indices.append(
                    jnp.zeros(end_idx_shape, dtype=end_idx_dtype)
                )

            scan_shd_btnh = (None, *self.config.shd_config.act_btnh)
            scan_shd_b = (None, *self.config.shd_config.act_btnh[:1])
            scan_cache_list.append({
                "k": sharding_utils.shard(jnp.stack(ks, axis=0), scan_shd_btnh),
                "v": sharding_utils.shard(jnp.stack(vs, axis=0), scan_shd_btnh),
                "end_index": sharding_utils.shard(
                    jnp.stack(end_indices, axis=0), scan_shd_b
                ),
            })
          else:
            scan_cache_list.append(None)

        scan_cache = tuple(scan_cache_list)

      # --- Compute sharing metadata for cross-group KV forwarding ---
      # origin_sub_indices: for each sub-layer position, which sub-position
      # holds its origin cache. Since the attention pattern is periodic across
      # scan groups, this is constant for all groups and can be inspected from
      # the last scan group.
      origin_sub_indices = [
          self.kv_cache_sharing_patterns[
              (num_scan_groups - 1) * pattern_len + s
          ]
          % pattern_len
          for s in range(pattern_len)
      ]
      is_origin_sub = [origin_sub_indices[s] != s for s in range(pattern_len)]
      is_shared_per_group = np.zeros(
          (num_scan_groups, pattern_len), dtype=np.bool_
      )
      is_origin_per_group = np.zeros(
          (num_scan_groups, pattern_len), dtype=np.bool_
      )
      has_sharing = False
      for g in range(num_scan_groups):
        for s in range(pattern_len):
          layer_idx = g * pattern_len + s
          origin_idx = self.kv_cache_sharing_patterns[layer_idx]
          if origin_idx != layer_idx:
            has_sharing = True
            is_shared_per_group[g, s] = True
            origin_g = origin_idx // pattern_len
            is_origin_per_group[origin_g, origin_sub_indices[s]] = True

      is_shared_jax: jaxtyping.Array | None = None
      is_origin_jax: jaxtyping.Array | None = None
      if has_sharing:
        if np.any(is_shared_per_group[: self.num_unshared_groups]):
          raise RuntimeError(
              "Unshared scan groups contain shared layers. Shared layers are"
              " only permitted in shared scan groups."
          )
        is_shared_jax = jnp.array(is_shared_per_group)
        is_origin_jax = jnp.array(is_origin_per_group)

        # Find which group contains the origin for each sub-layer.
        origin_group_for_sub = [0] * pattern_len
        for s in range(pattern_len):
          if is_origin_sub[s]:
            for g in range(num_scan_groups):
              if is_origin_per_group[g, s]:
                origin_group_for_sub[s] = g
                break

        # Initialize origin_kv carry from stacked cache.
        origin_kv_init_list: list[LayerCache] = []
        for s in range(pattern_len):
          c_s = scan_cache[s]
          if c_s is not None:
            g_idx = origin_group_for_sub[s]
            origin_kv_init_list.append({
                "k": c_s["k"][g_idx],
                "v": c_s["v"][g_idx],
                "end_index": c_s["end_index"][g_idx],
            })
          else:
            assert scan_cache[0] is not None
            c_zero = scan_cache[0]
            origin_kv_init_list.append({
                "k": jnp.zeros_like(c_zero["k"][0]),
                "v": jnp.zeros_like(c_zero["v"][0]),
                "end_index": jnp.zeros_like(c_zero["end_index"][0]),
            })
        origin_kv_init = tuple(origin_kv_init_list)
        sharing_metadata = (is_shared_jax, is_origin_jax)
        sharing_in_axis = (0, 0)
        carry_init = (x, origin_kv_init)
      else:
        sharing_metadata = None
        sharing_in_axis = None
        carry_init = x

      @nnx.scan(
          in_axes=(
              nnx.Carry,
              0,
              0,
              None,
              None,
              sharing_in_axis,
              0 if scan_per_layer_inputs is not None else None,
              None,
          ),
          out_axes=(
              nnx.Carry,
              tuple(
                  {"k": 0, "v": 0, "end_index": 0} for _ in range(pattern_len)
              ),
          ),
      )
      def scan_cache_body(
          carry: (
              tuple[jaxtyping.Array, tuple[LayerCache, ...]] | jaxtyping.Array
          ),
          group: ScanLayerGroup,
          group_cache: tuple[LayerCache | None, ...],
          positions: jaxtyping.Array,
          attn_mask: jaxtyping.Array | None,
          sharing_meta: tuple[jaxtyping.Array, jaxtyping.Array] | None,
          group_per_layer_inputs: jaxtyping.Array | None,
          segment_ids: jaxtyping.Array | splash.SegmentIds | None,
      ) -> tuple[
          tuple[jaxtyping.Array, tuple[LayerCache, ...]] | jaxtyping.Array,
          tuple[LayerCache, ...],
      ]:
        is_origin_slice: jaxtyping.Array | None = None
        if has_sharing:
          assert sharing_meta is not None
          assert isinstance(carry, tuple)
          x, origin_kv = carry
          is_shared_slice, is_origin_slice = sharing_meta
        else:
          assert isinstance(carry, jaxtyping.Array)
          x = carry
          origin_kv = None
          is_shared_slice = None

        new_group_cache: dict[int, LayerCache] = {}
        x = group(
            x,
            positions,
            attn_mask,
            cache=group_cache,
            origin_kv=origin_kv,
            is_shared=is_shared_slice,
            origin_sub_indices=(origin_sub_indices if has_sharing else None),
            per_layer_inputs=group_per_layer_inputs,
            new_cache=new_group_cache,
            segment_ids=segment_ids,
        )

        out_cache = tuple(new_group_cache[sub] for sub in range(pattern_len))

        if has_sharing:
          assert is_origin_slice is not None
          assert origin_kv is not None
          # Update origin_kv carry: for sub-positions that are origins in
          # this group, write their updated caches into the carry.
          updated_origin_kv: tuple[LayerCache, ...] = tuple(
              {
                  "k": jnp.where(
                      is_origin_slice[s],
                      new_group_cache[s]["k"],
                      origin_kv[s]["k"],
                  ),
                  "v": jnp.where(
                      is_origin_slice[s],
                      new_group_cache[s]["v"],
                      origin_kv[s]["v"],
                  ),
                  "end_index": jnp.where(
                      is_origin_slice[s],
                      new_group_cache[s]["end_index"],
                      origin_kv[s]["end_index"],
                  ),
              }
              for s in range(pattern_len)
          )
          return (x, updated_origin_kv), out_cache
        else:
          return x, out_cache

      if self.num_shared_groups == 0:
        result = scan_cache_body(
            carry_init,
            self.unshared_scan_groups,
            scan_cache,
            positions,
            attention_mask,
            sharing_metadata,
            scan_per_layer_inputs,
            segment_ids,
        )
        if has_sharing:
          assert isinstance(result[0], tuple)
          x_res_tuple: tuple[jaxtyping.Array, tuple[LayerCache, ...]] = result[
              0
          ]
          x = x_res_tuple[0]
          updated_scan_cache = result[1]
        else:
          assert isinstance(result[0], jaxtyping.Array)
          x = result[0]
          updated_scan_cache = result[1]
      else:
        unshared_scan_cache: tuple[LayerCache | None, ...] = tuple(
            LayerCache(
                k=c_s["k"][: self.num_unshared_groups],
                v=c_s["v"][: self.num_unshared_groups],
                end_index=c_s["end_index"][: self.num_unshared_groups],
            )
            if c_s is not None
            else None
            for c_s in scan_cache
        )
        shared_scan_cache: tuple[LayerCache | None, ...] = tuple(
            LayerCache(
                k=c_s["k"][self.num_unshared_groups :],
                v=c_s["v"][self.num_unshared_groups :],
                end_index=c_s["end_index"][self.num_unshared_groups :],
            )
            if c_s is not None
            else None
            for c_s in scan_cache
        )
        if has_sharing:
          assert is_shared_jax is not None and is_origin_jax is not None
          unshared_meta = (
              is_shared_jax[: self.num_unshared_groups],
              is_origin_jax[: self.num_unshared_groups],
          )
          shared_meta = (
              is_shared_jax[self.num_unshared_groups :],
              is_origin_jax[self.num_unshared_groups :],
          )
        else:
          unshared_meta = None
          shared_meta = None
        unshared_pli = (
            scan_per_layer_inputs[: self.num_unshared_groups]
            if scan_per_layer_inputs is not None
            else None
        )
        shared_pli = (
            scan_per_layer_inputs[self.num_unshared_groups :]
            if scan_per_layer_inputs is not None
            else None
        )

        res_unshared = scan_cache_body(
            carry_init,
            self.unshared_scan_groups,
            unshared_scan_cache,
            positions,
            attention_mask,
            unshared_meta,
            unshared_pli,
            segment_ids,
        )
        carry_mid = res_unshared[0]
        unshared_out_cache = res_unshared[1]

        res_shared = scan_cache_body(
            carry_mid,
            self.shared_scan_groups,
            shared_scan_cache,
            positions,
            attention_mask,
            shared_meta,
            shared_pli,
            segment_ids,
        )
        carry_final = res_shared[0]
        shared_out_cache = res_shared[1]

        if has_sharing:
          assert isinstance(carry_final, tuple)
          x_final_tuple: tuple[jaxtyping.Array, tuple[LayerCache, ...]] = (
              carry_final
          )
          x = x_final_tuple[0]
        else:
          assert isinstance(carry_final, jaxtyping.Array)
          x = carry_final
        updated_scan_cache: StackedCache = tuple(
            LayerCache(
                k=jnp.concatenate(
                    [unshared_out_cache[s]["k"], shared_out_cache[s]["k"]],
                    axis=0,
                ),
                v=jnp.concatenate(
                    [unshared_out_cache[s]["v"], shared_out_cache[s]["v"]],
                    axis=0,
                ),
                end_index=jnp.concatenate(
                    [
                        unshared_out_cache[s]["end_index"],
                        shared_out_cache[s]["end_index"],
                    ],
                    axis=0,
                ),
            )
            for s in range(pattern_len)
        )

      if is_stacked_cache:
        out_cache = updated_scan_cache
      else:
        # Unstack updated scan cache back into new_cache dict.
        for i in range(num_layers):
          if self.kv_cache_sharing_patterns[i] != i:
            continue  # Shared layers have no cache entry.

          group_idx = i // pattern_len
          sub_idx = i % pattern_len
          c = updated_scan_cache[sub_idx]
          new_cache[f"layer_{i}"] = {
              "k": c["k"][group_idx],
              "v": c["v"][group_idx],
              "end_index": c["end_index"][group_idx],
          }
        out_cache = new_cache
      return x, out_cache

    # --- True scan path (training, cache=None) ---
    pattern_len = len(self.scan_pattern)

    if self.config.frac_shared_layers == 0.0:
      # Single scan path for unshared models
      num_scan_groups = self.config.num_layers // pattern_len

      scan_per_layer_inputs = None
      if per_layer_inputs is not None:
        b, t, _, d = per_layer_inputs.shape
        reshaped = per_layer_inputs.reshape(
            (b, t, num_scan_groups, pattern_len, d)
        )
        shd_b, shd_t, _, _ = self.config.shd_config.act_btnh
        scan_per_layer_inputs = sharding_utils.shard(
            jnp.transpose(reshaped, (2, 0, 1, 3, 4)),
            (None, shd_b, shd_t, None, None),
        )

      @nnx.scan(
          in_axes=(
              nnx.Carry,
              0,
              None,
              None,
              0 if scan_per_layer_inputs is not None else None,
              None,
          ),
          out_axes=nnx.Carry,
      )
      def scan_body(
          x: jaxtyping.Array,
          group: ScanLayerGroup,
          positions: jaxtyping.Array,
          attn_mask: jaxtyping.Array | None,
          group_per_layer_inputs: jaxtyping.Array | None,
          segment_ids: jaxtyping.Array | splash.SegmentIds | None,
      ) -> jaxtyping.Array:
        return group(
            x,
            positions,
            attn_mask,
            per_layer_inputs=group_per_layer_inputs,
            segment_ids=segment_ids,
        )

      x = scan_body(
          x,
          self.unshared_scan_groups,
          positions,
          attention_mask,
          scan_per_layer_inputs,
          segment_ids,
      )
      return x, None

    else:
      # 2-Scan path for KV cache sharing models
      num_unshared_layers = int(
          self.config.num_layers
          - self.config.frac_shared_layers * self.config.num_layers
      )
      num_shared_layers = self.config.num_layers - num_unshared_layers
      num_unshared_groups = num_unshared_layers // pattern_len
      num_shared_groups = num_shared_layers // pattern_len

      scan_unshared_pli = None
      scan_shared_pli = None
      if per_layer_inputs is not None:
        b, t, _, d = per_layer_inputs.shape
        unshared_pli = per_layer_inputs[:, :, :num_unshared_layers, :]
        shared_pli = per_layer_inputs[:, :, num_unshared_layers:, :]

        reshaped_u = unshared_pli.reshape(
            (b, t, num_unshared_groups, pattern_len, d)
        )
        shd_b, shd_t, _, _ = self.config.shd_config.act_btnh
        scan_unshared_pli = sharding_utils.shard(
            jnp.transpose(reshaped_u, (2, 0, 1, 3, 4)),
            (None, shd_b, shd_t, None, None),
        )

        reshaped_s = shared_pli.reshape(
            (b, t, num_shared_groups, pattern_len, d)
        )
        scan_shared_pli = sharding_utils.shard(
            jnp.transpose(reshaped_s, (2, 0, 1, 3, 4)),
            (None, shd_b, shd_t, None, None),
        )

      global_sub_idx = (num_unshared_layers - 1) % pattern_len
      local_sub_idx = (num_unshared_layers - 2) % pattern_len

      def _make_dummy_kv(num_heads: int, head_dim: int) -> LayerKV:
        z = jnp.zeros(
            (x.shape[0], x.shape[1], num_heads, head_dim),
            dtype=x.dtype,
        )
        return {"k": z, "v": z}

      init_carry: tuple[jaxtyping.Array, OriginKV] = (
          x,
          {
              "global_origin": _make_dummy_kv(
                  self.config.num_global_kv_heads or self.config.num_kv_heads,
                  self.config.global_key_size or self.config.head_dim,
              ),
              "local_origin": _make_dummy_kv(
                  self.config.num_kv_heads,
                  self.config.head_dim,
              ),
          },
      )

      @nnx.scan(
          in_axes=(
              nnx.Carry,
              0,
              None,
              None,
              0 if scan_unshared_pli is not None else None,
              None,
          ),
          out_axes=nnx.Carry,
      )
      def scan_body_unshared(
          carry: tuple[jaxtyping.Array, OriginKV],
          group: ScanLayerGroup,
          positions: jaxtyping.Array,
          attn_mask: jaxtyping.Array | None,
          group_per_layer_inputs: jaxtyping.Array | None,
          segment_ids: jaxtyping.Array | splash.SegmentIds | None,
      ) -> tuple[jaxtyping.Array, OriginKV]:
        x_curr, _ = carry
        new_group_kvs: dict[int, LayerKV] = {}
        x_next = group(
            x_curr,
            positions,
            attn_mask,
            per_layer_inputs=group_per_layer_inputs,
            new_group_kvs=new_group_kvs,
            segment_ids=segment_ids,
        )
        return x_next, {
            "global_origin": new_group_kvs[global_sub_idx],
            "local_origin": new_group_kvs[local_sub_idx],
        }

      x, unshared_kvs = scan_body_unshared(
          init_carry,
          self.unshared_scan_groups,
          positions,
          attention_mask,
          scan_unshared_pli,
          segment_ids,
      )

      @nnx.scan(
          in_axes=(
              nnx.Carry,
              0,
              None,
              None,
              None,
              0 if scan_shared_pli is not None else None,
              None,
          ),
          out_axes=nnx.Carry,
      )
      def scan_body_shared(
          x: jaxtyping.Array,
          group: ScanLayerGroup,
          positions: jaxtyping.Array,
          attn_mask: jaxtyping.Array | None,
          origin_kvs: OriginKV,
          group_per_layer_inputs: jaxtyping.Array | None,
          segment_ids: jaxtyping.Array | splash.SegmentIds | None,
      ) -> jaxtyping.Array:
        return group(
            x,
            positions,
            attn_mask,
            origin_kv_global=origin_kvs["global_origin"],
            origin_kv_local=origin_kvs["local_origin"],
            per_layer_inputs=group_per_layer_inputs,
            segment_ids=segment_ids,
        )

      x = scan_body_shared(
          x,
          self.shared_scan_groups,
          positions,
          attention_mask,
          unshared_kvs,
          scan_shared_pli,
          segment_ids,
      )
      return x, None

  @nnx.jit(static_argnames=("batch_size", "max_seq_len", "dtype"))
  def init_cache(
      self,
      batch_size: int,
      max_seq_len: int,
      dtype: jnp.dtype,
  ) -> Cache | StackedCache:
    if self.config.use_scan_layers:
      pattern_len = len(self.scan_pattern)
      num_scan_groups = self.config.num_layers // pattern_len
      scan_cache_list: list[LayerCache | None] = []
      for sub_idx in range(pattern_len):
        group_layer_indices = [
            g * pattern_len + sub_idx for g in range(num_scan_groups)
        ]
        proto_i = next(
            (
                i
                for i in group_layer_indices
                if self.kv_cache_sharing_patterns[i] == i
            ),
            None,
        )
        if proto_i is not None:
          sub_layer = self.unshared_scan_groups.sub_layers[sub_idx]
          proto_cache = sub_layer.init_cache(batch_size, max_seq_len, dtype)
          shd_btnh = (None, *self.config.shd_config.act_btnh)
          shd_b = (None, *self.config.shd_config.act_btnh[:1])
          scan_cache_list.append({
              "k": sharding_utils.shard(
                  jnp.zeros(
                      (num_scan_groups, *proto_cache["k"].shape),
                      dtype=proto_cache["k"].dtype,
                  ),
                  shd_btnh,
              ),
              "v": sharding_utils.shard(
                  jnp.zeros(
                      (num_scan_groups, *proto_cache["v"].shape),
                      dtype=proto_cache["v"].dtype,
                  ),
                  shd_btnh,
              ),
              "end_index": sharding_utils.shard(
                  jnp.zeros(
                      (num_scan_groups, *proto_cache["end_index"].shape),
                      dtype=proto_cache["end_index"].dtype,
                  ),
                  shd_b,
              ),
          })
        else:
          scan_cache_list.append(None)
      return tuple(scan_cache_list)

    cache: Cache = {}
    for i, layer in enumerate(self.layers):
      if self.kv_cache_sharing_patterns[i] != i:
        continue  # skip shared layers.
      cache[f"layer_{i}"] = layer.init_cache(batch_size, max_seq_len, dtype)
    return cache

  def get_model_input(self) -> GemmaInput:
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 2
    return {
        "tokens": jnp.ones((dummy_batch_size, dummy_seq_len), dtype=jnp.int32),
        "positions": jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        "cache": None,
        "attention_mask": jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def num_embed(self) -> int:
    return self.config.num_embed
