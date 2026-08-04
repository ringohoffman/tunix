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

"""Classification heads and wrapper for Gemma4.

Provides head types (binary, multilabel) and a ``Gemma4ForClassification``
subclass that replaces the LM decode with a task-specific classification head.

Supports both unpacked (one example per row) and shared-prefix packed
(multiple examples sharing a system instruction per row via star attention).
"""

from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Self

import jax
import jax.numpy as jnp
import jaxtyping
from flax import nnx

from tunix.models.gemma4 import model as gemma4_model


class PoolStrategy(enum.Enum):
  LAST_TOKEN = "last_token"
  MEAN = "mean"


def pool_hidden_states(
    hidden: jaxtyping.Array,
    pool_mask: jaxtyping.Array,
    strategy: PoolStrategy = PoolStrategy.LAST_TOKEN,
) -> jaxtyping.Array:
  """Reduce ``[B, L, D]`` → ``[B, D]``."""
  if strategy == PoolStrategy.LAST_TOKEN:
    token_counts = jnp.sum(pool_mask.astype(jnp.int32), axis=-1)
    last_idx = jnp.maximum(token_counts - 1, 0)
    return hidden[jnp.arange(hidden.shape[0]), last_idx]

  if strategy == PoolStrategy.MEAN:
    mask_f = pool_mask.astype(hidden.dtype)[..., None]
    return jnp.sum(hidden * mask_f, axis=1) / jnp.maximum(
        jnp.sum(mask_f, axis=1), 1.0
    )

  raise ValueError(f"Unknown pool strategy: {strategy}")


def pool_packed_hidden_states(
    hidden: jaxtyping.Array,
    segment_ids: jaxtyping.Array,
    num_segments: int,
) -> jaxtyping.Array:
  """Extract the last-token hidden state for each packed segment.

  segment_ids layout: 0 = prefix/padding, 1..N = packed examples.

  Returns ``[B, N, D]`` where each ``[b, k, :]`` is the hidden state
  at the last token of segment ``k+1`` in batch element ``b``.
  Missing segments get the zero vector.
  """
  B, L, D = hidden.shape

  def _last_token_for_segment(b: int, seg: int) -> jaxtyping.Array:
    seg_mask = segment_ids[b] == (seg + 1)
    positions = jnp.arange(L) * seg_mask.astype(jnp.int32)
    last_pos = jnp.max(positions)
    has_segment = jnp.any(seg_mask)
    h = hidden[b, last_pos]
    return jnp.where(has_segment, h, jnp.zeros(D, dtype=hidden.dtype))

  pooled = jax.vmap(
      jax.vmap(_last_token_for_segment, in_axes=(None, 0)),
      in_axes=(0, None),
  )(jnp.arange(B), jnp.arange(num_segments))

  return pooled


def make_shared_prefix_attn_mask(
    segment_ids: jaxtyping.Array,
) -> jaxtyping.Array:
  """Star-pattern causal mask for shared-prefix packing.

  Prefix (0) attends causally to itself.  Each example segment (>0)
  attends causally to the prefix AND to itself, but NOT to other
  example segments.
  """
  seq_len = segment_ids.shape[-1]
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  same_segment = segment_ids[..., :, None] == segment_ids[..., None, :]
  is_prefix_key = segment_ids[..., None, :] == 0
  return causal[None, ...] & (same_segment | is_prefix_key)


class HeadType(enum.Enum):
  BINARY = "binary"
  MULTILABEL = "multilabel"


@dataclasses.dataclass(frozen=True, slots=True)
class ClassificationConfig:
  """Single source of truth for head shape and semantics."""

  head_type: HeadType
  num_classes: int
  class_names: tuple[str, ...] = ()
  dropout_rate: float = 0.0

  def __post_init__(self) -> None:
    if self.head_type == HeadType.BINARY and self.num_classes != 1:
      raise ValueError(
          f"BINARY head requires num_classes=1, got {self.num_classes}"
      )
    if self.class_names and len(self.class_names) != self.num_classes:
      raise ValueError(
          f"class_names length ({len(self.class_names)}) != "
          f"num_classes ({self.num_classes})"
      )

  @property
  def num_outputs(self) -> int:
    if self.head_type == HeadType.BINARY:
      return 1
    return self.num_classes


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ClassificationOutput:
  logits: jax.Array
  hidden_states: jax.Array

  def tree_flatten(self):
    return (self.logits, self.hidden_states), None

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Self:
    return cls(*children)


class ClassificationHead(nnx.Module, abc.ABC):

  @abc.abstractmethod
  def __call__(self, pooled: jaxtyping.Array) -> jaxtyping.Array: ...

  @property
  @abc.abstractmethod
  def config(self) -> ClassificationConfig: ...


class BinaryClassificationHead(ClassificationHead):
  """Single-logit sigmoid head. Labels: ``[B]`` float ``{0., 1.}``."""

  def __init__(
      self,
      embed_dim: int,
      classification_config: ClassificationConfig,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self._config = classification_config
    self.linear = nnx.Linear(embed_dim, 1, use_bias=True, rngs=rngs)
    self.dropout = (
        nnx.Dropout(rate=classification_config.dropout_rate, rngs=rngs)
        if classification_config.dropout_rate > 0.0
        else None
    )

  def __call__(self, pooled: jaxtyping.Array) -> jaxtyping.Array:
    x = self.dropout(pooled) if self.dropout is not None else pooled
    return self.linear(x).squeeze(-1)

  @property
  def config(self):
    return self._config


class MultilabelClassificationHead(ClassificationHead):
  """Per-class sigmoid head. Labels: ``[B, C]`` float multi-hot."""

  def __init__(
      self,
      embed_dim: int,
      classification_config: ClassificationConfig,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self._config = classification_config
    self.linear = nnx.Linear(
        embed_dim,
        classification_config.num_classes,
        use_bias=True,
        rngs=rngs,
    )
    self.dropout = (
        nnx.Dropout(rate=classification_config.dropout_rate, rngs=rngs)
        if classification_config.dropout_rate > 0.0
        else None
    )

  def __call__(self, pooled: jaxtyping.Array) -> jaxtyping.Array:
    x = self.dropout(pooled) if self.dropout is not None else pooled
    return self.linear(x)

  @property
  def config(self):
    return self._config


def build_head(
    classification_config: ClassificationConfig,
    embed_dim: int,
    *,
    rngs: nnx.Rngs,
) -> ClassificationHead:
  """Build the appropriate head from a ``ClassificationConfig``."""
  head_type = classification_config.head_type
  if head_type == HeadType.BINARY:
    return BinaryClassificationHead(embed_dim, classification_config, rngs=rngs)
  if head_type == HeadType.MULTILABEL:
    return MultilabelClassificationHead(
        embed_dim, classification_config, rngs=rngs
    )
  raise ValueError(f"Unknown head type: {head_type}")


class Gemma4ForClassification(gemma4_model.Gemma4):
  """Gemma4 subclass that replaces the LM head with a classification head.

  Calls ``forward_backbone`` (embed → layers → final norm) then pools
  the hidden states and projects through a task-specific head.  The
  vocabulary decode is never executed.

  Two modes:

  1. **Unpacked**: one example per row, ``segment_ids`` is ``None``.
  2. **Packed**: shared-prefix star attention, ``segment_ids`` marks
     prefix=0, examples=1..N.  Pools last-token per segment →
     ``[B*N, D]``, labels must be ``[B*N, ...]``.
  """

  def __init__(
      self,
      config: gemma4_model.ModelConfig,
      head: ClassificationHead,
      *,
      rngs: nnx.Rngs,
      pool_strategy: PoolStrategy = PoolStrategy.LAST_TOKEN,
      pad_id: int = 0,
  ) -> None:
    super().__init__(config, rngs=rngs)
    self.head = head
    self.pool_strategy = pool_strategy
    self.pad_id = pad_id

  def __call__(  # type: ignore[override]
      self,
      tokens: jaxtyping.Array,
      positions: jaxtyping.Array | None = None,
      attention_mask: jaxtyping.Array | None = None,
      *,
      pool_mask: jaxtyping.Array | None = None,
      segment_ids: jaxtyping.Array | None = None,
      num_packed_segments: int | None = None,
  ) -> ClassificationOutput:
    packed = segment_ids is not None

    if packed:
      assert num_packed_segments is not None
      if positions is None:
        raise ValueError("positions must be provided for packed mode")
      if attention_mask is None:
        attention_mask = make_shared_prefix_attn_mask(segment_ids)
    else:
      pad_mask = tokens != self.pad_id
      if positions is None:
        cumulative = jnp.cumsum(pad_mask.astype(jnp.int32), axis=-1)
        positions = cumulative - (cumulative >= 1).astype(jnp.int32)
      if attention_mask is None:
        seq_len = tokens.shape[-1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        valid = pad_mask[..., :, None] & pad_mask[..., None, :]
        attention_mask = causal[None, ...] & valid

    hidden, _ = self.forward_backbone(
        tokens,
        positions=positions,
        cache=None,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    if packed:
      pooled = pool_packed_hidden_states(
          hidden, segment_ids, num_packed_segments
      )
      B, N, D = pooled.shape
      pooled = pooled.reshape(B * N, D)
    else:
      if pool_mask is None:
        pool_mask = pad_mask
      pooled = pool_hidden_states(hidden, pool_mask, self.pool_strategy)

    logits = self.head(pooled)

    return ClassificationOutput(
        logits=logits,
        hidden_states=pooled,
    )
