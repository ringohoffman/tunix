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

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
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

  def _get_row_segment_positions(
      seg_row: jaxtyping.Array,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    def _get_seg_pos(
        k: jaxtyping.Array,
    ) -> tuple[jaxtyping.Array, jaxtyping.Array]:
      mask = seg_row == (k + 1)
      pos = jnp.max(jnp.where(mask, jnp.arange(L), 0))
      has_seg = jnp.any(mask)
      return pos, has_seg

    pos, has_seg = jax.vmap(_get_seg_pos)(jnp.arange(num_segments))
    return pos, has_seg

  last_pos, has_segment = jax.vmap(_get_row_segment_positions)(segment_ids)
  last_pos = jax.lax.stop_gradient(last_pos)
  has_segment = jax.lax.stop_gradient(has_segment)

  def _gather_row(
      h_row: jaxtyping.Array,
      pos_row: jaxtyping.Array,
      mask_row: jaxtyping.Array,
  ) -> jaxtyping.Array:
    gathered = h_row[pos_row]
    return jnp.where(mask_row[:, None], gathered, jnp.zeros_like(gathered))

  return jax.vmap(_gather_row)(hidden, last_pos, has_segment)


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


@dataclasses.dataclass(slots=True, kw_only=True)
class ClassificationModelConfig(gemma4_model.ModelConfig):
  """Configuration for Gemma4ForClassification model."""

  head_type: HeadType = HeadType.MULTILABEL
  num_classes: int = 1
  class_names: tuple[str, ...] = ()
  dropout_rate: float = 0.0
  max_examples_per_packed_sequence: int | None = None
  pool_strategy: PoolStrategy = PoolStrategy.LAST_TOKEN

  def __post_init__(self) -> None:
    super().__post_init__()
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
  def __call__(self, pooled: jaxtyping.Array) -> jaxtyping.Array:
    ...

  @property
  @abc.abstractmethod
  def config(self) -> ClassificationModelConfig:
    ...


class BinaryClassificationHead(ClassificationHead):
  """Single-logit sigmoid head. Labels: ``[B]`` float ``{0., 1.}``."""

  def __init__(
      self,
      config: ClassificationModelConfig,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self._config = config
    self.linear = nnx.Linear(config.embed_dim, 1, use_bias=True, rngs=rngs)
    self.dropout = (
        nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        if config.dropout_rate > 0.0
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
      config: ClassificationModelConfig,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self._config = config
    self.linear = nnx.Linear(
        config.embed_dim,
        config.num_classes,
        use_bias=True,
        rngs=rngs,
    )
    self.dropout = (
        nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        if config.dropout_rate > 0.0
        else None
    )

  def __call__(self, pooled: jaxtyping.Array) -> jaxtyping.Array:
    x = self.dropout(pooled) if self.dropout is not None else pooled
    return self.linear(x)

  @property
  def config(self):
    return self._config


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
      config: ClassificationModelConfig,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self.config: ClassificationModelConfig
    super().__init__(config, rngs=rngs)
    if config.head_type == HeadType.BINARY:
      self.head = BinaryClassificationHead(config, rngs=rngs)
    elif config.head_type == HeadType.MULTILABEL:
      self.head = MultilabelClassificationHead(config, rngs=rngs)
    else:
      raise ValueError(f"Unknown head type: {config.head_type}")

  def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
      self,
      tokens: jaxtyping.Array,
      positions: jaxtyping.Array | None = None,
      cache: gemma4_model.Cache | gemma4_model.StackedCache | None = None,
      attention_mask: jaxtyping.Array | None = None,
      decode_only_last_token: bool = False,
      segment_ids: jaxtyping.Array | None = None,
  ) -> ClassificationOutput:
    mask: jax.Array | None = None
    if segment_ids is not None:
      if positions is None:
        raise ValueError("positions must be provided for packed mode")
      if attention_mask is None:
        attention_mask = make_shared_prefix_attn_mask(segment_ids)
    else:
      if attention_mask is None:
        seq_len = tokens.shape[-1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attention_mask = causal[None, ...]

      # Derive valid token mask directly from attention_mask diagonal [B, T]
      mask = jnp.diagonal(attention_mask, axis1=-2, axis2=-1)
      assert mask is not None
      if mask.ndim > 2:
        mask = jnp.squeeze(mask, axis=-2)

      if positions is None:
        cumulative = jnp.cumsum(mask.astype(jnp.int32), axis=-1)
        positions = cumulative - (cumulative >= 1).astype(jnp.int32)

    hidden, _ = self.forward_backbone(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    if segment_ids is not None:
      if self.config.max_examples_per_packed_sequence is None:
        raise ValueError(
            "max_examples_per_packed_sequence must be specified when"
            " initializing Gemma4ForClassification for packed mode."
        )
      pooled = pool_packed_hidden_states(
          hidden, segment_ids, self.config.max_examples_per_packed_sequence
      )
      B, N, D = pooled.shape
      pooled = pooled.reshape(B * N, D)
    else:
      assert mask is not None
      pooled = pool_hidden_states(hidden, mask, self.config.pool_strategy)

    logits = self.head(pooled)

    return ClassificationOutput(
        logits=logits,
        hidden_states=pooled,
    )
