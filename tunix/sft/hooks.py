# Copyright 2025 Google LLC
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

"""Hooks for training and data loading."""

from __future__ import annotations

from typing import Any, Generic, Protocol, TYPE_CHECKING

from typing_extensions import TypeVar

if TYPE_CHECKING:
  import jax
  from tunix.sft.peft_trainer import PeftTrainer


PeftTrainerT_contra = TypeVar(
  "PeftTrainerT_contra",
  bound="PeftTrainer[Any]",
  default="PeftTrainer[Any]",
  contravariant=True,
)


class TrainingHooks(Protocol, Generic[PeftTrainerT_contra]):
  """Hooks to be used for training."""

  def on_train_start(self, train_ctx: PeftTrainerT_contra) -> None:
    """Called at the beginning of training."""

  def on_train_end(self, train_ctx: PeftTrainerT_contra) -> None:
    """Called at the end of training."""

  def on_train_step_start(self, train_ctx: PeftTrainerT_contra) -> None:
    """Called at the beginning of a training step."""

  def on_train_step_end(
      self,
      train_ctx: PeftTrainerT_contra,
      train_step: int,
      batch: Any,
      train_loss: jax.Array,
  ) -> None:
    """Called at the end of a training step."""

  def on_train_micro_step_end(
      self,
      train_ctx: PeftTrainerT_contra,
      batch: Any,
      train_loss: jax.Array,
      grad_norm: jax.Array | None = None,
      aux: dict[str, jax.Array] | None = None,
  ) -> None:
    """Called at the end of a micro-step during gradient accumulation."""

  def on_eval_start(self, train_ctx: PeftTrainerT_contra) -> None:
    """Called at the beginning of an evaluation phase."""

  def on_eval_end(
      self,
      train_ctx: PeftTrainerT_contra,
      eval_loss: jax.Array,
  ) -> None:
    """Called at the end of an evaluation phase."""

  def on_eval_step_start(self, train_ctx: PeftTrainerT_contra) -> None:
    """Called at the beginning of an evaluation step (batch)."""

  def on_eval_step_end(
      self,
      train_ctx: PeftTrainerT_contra,
      batch: Any,
      eval_loss: jax.Array,
  ) -> None:
    """Called at the end of an evaluation step (batch)."""

  def on_eval_micro_step_end(
      self,
      train_ctx: PeftTrainerT_contra,
      batch: Any,
      eval_loss: jax.Array,
      aux: dict[str, jax.Array] | None = None,
  ) -> None:
    """Called at the end of a micro-step during evaluation."""


class DataHooks(Protocol, Generic[PeftTrainerT_contra]):
  """Hooks to wire in external data loader and processing logic."""

  def load_next_train_batch(self, train_ctx: PeftTrainerT_contra) -> Any:
    """Loads the next batch of data for training."""

  def load_next_eval_batch(self, train_ctx: PeftTrainerT_contra) -> Any:
    """Loads the next batch of data for evaluation."""
