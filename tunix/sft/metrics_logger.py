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

"""Metric logger with a unified, protocol-based backend system."""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Any, Callable, Protocol, TYPE_CHECKING, TypeAlias, TypedDict, overload

from absl import logging
import jax
from metrax import logging as metrax_logging
import numpy as np
from tunix.sft import hooks
from tunix.sft import progress_bar
from tunix.utils import env_utils
from typing_extensions import override

if TYPE_CHECKING:
  from jax.typing import ArrayLike
  from numpy._typing import _ArrayLikeFloat_co, _FloatLike_co

  from tunix.sft import peft_trainer

  _ReductionFunction: TypeAlias = Callable[
      [_ArrayLikeFloat_co], np.floating[Any]
  ]

LoggingBackend = metrax_logging.LoggingBackend
TensorboardBackend = metrax_logging.TensorboardBackend
WandbBackend = metrax_logging.WandbBackend
CluBackend = getattr(metrax_logging, "CluBackend", None)

# User backends MUST be factories (callables) to keep Options pure and copyable.
BackendFactory = Callable[[], LoggingBackend]


class CustomBackendFactories(TypedDict):
  custom_backend: list[BackendFactory]


class LoggingBackendKwargs(TypedDict, total=False):
  clu: dict[str, Any]
  wandb: dict[str, Any]
  tensorboard: dict[str, Any]


@dataclasses.dataclass
class MetricsLoggerOptions:
  """Metrics Logger options."""

  log_dir: str
  project_name: str = "tunix"
  run_name: str = ""
  flush_every_n_steps: int = 100
  # Keyword arguments for backend initialization. The key is the backend name
  # (e.g., 'wandb', 'clu', 'tensorboard' or 'custom_backend' which uses custom
  # LoggingBackend factories) and the value is a dictionary of
  # keyword arguments to be passed to the backend's constructor.
  # For example:
  # backend_kwargs={
  #   'wandb': {
  #     'resume': 'must',
  #     'id': '12345',
  #     'project': 'my-project',
  #     'name': 'my-run',
  #   },
  #   'tensorboard': {
  #      'log_dir': '/path/to/log',
  #      'flush_every_n_steps': 100,
  #   }
  # }
  backend_kwargs: CustomBackendFactories | LoggingBackendKwargs = (
      dataclasses.field(default_factory=LoggingBackendKwargs)
  )

  def create_backends(self) -> list[LoggingBackend]:
    """Factory method to create a fresh set of live backends."""
    # Only create live backends on the main process.
    if jax.process_index() != 0:
      return []

    # Case 1: Override. Use user-provided factories.
    if (
        "custom_backend" in self.backend_kwargs
        and self.backend_kwargs["custom_backend"]
    ):
      # Sort factories so WandbBackend (and subclasses) are instantiated before
      # TensorboardBackend. wandb.init() patches the TB SummaryWriter class at
      # init time, so wandb.init() must run before any TB writers are created.
      def _is_wandb_factory(factory: BackendFactory) -> bool:
        return isinstance(factory, type) and issubclass(
            factory, metrax_logging.WandbBackend
        )

      sorted_factories = sorted(
          self.backend_kwargs["custom_backend"],
          key=lambda f: not _is_wandb_factory(f),
      )
      return [factory() for factory in sorted_factories]

    # Case 2: Defaults.
    active_backends: list[LoggingBackend] = []

    if env_utils.is_internal_env():
      if CluBackend is None:
        raise ImportError(
            "Internal environment detected, but CluBackend not available."
        )
      clu_kwargs = self.backend_kwargs.get("clu", {})
      active_backends.append(CluBackend(log_dir=self.log_dir, **clu_kwargs))
    else:
      try:
        wandb_kwargs = self.backend_kwargs.get("wandb", {})
        # WandbBackend must be created before TensorboardBackend so that
        # wandb.init() (and the sync_tensorboard TB patcher) runs before any
        # TB writers are created.
        active_backends.append(
            WandbBackend(
                project=self.project_name,
                name=self.run_name,
                **wandb_kwargs,
            )
        )
      except ImportError:
        logging.info("WandbBackend skipped: 'wandb' library not installed.")

      tb_kwargs = self.backend_kwargs.get("tensorboard", {})
      active_backends.append(
          TensorboardBackend(
              log_dir=self.log_dir,
              flush_every_n_steps=self.flush_every_n_steps,
              **tb_kwargs,
          )
      )
    return active_backends


class Mode(str, enum.Enum):
  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


def _calculate_geometric_mean(x: np.ndarray) -> np.ndarray:
  """Calculates geometric mean of a batch of values."""
  return np.exp(np.mean(np.log(x)))


@overload
def _to_np_array(v: ArrayLike) -> np.ndarray:
  ...


@overload
def _to_np_array(v: list[ArrayLike]) -> list[np.ndarray]:
  ...


def _to_np_array(
    v: ArrayLike | list[ArrayLike],
) -> np.ndarray | list[np.ndarray]:
  if isinstance(v, list):
    return [_to_np_array(x) for x in v]
  return np.asarray(v, dtype=np.float32)


class MetricsLogger:
  """Simple Metrics logger.

  Log metrics to multiple backends. If no backends are specified, it will log to
  the default backends.
  """

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
  ) -> None:
    self._metrics: dict[str, dict[str, dict[str, list[_FloatLike_co]]]] = (
        collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
        )
    )
    self._backends = (
        metrics_logger_options.create_backends()
        if metrics_logger_options
        else []
    )
    if metrics_logger_options and jax.process_index() == 0:
      has_tensorboard_backend = any(
          isinstance(babckend, TensorboardBackend)
          for babckend in self._backends
      )
      for backend in self._backends:
        if (
            isinstance(backend, WandbBackend)
            and backend._sync_tensorboard
            and has_tensorboard_backend
        ):
          # if sync_tensorboard=True, wandb automatically copies its scalars
          # from Tensorboard and wandb.log must not be used
          continue
        jax.monitoring.register_scalar_listener(backend.log_scalar)

  def log(
      self,
      metrics_prefix: str,
      metric_name: str,
      scalar_value: _FloatLike_co,
      mode: Mode | str,
      step: int,
      *,
      formatted_name: str | None = None,
  ):
    """Logs the scalar metric value to local history and via jax.monitoring.

    Args:
      metrics_prefix: Prefix for metric grouping in internal storage.
      metric_name: Name of the metric (e.g., "loss", "perplexity").
      scalar_value: The scalar value to log.
      mode: Training mode (train/eval).
      step: The training step number.
      formatted_name: Optional pre-formatted metric name for
        ``jax.monitoring.record_scalar``. If ``None``, defaults to
        ``"{metrics_prefix}/{mode}/{metric_name}"``. This allows callers to
        override the name format (e.g., ``"loss/train/batch"``) while keeping
        internal storage keyed by ``(prefix, mode, metric_name)`` for progress
        bar compatibility.
    """
    self._metrics[metrics_prefix][mode][metric_name].append(scalar_value)

    if formatted_name is None:
      formatted_name = f"{metrics_prefix}/{mode}/{metric_name}"

    jax.monitoring.record_scalar(formatted_name, float(scalar_value), step=step)

  def metric_exists(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ) -> bool:
    """Checks if the metric exists for the given metric name and mode."""
    if metrics_prefix not in self._metrics:
      return False
    if mode not in self._metrics[metrics_prefix]:
      return False
    return metric_name in self._metrics[metrics_prefix][mode]

  def get_metric(self, metrics_prefix, metric_name: str, mode: Mode | str):
    """Returns the mean metric value for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f"Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
      )
    values = np.stack(self._metrics[metrics_prefix][mode][metric_name])
    if metric_name == "perplexity":
      return _calculate_geometric_mean(values)
    return np.mean(values)

  def get_metric_history(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ):
    """Returns all past metric values for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f" Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
          f" Available metrics for mode '{mode}':"
          f" {list(self._metrics[metrics_prefix][mode].keys())}"
      )
    return np.stack(self._metrics[metrics_prefix][mode][metric_name])

  def close(self):
    """Closes all registered logging backends."""
    for backend in self._backends:
      try:
        jax.monitoring.unregister_scalar_listener(backend.log_scalar)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      backend.close()
    try:
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricsBuffer:
  """Metrics collected for a specific step.

  Attributes:
    step: The training step number.
    metrics: Dictionary for storing all metrics. The key is the metric name, and
      the value is a list of metric values.
  """

  step: int
  metrics: dict[str, list[ArrayLike]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


@dataclasses.dataclass(frozen=True)
class MetricNameFormatter:
  """Controls how metric names are formatted for logging.

  The default format produces TensorBoard-friendly names where the
  metric name comes first for natural grouping::

      loss/train/batch
      perplexity/valid/epoch

  For the legacy tunix format (mode/metric), use::

      MetricNameFormatter(template="{mode}/{metric}")

  Available placeholders:

  - ``{metric}`` — metric name (e.g., "loss", "perplexity")
  - ``{mode}`` — training mode, after ``mode_names`` mapping
  - ``{level}`` — aggregation level ("batch" or "epoch")
  - ``{prefix}`` — ``metrics_prefix`` from ``TrainingConfig``
  """

  template: str = "{metric}/{mode}/{level}"
  mode_names: dict[str, str] = dataclasses.field(
      default_factory=lambda: {"train": "train", "eval": "valid"},
  )

  def format(
      self,
      metric: str,
      mode: Mode | str,
      level: str,
      prefix: str = "",
  ) -> str:
    mode_str = self.mode_names.get(str(mode), str(mode))
    name = self.template.format(
        metric=metric,
        mode=mode_str,
        level=level,
        prefix=prefix,
    )
    # Strip trailing slashes when level is empty (e.g. "loss/train/" -> "loss/train").
    return name.rstrip("/")


class StepMetricsFn(Protocol):
  """Protocol for mutating aggregated metrics before logging.

  This function receives the aggregated metrics dict and may add
  derived metrics (e.g. perplexity from loss) by mutating it in place.
  """

  def __call__(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      step: int,
      mode: Mode,
      metrics: dict[str, _FloatLike_co],
  ) -> None:
    ...


class MetricLoggingHook(hooks.TrainingHooks):
  """General-purpose metric logging hook with buffering and formatting.

  Handles:

  - Double-buffered metric writing (overlap I/O with next step)
  - JAX array → numpy conversion via ``_to_np_array``
  - Configurable metric naming via :class:`MetricNameFormatter`
  - Both per-step (train) and aggregate (eval) logging
  - Progress bar integration

  **What** gets logged is controlled by the ``step_metrics_fn`` callback
  provided during initialization. The default implementation logs all
  metrics from the buffer.

  The hook logs at two levels:

  - **step** — logged at each training step.
  - **aggregate** — logged at the end of each eval phase: the mean training
    loss since the last eval, and the mean eval loss across all eval steps.

  Args:
    formatter: Controls metric name formatting.  Defaults to
      ``"{metric}/{mode}/{level}"`` which produces TensorBoard-friendly names
      like ``"loss/train/step"``.
    show_progress_bar: If ``True``, show a tqdm progress bar during training.
      Defaults to ``True``.
    tqdm_train_metrics: Metric names to display in the progress bar.
  """

  def __init__(
      self,
      metric_name_formatter: MetricNameFormatter | None = None,
      *,
      show_progress_bar: bool = True,
      tqdm_train_metrics: list[str] | None = None,
      step_metrics_fn: StepMetricsFn | None = None,
      metric_reducers: dict[str, _ReductionFunction] | None = None,
  ) -> None:
    self._metric_name_formatter = metric_name_formatter or MetricNameFormatter()
    self._show_progress_bar = show_progress_bar
    self._tqdm_train_metrics = tqdm_train_metrics or ["loss"]
    self._step_metrics_fn = step_metrics_fn
    self._metric_reducers = metric_reducers or {}
    self._metric_reducers.setdefault("loss", np.mean)
    self._metric_reducers.setdefault("grad_norm", np.mean)

    self._buffered_train_metrics: MetricsBuffer | None = None
    self._prev_buffered_train_metrics: MetricsBuffer | None = None
    self._buffered_eval_metrics: MetricsBuffer | None = None
    self._pbar: progress_bar.ProgressBar | None = None
    self._mode: Mode = Mode.TRAIN

    # Epoch-level accumulators (cleared at each eval boundary).
    self._epoch_train_buffer: MetricsBuffer | None = None
    self._epoch_eval_buffer: MetricsBuffer | None = None

  def _buffer_metrics(
      self,
      metrics_buffer: MetricsBuffer | None,
      loss: ArrayLike,
      step: int,
      aux: dict[str, ArrayLike] | None = None,
  ) -> MetricsBuffer:
    if metrics_buffer is None:
      metrics_buffer = MetricsBuffer(step=step)
    else:
      assert metrics_buffer.step == step

    metrics_buffer.metrics["loss"].append(loss)

    if aux is not None:
      for k, v in aux.items():
        metrics_buffer.metrics[k].append(v)

    return metrics_buffer

  def _format_name(
      self,
      metric: str,
      level: str,
      prefix: str = "",
  ) -> str:
    """Format a metric name using the configured formatter."""
    return self._metric_name_formatter.format(
        metric=metric,
        mode=self._mode,
        level=level,
        prefix=prefix,
    )

  def _log_metrics(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      metrics: dict[str, _FloatLike_co],
      step: int,
      level: str,
  ) -> None:
    """Log a flat dict of metrics using the configured formatter."""
    if (metrics_logger := train_ctx.metrics_logger) is None:
      return

    prefix = train_ctx.metrics_prefix
    for metric_name, value in metrics.items():
      formatted = self._format_name(metric_name, level=level, prefix=prefix)
      metrics_logger.log(
          prefix,
          metric_name,
          value,
          self._mode,
          step,
          formatted_name=formatted,
      )

  def _write_metrics(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      metrics_buffer: MetricsBuffer,
      level: str = "",
  ) -> None:
    aggregated_metrics: dict[str, _FloatLike_co] = {
        k: self._metric_reducers.get(k, np.mean)(_to_np_array(v))
        for k, v in metrics_buffer.metrics.items()
    }
    if self._step_metrics_fn is not None:
      self._step_metrics_fn(
          train_ctx,
          step=metrics_buffer.step,
          mode=self._mode,
          metrics=aggregated_metrics,
      )
    self._log_metrics(train_ctx, aggregated_metrics, metrics_buffer.step, level)

  @staticmethod
  def _accumulate_epoch(
      epoch_buffer: MetricsBuffer | None,
      step_buffer: MetricsBuffer,
  ) -> MetricsBuffer:
    """Merge a per-step buffer into an epoch-level accumulator."""
    if epoch_buffer is None:
      epoch_buffer = MetricsBuffer(step=step_buffer.step)
    else:
      epoch_buffer.step = step_buffer.step
    for k, v in step_buffer.metrics.items():
      epoch_buffer.metrics[k].extend(v)
    return epoch_buffer

  def _may_update_pbar(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      metrics: list[str],
  ) -> None:
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      self._pbar.update()

  def _write_train_metrics(
      self,
      train_ctx: peft_trainer.PeftTrainer,
  ) -> None:
    """Writes previous buffered train metrics.

    Uses double-buffering to overlap I/O with the next training step:
    the first step is skipped so its metrics can be written while the
    second step is computing.
    """
    if self._prev_buffered_train_metrics is None:
      # skip the first step so we can overlap I/O with next step.
      self._prev_buffered_train_metrics = self._buffered_train_metrics
      self._buffered_train_metrics = None
      return

    # increment the step by one for logging purpose, because train_step is not
    # incremented until the next model update.
    self._prev_buffered_train_metrics.step += 1
    self._write_metrics(
        train_ctx, self._prev_buffered_train_metrics, level="step"
    )
    self._may_update_pbar(train_ctx, self._tqdm_train_metrics)

    # Accumulate for epoch-level train metrics.
    # Accumulate for aggregate train metrics.
    self._epoch_train_buffer = self._accumulate_epoch(
        self._epoch_train_buffer,
        self._prev_buffered_train_metrics,
    )

    self._prev_buffered_train_metrics = self._buffered_train_metrics
    self._buffered_train_metrics = None

  @override
  def on_train_start(
      self,
      train_ctx: peft_trainer.PeftTrainer,
  ) -> None:
    if (
        self._show_progress_bar
        and self._pbar is None
        and train_ctx.config.max_steps is not None
        and train_ctx.metrics_logger is not None
    ):
      self._pbar = progress_bar.ProgressBar(
          metrics_prefix=train_ctx.metrics_prefix,
          metrics_logger=train_ctx.metrics_logger,
          initial_steps=train_ctx._train_steps,
          max_steps=train_ctx.config.max_steps,
          description=train_ctx.config.pbar_description,
      )

  @override
  def on_train_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
  ) -> None:
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None

  @override
  def on_train_micro_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      batch: Any,
      train_loss: ArrayLike,
      grad_norm: ArrayLike | None = None,
      aux: dict[str, ArrayLike] | None = None,
  ) -> None:
    self._mode = Mode.TRAIN

    aux = {
        **(aux or {}),
        **({"grad_norm": grad_norm} if grad_norm is not None else {}),
    }
    self._buffered_train_metrics = self._buffer_metrics(
        self._buffered_train_metrics,
        loss=train_loss,
        step=train_ctx._train_steps,
        aux=aux,
    )

  @override
  def on_train_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      train_step: int,
      batch: Any,
      train_loss: ArrayLike,
  ) -> None:
    self._mode = Mode.TRAIN
    self._write_train_metrics(train_ctx)

  @override
  def on_eval_micro_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      batch: Any,
      eval_loss: ArrayLike,
      aux: dict[str, ArrayLike] | None = None,
  ) -> None:
    self._mode = Mode.EVAL
    self._buffered_eval_metrics = self._buffer_metrics(
        self._buffered_eval_metrics,
        loss=eval_loss,
        step=train_ctx._train_steps,
        aux=aux,
    )

  @override
  def on_eval_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      batch: Any,
      eval_loss: ArrayLike,
  ) -> None:
    self._mode = Mode.EVAL
    if self._buffered_eval_metrics is not None:
      # Accumulate for epoch-level eval metrics.
      self._epoch_eval_buffer = self._accumulate_epoch(
          self._epoch_eval_buffer,
          self._buffered_eval_metrics,
      )
      self._buffered_eval_metrics = None

  @override
  def on_eval_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      eval_loss: ArrayLike,
  ) -> None:
    self._mode = Mode.EVAL

    # Write epoch-level eval metrics. If eval batch metrics were logged
    # per-step, the epoch buffer holds the accumulated data. Otherwise,
    # fall back to the per-step buffer that has been growing naturally.
    eval_epoch_buffer = self._epoch_eval_buffer or self._buffered_eval_metrics
    if eval_epoch_buffer is not None:
      self._write_metrics(train_ctx, eval_epoch_buffer)
    self._epoch_eval_buffer = None
    self._buffered_eval_metrics = None

    # Write aggregate train metrics accumulated since last eval.
    # Override step to align with the eval boundary — the double-buffering
    # in _write_train_metrics causes the buffer's step to lag by 1.
    if self._epoch_train_buffer is not None:
      self._mode = Mode.TRAIN
      self._epoch_train_buffer.step = train_ctx._train_steps
      self._write_metrics(train_ctx, self._epoch_train_buffer)
      self._epoch_train_buffer = None
      self._mode = Mode.EVAL
