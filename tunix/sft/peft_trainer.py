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

"""PEFT trainer."""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import dataclasses
import functools
import time
from typing import Any, Callable, Concatenate, ParamSpec, Self, TypeAlias, TYPE_CHECKING

from absl import logging
import flax
from flax import nnx
import jax
import jax.stages
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import profiler
from tunix.sft import sharding_utils
from tunix.sft import utils

if TYPE_CHECKING:
  from numpy._typing import _FloatLike_co

_ModelInput: TypeAlias = dict[str, ArrayLike]
P = ParamSpec("P")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None
  eval_at_start: bool = True

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  # Configs for performance metrics.
  perf_metrics_options: perf_metrics.PerfMetricsOptions | None = None

  data_sharding_axis: tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metrics_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  # Sequence packing configuration.
  max_seq_token_per_tpu: int | None = None

  # Optional JAX compiler options (env overrides)
  compiler_options: jax.stages.CompilerOptions | None = None

  def get_with_default(self, key: str, default: Any) -> Any:
    val = getattr(self, key)
    if val is None:
      return default
    return val


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to the model.
  input_tokens: jax.Array | np.ndarray

  # A mask that determines which input tokens are valid.
  input_mask: jax.Array | np.ndarray

  # Optional images for vision models.
  images: jax.Array | np.ndarray | None = None


def _sft_step_metrics_fn(
    train_ctx: PeftTrainer,
    step: int,
    mode: sft_metrics_logger.Mode,
    metrics: dict[str, _FloatLike_co],
) -> None:
  """Computes legacy SFT metrics including perplexity and learning rate."""
  metrics["perplexity"] = np.exp(metrics["loss"])

  if mode == sft_metrics_logger.Mode.TRAIN:
    logging.info(
        "Train step %d training loss: %f - training perplexity: %f",
        step,
        metrics["loss"],
        metrics["perplexity"],
    )

  learning_rate = train_ctx._try_get_learning_rate()
  if learning_rate is not None:
    metrics["learning_rate"] = learning_rate


class PeftTrainer:
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    loss_fn: The loss function to use.
    eval_loss_fn: The loss function to use for evaluation.
    checkpoint_manager: The checkpoint manager to use.
    metrics_logger: The metrics logger to use.
    metrics_prefix: The prefix for metric names for logging.
    is_managed_externally: Whether the trainer is managed externally.
    training_hooks: The training hooks to use.
    data_hooks: The data hooks to use.
  """

  supports_sequence_packing = False

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      metrics_logger: MetricsLogger | None = None,
      perf_tracer: perf_trace.Tracer | None = None,
      perf_tracer_v2: perf_tracer_lib.Tracer | None = None,
  ) -> None:
    # TODO(noghabi): Implement sequence packing for SFT and remove this check.
    if (
        training_config.max_seq_token_per_tpu is not None
        and not self.supports_sequence_packing
    ):
      raise ValueError(
          "Sequence packing is not supported in SFT PeftTrainer yet."
      )

    self.model = model
    self.config = training_config
    self._lora_enabled = utils.is_lora_enabled(self.model)
    if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)

    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    self.metrics_logger = metrics_logger
    self.metrics_prefix = self.config.metrics_prefix
    if self.metrics_logger is None:
      self.metrics_logger = MetricsLogger(
          self.config.metrics_logging_options,
      )
    self.is_managed_externally = False
    self._perf_tracer = (
        perf_tracer if perf_tracer is not None else perf_trace.NoopTracer()
    )
    self._perf_tracer_v2 = (
        perf_tracer_v2
        if perf_tracer_v2 is not None
        else perf_tracer_lib.NoopTracer()
    )

    self._train_steps = 0  # represent # of times model has been updated
    self._iter_steps = 0  # represent # of times trainer has looped
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: sft_metrics_logger.Mode = sft_metrics_logger.Mode.TRAIN
    self._has_aux = False
    self._pbar = None

    self._train_steps, self._restored_custom_metadata = (
        self.checkpoint_manager.maybe_restore(
            self.model,
            self.optimizer,
            restore_only_lora_params=self._lora_enabled,
        )
    )
    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    max_step = None
    if self.config.max_steps is not None:
      max_step = self.config.max_steps * self.config.get_with_default(
          "gradient_accumulation_steps", 1
      )
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=max_step,
        profiler_options=self.config.profiler_options,
    )

    metric_name_formatter = sft_metrics_logger.MetricNameFormatter(
        template="{mode}/{metric}",
        mode_names={"train": "train", "eval": "eval"},
    )
    metric_logging_hook = sft_metrics_logger.MetricLoggingHook(
        metric_name_formatter=metric_name_formatter,
        tqdm_train_metrics=["loss", "perplexity", "learning_rate"],
        step_metrics_fn=_sft_step_metrics_fn,
    )
    self.with_training_hooks(metric_logging_hook)

    self.data_hooks = None
    self._jit_cache = set()
    self._mini_batch_size = None

  def with_training_hooks(
      self, training_hooks: hooks.TrainingHooks | Iterable[hooks.TrainingHooks]
  ) -> Self:
    self.training_hooks = (
        list(training_hooks)
        if isinstance(training_hooks, Iterable)
        else [training_hooks]
    )
    return self

  def with_data_hooks(self, data_hooks: hooks.DataHooks):
    self.data_hooks = data_hooks

  def clear_jit_cache(self):
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overriding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P], ArrayLike | tuple[ArrayLike, Any]
      ],
      has_aux: bool = False,
  ):
    self.clear_jit_cache()
    self.loss_fn = loss_fn
    self.eval_loss_fn = loss_fn
    self._has_aux = has_aux
    return self

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> tuple[ArrayLike, Any | None, ArrayLike]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      A tuple containing the loss, auxiliary data (or None if has_aux is False),
      and the gradient norm.
    """
    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, inputs)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux, grad_norm
    else:
      return out, None, grad_norm

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | tuple[ArrayLike, Any]:
    out = self.eval_loss_fn(model, inputs)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def create_train_step_fn(
      self,
  ) -> Callable[..., tuple[ArrayLike, Any | None, ArrayLike]]:
    """Creates the train step function."""
    return self._train_step

  def create_eval_step_fn(self) -> Callable[..., ArrayLike]:
    """Creates the eval step function."""
    return self._eval_step

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh.empty:
      return
    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)

    total_bytes = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves(optimizer_state)
        if hasattr(leaf, 'nbytes')
    )
    global_total_gb = total_bytes / (1024**3)
    fsdp_size = mesh.shape.get('fsdp', 1)
    logging.info(
        '_shard_optimizer: global optimizer state = %.2f GB across mesh=%s '
        'using jax.device_put',
        global_total_gb,
        dict(mesh.shape),
    )

    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    def _to_sharding(spec: Any) -> shd.Sharding:
      if isinstance(spec, shd.Sharding):
        return spec
      if isinstance(spec, shd.PartitionSpec):
        return shd.NamedSharding(mesh, spec)
      if isinstance(spec, tuple):
        return shd.NamedSharding(mesh, shd.PartitionSpec(*spec))
      return shd.NamedSharding(mesh, shd.PartitionSpec())

    optimizer_shardings = jax.tree.map(_to_sharding, optimizer_pspecs)
    optimizer_sharded_state = jax.device_put(
        optimizer_state, optimizer_shardings
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

    per_chip_gb = global_total_gb / fsdp_size if fsdp_size > 0 else global_total_gb
    logging.info(
        '_shard_optimizer: SHARDED successfully -> %.2f GB per device '
        '(%.2f GB global total divided across FSDP=%d chips)',
        per_chip_gb,
        global_total_gb,
        fsdp_size,
    )

  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ) -> tuple[
      Callable[..., tuple[jax.Array, Any, jax.Array]],
      Callable[..., tuple[jax.Array, Any]],
  ]:
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.
      cache_nnx_graph: If True, the nnx graph will be cached.

    Returns:
      A tuple of train and eval step functions.
    """
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return train_step, eval_step

    if self._jitted_train_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      self._jitted_train_step_fn = nnx.jit(
          train_step,
          donate_argnames=("optimizer",),
          compiler_options=self.config.compiler_options,
      )
      self._jitted_eval_step_fn = nnx.jit(
          eval_step,
          compiler_options=self.config.compiler_options,
      )

      def maybe_cache_and_partial(f, *args):
        if cache_nnx_graph:
          # wrap with partial so we can access jitted_fn in a consistent way.
          return functools.partial(nnx.cached_partial(f, *args))
        else:
          return functools.partial(f, *args)

      self._jitted_train_step_fn = maybe_cache_and_partial(
          self._jitted_train_step_fn, self.model, self.optimizer
      )
      self._jitted_eval_step_fn = maybe_cache_and_partial(
          self._jitted_eval_step_fn, self.model
      )
    return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _prepare_inputs(self, input_data: Any) -> Any:
    """Override this function for additional input preparation."""
    return input_data

  def _post_process_train_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from train step."""
    pass

  def _post_process_eval_step(self, aux: Any) -> None:
    """Override this function for post processing aux data from eval step."""
    pass

  def _try_get_learning_rate(self) -> float | None:
    """Returns the learning rate from the optimizer state if available."""
    try:
      return self.optimizer.opt_state.hyperparams["learning_rate"].value
    except AttributeError:
      for chainpart in self.optimizer.opt_state:
        if isinstance(chainpart, optax.EmptyState):
          break
        if hasattr(chainpart, "hyperparams"):
          return chainpart.hyperparams["learning_rate"].value
      return None

  @contextlib.contextmanager
  def _switch_mode(self, mode: sft_metrics_logger.Mode):
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  def train(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    """Training loop."""
    logging.log_first_n(
        logging.INFO,
        f"Training with mesh: {pxla.thread_resources.env.physical_mesh}",
        1,
    )
    train_step, eval_step = self.jit_train_and_eval_step(
        skip_jit, cache_nnx_graph
    )
    if not skip_jit:
      cache_size = train_step.func.jitted_fn._cache_size()  # pytype: disable=attribute-error
      logging.log_if(
          logging.INFO,
          f"Compiled train_step cache size: {cache_size}",
          condition=cache_size not in self._jit_cache,
      )
      self._jit_cache.add(cache_size)

    if eval_ds and self.config.eval_at_start:
      self._run_eval(eval_ds, eval_step)

    for hook in self.training_hooks:
      hook.on_train_start(self)

    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    while True:
      self._prof.maybe_activate(self._iter_steps)
      with jax.profiler.StepTraceAnnotation("train", step_num=self._iter_steps):
        train_example = None
        if self.data_hooks:
          train_example = self.data_hooks.load_next_train_batch(self)
        else:
          try:
            train_example = next(train_iterator)
            if not self.is_managed_externally:
              # TODO(mridulsahu): Add support to restore the iterator state
              # instead of skipping the already trained examples.
              if index < self._iter_steps:
                # Skip the examples that are already trained.
                index += 1
                continue
            index += 1
          except StopIteration:
            pass

        if train_example is None:
          break

        # Stop training if max_steps is reached.
        if (
            not self.is_managed_externally
            and self.config.max_steps is not None
            and self._train_steps >= self.config.max_steps
        ):
          break

        train_example = self._prepare_inputs(train_example)
        train_example = sharding_utils.shard_input(
            train_example, self.config.data_sharding_axis
        )

        self._throttler.wait_for_next()
        for hook in self.training_hooks:
          hook.on_train_step_start(self)

        # Collect tags for the span
        metadata = self.custom_checkpoint_metadata()
        global_step = metadata.get("global_step")

        if global_step is not None:
          # Offset by 1 since global_step is incremented for checkpointing.
          global_step -= 1
          if global_step > 0:
            if self._mini_batch_size is None:
              self._mini_batch_size = max(1, self._train_steps // global_step)
            mini_batch = self._train_steps % self._mini_batch_size
          else:
            mini_batch = self._train_steps
        else:
          mini_batch = None
          global_step = None
        micro_batch = self._iter_steps % self.config.get_with_default(
            "gradient_accumulation_steps", 1
        )
        tags = {
            perf_constants.STEP: global_step,
            perf_constants.ROLE: metadata.get("role"),
            perf_constants.MICRO_BATCH: micro_batch,
            perf_constants.MINI_BATCH: mini_batch,
        }

        with self._perf_tracer.span(
            "peft_train_step",
            pxla.thread_resources.env.physical_mesh.devices,
        ) as span, self._perf_tracer_v2.span(
            perf_constants.PEFT_TRAIN,
            pxla.thread_resources.env.physical_mesh.devices,
            tags=tags,
        ) as span_v2:
          train_loss, aux, grad_norm = train_step(train_example)
          span.device_end([train_loss])
          span_v2.async_end([train_loss])

        self._throttler.add_computation(train_loss)
        for hook in self.training_hooks:
          hook.on_train_micro_step_end(
              self,
              train_example,
              train_loss,
              grad_norm,
              aux,
          )
        # NB: put this after _buffer_metrics is important
        self._post_process_train_step(aux)
        self._iter_steps += 1

        if (
            self._iter_steps
            % self.config.get_with_default("gradient_accumulation_steps", 1)
            == 0
        ):
          self._train_steps += 1
          for hook in self.training_hooks:
            hook.on_train_step_end(
                self,
                self._train_steps,
                train_example,
                train_loss,
            )

          # Checkpoint frequency is configured by checkpointing_options.
          self.checkpoint_manager.save(
              self._train_steps,
              self.model,
              self.optimizer,
              save_only_lora_params=self._lora_enabled,
              custom_metadata=self.custom_checkpoint_metadata(),
          )

          if (
              eval_ds
              and self._train_steps % self.config.eval_every_n_steps == 0
          ):
            self._run_eval(eval_ds, eval_step)

      self._prof.maybe_deactivate(self._iter_steps)

    self._throttler.wait_for_all()
    logging.info(
        "Train loop finished in: %.4f seconds",
        time.perf_counter() - last_step_completion_time,
    )
    for hook in self.training_hooks:
      hook.on_train_end(self)
    if not self.is_managed_externally:
      self.close()

  def _save_last_checkpoint(self):
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          self.optimizer,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    """Returns the number of train steps taken."""
    return self._train_steps

  @property
  def iter_steps(self) -> int:
    """Returns the number of iterator steps taken."""
    return self._iter_steps

  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    """Override this function to return the custom metadata for the checkpoint manager."""
    return {}

  def close(self):
    """Closes the trainer and its associated resources.

    This includes saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    for hook in self.training_hooks:
      hook.on_train_step_end(self, self._train_steps, None, 0.0)
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    if self.metrics_logger is not None:
      self.metrics_logger.close()

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step_fn: Callable[..., Any],
  ) -> None:
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    eval_iterator = iter(eval_ds)
    with self._switch_mode(sft_metrics_logger.Mode.EVAL):
      eval_loss, eval_steps = 0, 0
      for hook in self.training_hooks:
        hook.on_eval_start(self)
      while True:
        if self.data_hooks:
          eval_example = self.data_hooks.load_next_eval_batch(self)
        else:
          try:
            eval_example = next(eval_iterator)
          except StopIteration:
            eval_example = None
        if eval_example is None:
          break
        eval_example = self._prepare_inputs(eval_example)
        eval_example = sharding_utils.shard_input(
            eval_example, self.config.data_sharding_axis
        )
        for hook in self.training_hooks:
          hook.on_eval_step_start(self)
        loss, aux = eval_step_fn(eval_example)
        loss = jax.lax.stop_gradient(loss)
        for hook in self.training_hooks:
          hook.on_eval_micro_step_end(self, eval_example, loss, aux)
        self._post_process_eval_step(aux)
        eval_loss += loss
        eval_steps += 1
        for hook in self.training_hooks:
          hook.on_eval_step_end(self, eval_example, loss)

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      logging.info(
          "Train step %d eval completed.",
          self._train_steps,
      )
      for hook in self.training_hooks:
        hook.on_eval_end(self, eval_loss)


def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> ArrayLike:
  """Default loss function for PEFT training."""
  # Weird kwargs workaround because not all models support `images` right now.
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(input_tokens, positions, None, attention_mask, **kwargs)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
