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
from typing import Any, Callable, Generic, ParamSpec, Self, TYPE_CHECKING, TypeAlias, overload

from absl import logging
import flax
from flax import nnx
import flax.struct
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding
import jax.stages
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
from typing_extensions import Concatenate, TypeVar

if TYPE_CHECKING:
  from numpy._typing import _FloatLike_co

ModuleT = TypeVar("ModuleT", bound=nnx.Module, default=nnx.Module)
P = ParamSpec("P")
R = TypeVar("R")
G1 = TypeVar("G1")
G2 = TypeVar("G2")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions

Loss: TypeAlias = jax.Array
Aux: TypeAlias = dict[str, jax.Array] | None
GradNorm: TypeAlias = jax.Array


class Kernel(Generic[P, R]):
  """A callable that starts eager and can be JIT-compiled via ``compile()``.

  Wraps a function with graph-node arguments (model, optimizer, etc.) baked
  in via ``functools.partial``.  ``compile()`` replaces the inner callable
  with an ``nnx.jit``-compiled version that has the same call signature,
  so callers never need to branch on compilation state.

  Usage::

      k = Kernel(my_fn, model, optimizer, donate_argnames=("optimizer",))
      k(inputs)            # eager
      k.compile(...)
      k(inputs)            # JIT'd, same call
      k.reset()
      k(inputs)            # back to eager
  """

  # TODO: https://github.com/microsoft/pyright/issues/11591
  # pyright does not support TypeVarTuple inside Concatenate
  @overload
  def __init__(
      self,
      fn: Callable[P, R],
      *,
      donate_argnames: tuple[str, ...] | None = None,
  ) -> None:
    ...

  @overload
  def __init__(
      self,
      fn: Callable[Concatenate[G1, P], R],
      __graph_arg1: G1,
      /,
      *,
      donate_argnames: tuple[str, ...] | None = None,
  ) -> None:
    ...

  @overload
  def __init__(
      self,
      fn: Callable[Concatenate[G1, G2, P], R],
      __graph_arg1: G1,
      __graph_arg2: G2,
      /,
      *,
      donate_argnames: tuple[str, ...] | None = None,
  ) -> None:
    ...

  def __init__(
      self,
      fn: Callable[..., R],
      *graph_args: Any,
      donate_argnames: tuple[str, ...] | None = None,
  ) -> None:
    if isinstance(fn, functools.partial):
      target = getattr(fn, "func", fn)

      @functools.wraps(target)
      def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

      self._fn = _wrapped
    else:
      self._fn = fn
    self._graph_args = graph_args
    self._donate_argnames = donate_argnames
    self._call: Callable[..., R] = functools.partial(self._fn, *graph_args)

  def compile(
      self,
      *,
      compiler_options: jax.stages.CompilerOptions | None = None,
      cache_nnx_graph: bool = False,
  ) -> None:
    """Replace the inner callable with a JIT-compiled version."""
    jitted = nnx.jit(
        self._fn,
        donate_argnames=self._donate_argnames,
        compiler_options=compiler_options,
    )
    if cache_nnx_graph:
      self._call = functools.partial(
          nnx.cached_partial(jitted, *self._graph_args)
      )
    else:
      self._call = functools.partial(jitted, *self._graph_args)

  def reset(self) -> None:
    """Reset to eager mode."""
    self._call = functools.partial(self._fn, *self._graph_args)

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
    return self._call(*args, **kwargs)


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
    train_ctx: PeftTrainer[Any],
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


class PeftTrainer(Generic[ModuleT]):
  """PEFT trainer for LoRA. Only LoRA parameters are updated.

  Lifecycle::

      trainer = PeftTrainer(model, optimizer, config, loss_fn=my_loss_fn)
      trainer.compile()          # JIT-compile train/eval kernels
      trainer.train(train_ds)    # runs the loop (calls compile() if needed)

      # Or call steps manually:
      trainer.compile()
      loss, aux, grad_norm = trainer.train_step(batch)
      loss, aux = trainer.eval_step(batch)

      # Eager (no JIT) also works — just don't call compile():
      loss, aux, grad_norm = trainer.train_step(batch)

  Subclasses override ``compile()`` and ``train_step()`` to set up custom
  JIT kernels (e.g. multi-kernel GRPO).

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use.
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
      model: ModuleT,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      loss_fn: Callable[..., Any] | None = None,
      eval_loss_fn: Callable[..., Any] | None = None,
      metrics_logger: MetricsLogger | None = None,
      perf_tracer: perf_trace.Tracer | None = None,
      perf_tracer_v2: perf_tracer_lib.Tracer | None = None,
      wrt: nnx.filterlib.Filter | None = None,
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

    gradient_transform = optimizer
    if training_config.gradient_accumulation_steps is not None:
      gradient_transform = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      ).gradient_transformation()

    if wrt is None:
      wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param

    self.optimizer = nnx.Optimizer(self.model, gradient_transform, wrt=wrt)

    self.loss_fn: Callable[..., Any] | None = loss_fn
    self.eval_loss_fn: Callable[..., Any] | None = (
        eval_loss_fn if eval_loss_fn is not None else loss_fn
    )
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

    self._compiled: bool = False
    self._train_kernel = Kernel(
        self._train_step_impl,
        self.model,
        self.optimizer,
        donate_argnames=("optimizer",),
    )
    self._eval_kernel = Kernel(self._eval_step_impl, self.model)

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

    self.data_hooks: hooks.DataHooks | None = None
    self._jit_cache: set[int] = set()
    self._mini_batch_size: int | None = None

  def with_training_hooks(
      self, training_hooks: hooks.TrainingHooks | Iterable[hooks.TrainingHooks]
  ) -> Self:
    self.training_hooks = (
        list(training_hooks)
        if isinstance(training_hooks, Iterable)
        else [training_hooks]
    )
    return self

  def with_data_hooks(self, data_hooks: hooks.DataHooks) -> None:
    self.data_hooks = data_hooks

  def clear_jit_cache(self) -> None:
    """Clears compiled state, forcing recompilation on next ``compile()``.

    Automatically resets all ``Kernel`` attributes on this trainer.
    """
    for attr in vars(self).values():
      if isinstance(attr, Kernel):
        attr.reset()
    self._compiled = False

  def _train_step_impl(
      self, model: ModuleT, optimizer: nnx.Optimizer[Any], inputs: Any
  ) -> tuple[Loss, Aux, GradNorm]:
    """Raw train step body — forward, backward, optimizer update.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      A tuple of (loss, aux_or_None, grad_norm).
    """
    if (loss_fn := self.loss_fn) is None:
      raise ValueError(
          "loss_fn must be provided before training. Pass loss_fn to"
          " PeftTrainer.__init__."
      )

    def _wrapped_loss_fn(m: ModuleT, inp: Any) -> tuple[Loss, Aux]:
      out = loss_fn(m, inp)
      if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
      return out, None

    grad_fn = nnx.value_and_grad(
        _wrapped_loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=True,
    )
    (loss, aux), grads = grad_fn(model, inputs)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return loss, aux, grad_norm

  def _eval_step_impl(self, model: ModuleT, inputs: Any) -> tuple[Loss, Aux]:
    if (eval_loss_fn := self.eval_loss_fn) is None:
      raise ValueError(
          "eval_loss_fn must be provided before evaluating. Pass eval_loss_fn"
          " to PeftTrainer.__init__."
      )
    out = eval_loss_fn(model, inputs)
    if isinstance(out, tuple) and len(out) == 2:
      return out[0], out[1]
    return out, None

  def _shard_optimizer(self, mesh: jax.sharding.Mesh | None = None) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    if mesh is None:
      mesh = jax.sharding.get_mesh()
    if mesh is None or mesh.empty:
      return
    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)

    total_bytes = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves(optimizer_state)
        if hasattr(leaf, "nbytes")
    )
    global_total_gb = total_bytes / (1024**3)
    fsdp_size = mesh.shape.get("fsdp", 1)
    logging.info(
        "_shard_optimizer: global optimizer state = %.2f GB across mesh=%s "
        "using jax.device_put",
        global_total_gb,
        dict(mesh.shape),
    )

    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    def _to_sharding(spec: Any) -> jax.sharding.Sharding:
      if isinstance(spec, jax.sharding.Sharding):
        return spec
      if isinstance(spec, jax.sharding.PartitionSpec):
        return jax.sharding.NamedSharding(mesh, spec)
      if isinstance(spec, tuple):
        return jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(*spec)
        )
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    optimizer_shardings = jax.tree.map(_to_sharding, optimizer_pspecs)
    optimizer_sharded_state = jax.device_put(
        optimizer_state, optimizer_shardings
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

    per_chip_gb = (
        global_total_gb / fsdp_size if fsdp_size > 0 else global_total_gb
    )
    logging.info(
        "_shard_optimizer: SHARDED successfully -> %.2f GB per device "
        "(%.2f GB global total divided across FSDP=%d chips)",
        per_chip_gb,
        global_total_gb,
        fsdp_size,
    )

  def compile(self, *, cache_nnx_graph: bool = False) -> None:
    """JIT-compile all ``Kernel`` instances on this trainer. Idempotent."""
    if self._compiled:
      return

    self._shard_optimizer(jax.sharding.get_mesh())

    opts: dict[str, Any] = dict(
        compiler_options=self.config.compiler_options,
        cache_nnx_graph=cache_nnx_graph,
    )
    for attr in vars(self).values():
      if isinstance(attr, Kernel):
        attr.compile(**opts)

    self._compiled = True

  def train_step(self, inputs: Any) -> tuple[Loss, Aux, GradNorm]:
    """Execute a single training step (eager or compiled)."""
    return self._train_kernel(inputs)

  def eval_step(self, inputs: Any) -> tuple[Loss, Aux]:
    """Execute a single eval step (eager or compiled)."""
    return self._eval_kernel(inputs)

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
      train_ds: Iterable[Any] | None = None,
      eval_ds: Iterable[Any] | None = None,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    """Training loop.

    Calls ``compile()`` automatically if not already compiled.
    """
    mesh = jax.sharding.get_mesh()
    logging.log_first_n(
        logging.INFO,
        f"Training with mesh: {mesh}",
        1,
    )

    self.compile(cache_nnx_graph=cache_nnx_graph)

    if eval_ds and self.config.eval_at_start:
      self._run_eval(eval_ds)

    for hook in self.training_hooks:
      hook.on_train_start(self)

    train_iterator = iter(train_ds) if train_ds is not None else iter(())
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
        global_step: int | None = metadata.get("global_step")

        if global_step is not None:
          # Offset by 1 since global_step is incremented for checkpointing.
          global_step -= 1
          if global_step > 0:
            if self._mini_batch_size is None:
              self._mini_batch_size = max(1, self._train_steps // global_step)
            mini_batch: int | None = self._train_steps % self._mini_batch_size
          else:
            mini_batch = self._train_steps
        else:
          mini_batch = None
          global_step = None
        micro_batch = self._iter_steps % self.config.get_with_default(
            "gradient_accumulation_steps", 1
        )
        tags: dict[str, Any] = {
            perf_constants.STEP: global_step,
            perf_constants.ROLE: metadata.get("role"),
            perf_constants.MICRO_BATCH: micro_batch,
            perf_constants.MINI_BATCH: mini_batch,
        }

        with self._perf_tracer.span(
            "peft_train_step",
            mesh.devices,
        ) as span, self._perf_tracer_v2.span(
            perf_constants.PEFT_TRAIN,
            mesh.devices,
            tags=tags,
        ) as span_v2:
          train_loss, aux, grad_norm = self.train_step(train_example)
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
            self._run_eval(eval_ds)

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

  def _save_last_checkpoint(self) -> None:
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

  def close(self) -> None:
    """Closes the trainer and its associated resources.

    This includes saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    for hook in self.training_hooks:
      hook.on_train_step_end(self, self._train_steps, None, jnp.asarray(0))
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    if self.metrics_logger is not None:
      self.metrics_logger.close()

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
  ) -> None:
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    with self._switch_mode(sft_metrics_logger.Mode.EVAL):
      for hook in self.training_hooks:
        hook.on_eval_start(self)

      if self.eval_loss_fn is None:
        logging.info(
            "No eval_loss_fn configured on trainer; skipping loss evaluation on"
            " eval_ds."
        )
        for hook in self.training_hooks:
          hook.on_eval_end(self, jnp.asarray(0.0))
        return

      eval_iterator = iter(eval_ds)
      eval_loss: float | jax.Array = 0
      eval_steps = 0
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
        loss, aux = self.eval_step(eval_example)
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
        hook.on_eval_end(self, jnp.asarray(eval_loss))
