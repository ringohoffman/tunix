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

"""Peft trainer unittest."""

import contextlib
import functools
import os
import tempfile
from typing import Any, Tuple
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import peft_trainer
from tunix.sft import profiler
from tunix.tests import test_common as tc
from tunix.utils import compat

TEST_LEARNING_RATE = 1e-3

# CPU environment setup to simulate multi device env.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


def create_sharded_model(model_ctor, rngs, mesh):
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    model = model_ctor(config=tc.ModelConfig(), rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with compat.set_mesh(mesh):
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


def dummy_loss_fn(model: nnx.Module, x: peft_trainer.TrainingInput):
  positions = jnp.arange(x.input_tokens.shape[1])
  attention_mask = jnp.ones_like(x.input_tokens)
  logits, _ = model(x.input_tokens, positions, None, attention_mask)
  logits = logits[:, :-1, :]
  target_tokens = x.input_tokens[:, 1:]
  target_mask = x.input_mask[:, 1:]
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor


def dummy_datasets(batch_size: int, repeat: int = 1):
  # (num_batch, batch_size, seq_len)
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      peft_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


global_counter = 0


class PeftTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.temp_path = self.create_tempdir().full_path
    except Exception:
      self.temp_path = tempfile.TemporaryDirectory().name

    # CPU env setup to simulate multi device env. Won't affect TPU env. But
    # need to be careful not to use self.num_cpus in TPU env.
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)

    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)
    total_devices = jax.device_count()
    self.mesh = shd.Mesh(
        devices=np.array(jax.devices()).reshape(2, total_devices // 2),
        axis_names=('fsdp', 'tp'),
    )

    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)

  def test_compile_once(self):
    class CountCompiledTimesTrainer(peft_trainer.PeftTrainer):

      def _train_step_impl(self, model, optimizer, inputs):
        global global_counter
        global_counter += 1
        return super()._train_step_impl(model, optimizer, inputs)

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs), mesh=self.mesh
    )
    trainer = CountCompiledTimesTrainer(
        model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )
    global global_counter
    global_counter = 0  # make mypy happy
    train_ds = dummy_datasets(batch_size=4, repeat=5)
    with self.mesh:
      trainer.train(train_ds, self.eval_ds)
    self.assertLessEqual(global_counter, 2)

  @parameterized.named_parameters(
      ('cache_nnx_graph', True),
      ('no_cache_nnx_graph', False),
  )
  def test_basic_training(self, cache_nnx_graph: bool):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    optimizer = optax.inject_hyperparams(optax.sgd)(
        learning_rate=optax.constant_schedule(TEST_LEARNING_RATE)
    )
    trainer = peft_trainer.PeftTrainer(
        model, optimizer, config, loss_fn=dummy_loss_fn
    )

    trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=cache_nnx_graph)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertGreater(
        trainer.metrics_logger.get_metric('', 'perplexity', 'train'), 0
    )
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )
    self.assertGreater(
        trainer.metrics_logger.get_metric('', 'perplexity', 'eval'), 0
    )
    self.assertGreater(trainer._train_steps, 0)

    self.assertGreater(
        len(
            trainer.metrics_logger.get_metric_history('', 'perplexity', 'train')
        ),
        0,
    )

    trainer.train(self.train_ds)  # No eval dataset.

  @parameterized.named_parameters(
      ('lora_disabled_distributed', False, True),
      ('lora_disabled_single_device', False, False),
      ('lora_enabled_distributed', True, True),
      ('lora_enabled_single_device', True, False),
  )
  def test_checkpoint_save_and_restore(
      self, enable_lora: bool, distributed: bool
  ):
    def create_model_and_optimizer():
      rngs = nnx.Rngs(0)
      if distributed:
        model, _ = create_sharded_model(tc.ToyTransformer, rngs, self.mesh)
      else:
        model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
      if enable_lora:
        model = tc.get_lora_model(model)

      optimizer = optax.inject_hyperparams(optax.adamw)(
          learning_rate=optax.constant_schedule(TEST_LEARNING_RATE)
      )
      return model, optimizer

    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=100,
        checkpoint_root_directory=f'{self.temp_path}/{self.id()}/checkpoints',
    )

    model, optimizer = create_model_and_optimizer()
    original_model_state = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam if enable_lora else nnx.Param)
    )

    ctx = (
        compat.set_mesh(self.mesh) if distributed else contextlib.nullcontext()
    )

    with ctx:
      trainer = peft_trainer.PeftTrainer(
          model, optimizer, config, loss_fn=dummy_loss_fn
      )
      trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=True)
      trained_model_state = nnx.state(
          model, nnx.LoRAParam if enable_lora else nnx.Param
      )
      trained_opt_state = nnx.state(trainer.optimizer, nnx.optimizer.OptState)

      jax.tree.map_with_path(
          tc.assert_not_equal, original_model_state, trained_model_state
      )

      # Resume from checkpoint with a new model and optimizer, and check that
      # the model and optimizer states are the same as the trained ones.
      new_model, new_optimizer = create_model_and_optimizer()

      resumed_trainer = peft_trainer.PeftTrainer(
          new_model, new_optimizer, config, loss_fn=dummy_loss_fn
      )
      resumed_model_state = nnx.state(
          resumed_trainer.model, nnx.LoRAParam if enable_lora else nnx.Param
      )
      resumed_opt_state = nnx.state(
          resumed_trainer.optimizer, nnx.optimizer.OptState
      )

      jax.tree.map_with_path(
          tc.assert_equal, trained_model_state, resumed_model_state
      )
      jax.tree.map_with_path(
          tc.assert_equal, trained_opt_state, resumed_opt_state
      )

      resumed_trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=True)

      resumed_opt_state = nnx.state(
          resumed_trainer.optimizer, nnx.optimizer.OptState
      )

      jax.tree.map(
          lambda x, y: self.assertTrue(
              x.sharding.is_equivalent_to(y.sharding, ndim=x.ndim)
          ),
          trained_opt_state,
          resumed_opt_state,
      )

  def test_basic_training_with_hooks(self):
    train_ds = dummy_datasets(batch_size=4, repeat=2)
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100, eval_at_start=False
    )
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    class TrackingHooks(hooks.TrainingHooks):

      def __init__(self):
        self.train_start_count = 0
        self.train_step_start_count = 0
        self.train_step_end_count = 0
        self.eval_step_start_count = 0
        self.eval_step_end_count = 0
        self.train_end_count = 0

      def on_train_start(self, trainer):
        self.train_start_count += 1

      def on_train_step_start(self, trainer):
        self.train_step_start_count += 1

      def on_train_step_end(self, trainer, step, train_example, train_loss):
        self.train_step_end_count += 1

      def on_eval_step_start(self, trainer):
        self.eval_step_start_count += 1

      def on_eval_step_end(self, trainer, eval_example, loss):
        self.eval_step_end_count += 1

      def on_train_end(self, trainer):
        self.train_end_count += 1

    tracking_hooks = TrackingHooks()
    trainer = peft_trainer.PeftTrainer(
        model,
        optax.sgd(1e-3),
        config,
        loss_fn=dummy_loss_fn,
    )
    trainer.with_training_hooks(tracking_hooks)
    trainer.train(train_ds, self.eval_ds)

    self.assertEqual(tracking_hooks.train_start_count, 1)
    self.assertEqual(tracking_hooks.train_step_start_count, 4)
    self.assertEqual(tracking_hooks.train_step_end_count, 5)
    self.assertEqual(tracking_hooks.eval_step_start_count, 4)
    self.assertEqual(tracking_hooks.eval_step_end_count, 4)
    self.assertEqual(tracking_hooks.train_end_count, 1)

  def test_reusing_trainer(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )
    trainer.train(self.train_ds, None)

    previous_call = trainer._train_kernel._call

    trainer.clear_jit_cache()
    trainer.train(self.train_ds, None)
    curr_call = trainer._train_kernel._call
    self.assertIsNot(previous_call, curr_call)

  @mock.patch.object(profiler, 'Profiler')
  def test_basic_training_with_profiler(self, mock_profiler_init):
    mock_profiler_instance = mock.MagicMock()
    mock_profiler_init.return_value = mock_profiler_instance
    mock_profiler_instance.should_activate.side_effect = (
        lambda step: step == profiler_options.skip_first_n_steps
    )
    mock_profiler_instance.should_deactivate.side_effect = (
        lambda step: step
        == (
            profiler_options.skip_first_n_steps
            + profiler_options.profiler_steps
        )
    )
    profiler_options = profiler.ProfilerOptions(
        '/tmp/profiler', skip_first_n_steps=2, profiler_steps=3
    )
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100, profiler_options=profiler_options
    )
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )

    train_ds = dummy_datasets(batch_size=4, repeat=4)
    trainer.train(train_ds)  # No eval dataset.

    mock_profiler_init.assert_called_once_with(
        initial_step=0,
        max_step=config.max_steps,
        profiler_options=profiler_options,
    )
    expected_calls = (
        # steps 0 through 8.
        [mock.call.maybe_activate(step) for step in range(8)]
        # steps 1 through 9 as step number is incremented during each step.
        + [mock.call.maybe_deactivate(step) for step in range(1, 9)]
    )
    mock_profiler_instance.assert_has_calls(
        expected_calls,
        any_order=True,
    )

  def test_dist_training(self):
    rngs = nnx.Rngs(0)
    model, _ = create_sharded_model(tc.ToyTransformer, rngs, self.mesh)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )

    with self.mesh:
      trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    self.assertEqual(
        variables.layers[0].w1.kernel.value.sharding.spec,
        shd.PartitionSpec('fsdp', 'tp'),
    )
    self.assertEqual(
        variables.layers[0].w2.kernel.value.sharding.spec,
        shd.PartitionSpec('tp', 'fsdp'),
    )

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    # compare with unsharded model
    rngs = nnx.Rngs(0)
    unsharded_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    trainer = peft_trainer.PeftTrainer(
        unsharded_model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )
    trainer.train(self.train_ds, self.eval_ds)
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    self.assertIsInstance(
        unsharded_variables.layers[0].w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    jax.tree.map_with_path(tc.assert_close, variables, unsharded_variables)

  def test_custom_loss_fn(self):
    def custom_loss_fn(
        model: nnx.Module,
        x: peft_trainer.TrainingInput,
    ) -> jax.Array:
      positions = jnp.arange(x.input_tokens.shape[1])
      attention_mask = jnp.ones_like(x.input_tokens)
      logits, _ = model(x.input_tokens, positions, None, attention_mask)
      logits = logits[:, :-1, :]
      target_tokens = x.input_tokens[:, 1:]
      target_mask = x.input_mask[:, 1:]
      one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
      one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
      return optax.softmax_cross_entropy(logits, one_hot).mean()

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config, loss_fn=custom_loss_fn
    )
    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  @parameterized.named_parameters(
      ('scalar', TEST_LEARNING_RATE),
      ('constant_schedule', optax.constant_schedule(TEST_LEARNING_RATE)),
  )
  def test_lora_training(self, learning_rate_scheduler):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    )

    original_params = jax.tree.map(
        jnp.copy, nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )
    original_lora_params = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam)
    )
    optimizer = optax.inject_hyperparams(optax.sgd)(
        learning_rate=learning_rate_scheduler
    )
    trainer = peft_trainer.PeftTrainer(
        model, optimizer, config, loss_fn=dummy_loss_fn
    )

    trainer.train(self.train_ds, self.eval_ds)
    params = nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    lora_params = nnx.state(model, nnx.LoRAParam)

    jax.tree.map_with_path(tc.assert_equal, original_params, params)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_lora_params, lora_params
    )
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )

  @parameterized.named_parameters(
      ('scalar', TEST_LEARNING_RATE),
      ('constant_schedule', optax.constant_schedule(TEST_LEARNING_RATE)),
  )
  def test_gradient_accumulation(self, learning_rate_schedule):
    def train(
        train_ds,
        gradient_accumulation_steps: int | None,
        learning_rate_schedule: int | optax.Schedule,
    ):
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=2,
          max_steps=100,
          gradient_accumulation_steps=gradient_accumulation_steps,
      )
      rngs = nnx.Rngs(0)
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

      optimizer = optax.inject_hyperparams(optax.sgd)(
          learning_rate=learning_rate_schedule
      )
      trainer = peft_trainer.PeftTrainer(
          model, optimizer, config, loss_fn=dummy_loss_fn
      )
      trainer.train(train_ds, self.eval_ds)
      self.assertEqual(
          trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
          TEST_LEARNING_RATE,
      )
      return nnx.state(model, nnx.Param), trainer

    train_ds = dummy_datasets(batch_size=4, repeat=4)
    params, trainer = train(
        train_ds,
        gradient_accumulation_steps=None,
        learning_rate_schedule=learning_rate_schedule,
    )
    params_with_grad_accumulation, grad_accu_trainer = train(
        dummy_datasets(batch_size=2, repeat=4),
        gradient_accumulation_steps=2,
        learning_rate_schedule=learning_rate_schedule,
    )
    jax.tree.map_with_path(
        functools.partial(tc.assert_close, atol=1e-7, rtol=1e-7),
        params,
        params_with_grad_accumulation,
    )
    self.assertEqual(trainer.train_steps, grad_accu_trainer.train_steps)
    self.assertEqual(trainer.iter_steps * 2, grad_accu_trainer.iter_steps)
    np.testing.assert_allclose(
        trainer.metrics_logger.get_metric('', 'loss', 'train'),
        grad_accu_trainer.metrics_logger.get_metric('', 'loss', 'train'),
        atol=1e-5,
        rtol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='without_grad_accu',
          grad_accu=1,
          resume_step=0,
          expected_save_steps=[1, 2, 3, 4],
      ),
      dict(
          testcase_name='grad_accu',
          grad_accu=2,
          resume_step=0,
          expected_save_steps=[1, 2],
      ),
      dict(
          testcase_name='with_resume',
          grad_accu=1,
          resume_step=1,
          expected_save_steps=[2, 3, 4],
      ),
      dict(
          testcase_name='with_resume_and_grad_accu',
          grad_accu=2,
          resume_step=1,
          expected_save_steps=[2],
      ),
  )
  @mock.patch.object(checkpoint_manager, 'CheckpointManager')
  def test_checkpointing(
      self,
      mock_checkpoint_manager_init,
      grad_accu,
      resume_step,
      expected_save_steps,
  ):
    mock_checkpoint_manager = mock.MagicMock()
    mock_checkpoint_manager_init.return_value = mock_checkpoint_manager
    mock_checkpoint_manager.maybe_restore.return_value = (resume_step, {})
    mock_checkpoint_manager.save.return_value = True
    mock_checkpoint_manager.latest_step.return_value = (
        expected_save_steps[-1] - 1
    )  # force save at close
    checkpoint_options = ocp.CheckpointManagerOptions()
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=100,
        gradient_accumulation_steps=grad_accu,
        checkpoint_root_directory='/tmp/checkpoint',
        checkpointing_options=checkpoint_options,
    )
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    )
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config, loss_fn=dummy_loss_fn
    )

    train_ds = eval_ds = dummy_datasets(batch_size=2, repeat=1)  # 4 batches
    trainer.train(train_ds, eval_ds)

    mock_checkpoint_manager_init.assert_called_once_with(
        root_directory='/tmp/checkpoint', options=checkpoint_options
    )
    # Assert that the checkpoint manager is called with the correct arguments
    # and does not have any unexpected calls.
    mock_checkpoint_manager.assert_has_calls(
        [
            mock.call.maybe_restore(
                mock.ANY, mock.ANY, restore_only_lora_params=True
            ),
            *[
                mock.call.save(
                    i,
                    mock.ANY,
                    mock.ANY,
                    save_only_lora_params=True,
                    custom_metadata={},
                )
                for i in expected_save_steps
            ],
            mock.call.latest_step(),
            mock.call.save(
                expected_save_steps[-1],
                mock.ANY,
                mock.ANY,
                save_only_lora_params=True,
                force=True,
            ),
            mock.call.close(),
        ],
        any_order=False,
    )

  def test_loss_fn_with_aux(self):
    def custom_loss_fn(
        model: nnx.Module,
        x: peft_trainer.TrainingInput,
    ) -> Tuple[jax.Array, Any]:
      del model, x
      return jnp.array(1.0), {'foo': 1, 'bar': 2}

    train_invoke = {'foo': 0, 'bar': 0}
    eval_invoke = {'foo': 1, 'bar': 1}

    class CustomTrainer(peft_trainer.PeftTrainer):

      def _post_process_train_step(self, aux):
        train_invoke['foo'] += aux['foo']
        train_invoke['bar'] += aux['bar']

      def _post_process_eval_step(self, aux):
        eval_invoke['foo'] *= aux['foo']
        eval_invoke['bar'] *= aux['bar']

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    trainer = CustomTrainer(
        model, optax.sgd(1e-3), config, loss_fn=custom_loss_fn
    )

    trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(train_invoke, {'foo': 2, 'bar': 4})
    self.assertEqual(eval_invoke, {'foo': 1, 'bar': 16})

  def test_injected_params(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    learning_rate_scheduler = optax.constant_schedule(TEST_LEARNING_RATE)
    optimizer = optax.inject_hyperparams(optax.sgd)(
        momentum=0.001,
        learning_rate=learning_rate_scheduler,
    )

    trainer = peft_trainer.PeftTrainer(
        model, optimizer, config, loss_fn=dummy_loss_fn
    )
    trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )


class KernelTest(parameterized.TestCase):

  def test_kernel_preserves_function_name_for_plain_callable(self):
    def my_custom_train_step(x: jax.Array) -> jax.Array:
      return x * 2

    k = peft_trainer.Kernel(my_custom_train_step)
    self.assertEqual(k._fn.__name__, 'my_custom_train_step')
    self.assertEqual(k._fn.__qualname__, my_custom_train_step.__qualname__)
    self.assertEqual(k(jnp.array(3)), 6)

  def test_kernel_preserves_function_name_for_functools_partial(self):
    def policy_gradient_train_step(bias: int, x: jax.Array) -> jax.Array:
      return x + bias

    p = functools.partial(policy_gradient_train_step, 10)
    k = peft_trainer.Kernel(p)
    self.assertEqual(k._fn.__name__, 'policy_gradient_train_step')
    self.assertEqual(
        k._fn.__qualname__, policy_gradient_train_step.__qualname__
    )
    self.assertEqual(k(jnp.array(5)), 15)

  def test_kernel_compile_generates_named_xla_module(self):
    def rollout_generation_step(scale: float, x: jax.Array) -> jax.Array:
      return x * scale

    p = functools.partial(rollout_generation_step, 3.0)
    k = peft_trainer.Kernel(p)
    k.compile()

    lowered = nnx.jit(k._fn).lower(jnp.ones((4,), dtype=jnp.float32))
    hlo_text = lowered.as_text()
    self.assertIn('rollout_generation_step', hlo_text)


if __name__ == '__main__':
  absltest.main()
