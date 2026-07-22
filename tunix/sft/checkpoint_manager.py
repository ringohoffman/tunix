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

"""Checkpoint manager for PEFT."""

import os
from pathlib import Path
import time
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp

_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    max_to_keep=3,
)


def is_pathways_persistence_enabled() -> bool:
  """Returns True if Pathways persistence API or proxy platform is enabled."""
  return (
      os.getenv("ENABLE_PATHWAYS_PERSISTENCE") == "1"
      or "proxy" in os.getenv("JAX_PLATFORMS", "")
  )


def gcsfuse_to_gs_path(path: str) -> str:
  """Translates a local GCSFuse mount path into a direct `gs://` URI.

  Args:
    path: Input path string (POSIX local mount or gs:// URI).

  Returns:
    The translated `gs://` URI if `path` is on a GCSFuse mount point,
    otherwise the original `path`.
  """
  if path.startswith("gs://"):
    return path

  abs_path = Path(path).resolve()
  proc_mounts = Path("/proc/mounts")
  if proc_mounts.exists():
    try:
      with proc_mounts.open("r", encoding="utf-8") as f:
        for line in f:
          parts = line.split()
          if len(parts) >= 3:
            device, mount_point_str, fstype, *_ = parts
            if "gcsfuse" in fstype or "gcsfuse" in device or "fuse" in fstype:
              mount_point = Path(mount_point_str).resolve()
              if abs_path == mount_point or mount_point in abs_path.parents:
                bucket_name = device.split(":")[-1].strip("/")
                if (
                    not bucket_name
                    or "/" in bucket_name
                    or bucket_name in ("gcsfuse", "fuse", "/dev/fuse")
                ):
                  bucket_name = mount_point.name
                rel_path = abs_path.relative_to(mount_point)
                gs_path = f"gs://{bucket_name}/{rel_path}".rstrip("/")
                logging.info(
                    "[Checkpointing] Translated GCSFuse mount path %r -> %r "
                    "to enable TPU DMA saving.",
                    path,
                    gs_path,
                )
                return gs_path
    except (OSError, UnicodeDecodeError) as e:
      logging.warning(
          "Could not read /proc/mounts for GCSFuse translation: %s", e
      )

  return path


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      if is_pathways_persistence_enabled():
        root_directory = gcsfuse_to_gs_path(root_directory)
      # When using Pathways, the checkpoint manager only supports persistence
      # APIs now.
      concurrent_gb_args = {}
      if options and hasattr(options, 'save_device_host_concurrent_gb') and options.save_device_host_concurrent_gb is not None:
        concurrent_gb_args['save_device_host_concurrent_gb'] = options.save_device_host_concurrent_gb

      if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(
                use_ocdbt=True,
                use_zarr3=False,
                **concurrent_gb_args,
            ),
            'optimizer_state': ocp.PyTreeCheckpointHandler(
                use_ocdbt=True,
                use_zarr3=False,
                **concurrent_gb_args,
            ),
        }
        if os.getenv('ENABLE_PATHWAYS_PERSISTENCE', ''):
          logging.info(
              'Using persistence API for checkpointing with Pathways.'
          )
        else:
          logging.warning(
              'Checkpointing without the persistence API, be aware of potential'
              ' OOM.'
          )
      else:
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(**concurrent_gb_args),
            'optimizer_state': ocp.PyTreeCheckpointHandler(**concurrent_gb_args),
        }
      item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_handlers=item_handlers,
          options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
      )

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if self._checkpoint_manager is None:
      return None
    return self._checkpoint_manager.latest_step()

  def save(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      optimizer: The optimizer to save the params for. If None, the optimizer
        will not be saved.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    model_cp_args = ocp.args.PyTreeSave(
        item=params, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params)
    )

    cp_save_args = {
        'model_params': model_cp_args,
    }
    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      cp_save_args['optimizer_state'] = optimizer_cp_args
    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(**cp_save_args),
        custom_metadata=custom_metadata or {},
        force=force,
    )

  def restore_optimizer_state(
      self,
      optimizer: nnx.Optimizer,
      step: int,
  ) -> bool:
    """Restores optimizer state from a checkpoint if available.

    Args:
      optimizer: The optimizer to restore state for.
      step: The checkpoint step to restore from.

    Returns:
      True if optimizer state was restored, False otherwise.
    """
    if self._checkpoint_manager is None:
      return False
    metadata = self._checkpoint_manager.metadata(step)
    if not metadata or 'optimizer_state' not in metadata.item_metadata:
      return False

    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)

    # Scalar values in optimizer states like step and count is initialized as
    # SingleDeviceSharding, which will fail if optimizer is sharded. To fix
    # it, we will replicate the scalar values.
    shardings = jax.tree_util.tree_map(lambda x: x.sharding, optimizer_state)
    try:
      named_sharding = next(
          s
          for s in jax.tree_util.tree_leaves(shardings)
          if isinstance(s, jax.sharding.NamedSharding)
      )
      fixed_sharding = nnx.get_named_sharding(
          optimizer_state, named_sharding.mesh
      )
    except StopIteration:
      fixed_sharding = shardings

    optimizer_cp_args = ocp.args.PyTreeRestore(
        item=optimizer_state,
        partial_restore=True,
        restore_args=ocp.checkpoint_utils.construct_restore_args(
            target=optimizer_state, sharding_tree=fixed_sharding
        ),
    )
    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            optimizer_state=optimizer_cp_args,
        ),
    )
    nnx.update(optimizer, ckpt.optimizer_state)
    logging.info('Restored optimizer state from step: %d', step)
    return True

  @classmethod
  def restore_optimizer_from_path(
      cls,
      optimizer: nnx.Optimizer,
      checkpoint_path: str,
  ) -> bool:
    """Restores optimizer state from a checkpoint path (step directory or root).

    Args:
      optimizer: The optimizer to restore state for.
      checkpoint_path: Path to a step directory (e.g. '.../checkpoints/20000')
        or a model_params subdirectory.

    Returns:
      True if optimizer state was restored, False otherwise.
    """
    ckpt_path = checkpoint_path.rstrip('/')
    if ckpt_path.endswith('/model_params'):
      ckpt_path = ckpt_path[: -len('/model_params')]
    step_str = os.path.basename(ckpt_path)
    if not step_str.isdigit():
      return False
    ckpt_root = os.path.dirname(ckpt_path)
    step_dir = os.path.join(ckpt_root, step_str)
    if not os.path.exists(os.path.join(step_dir, 'optimizer_state')):
      return False

    mgr = cls(ckpt_root)
    success = mgr.restore_optimizer_state(optimizer, step=int(step_str))
    mgr.close()
    return success

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> Tuple[int, dict[str, Any]]:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      optimizer: The optimizer to restore the params for. If None or if
        optimizer state is not found in the checkpoint, the optimizer will not
        be restored.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if self._checkpoint_manager is None:
      return 0, {}
    if step is None:
      step = self._checkpoint_manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0, {}

    metadata = self._checkpoint_manager.metadata(step)

    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)

    model_cp_args = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(
            target=abstract_params
        ),
    )

    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            model_params=model_cp_args,
        ),
    )
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, ckpt.model_params)

    # Restore optimizer state if available.
    if optimizer is not None:
      self.restore_optimizer_state(optimizer, step)

    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    custom_metadata = metadata.custom_metadata if metadata else {}
    return step, custom_metadata

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()
