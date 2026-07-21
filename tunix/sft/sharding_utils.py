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

"""Utilities for sharding tensors."""

from __future__ import annotations

import jax
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def shard_input(
    input_data: jax.Array, data_sharding_axis: tuple[str, ...]
) -> jax.Array:
  """Shards the input data across the available devices.

  Args:
    input_data: The input data to be sharded, expected to be a TrainingInput
      dataclass.
    data_sharding_axis: The sharding axis for the input data, e.g. ("fsdp",).

  Returns:
    The sharded TrainingInput.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data

  pspec = shd.PartitionSpec(*data_sharding_axis)
  # Check if the input is already sharded with the target mesh to avoid
  # re-sharding.
  is_sharded = jax.tree.map(
      lambda x: isinstance(x, jax.Array)
      and hasattr(x, "sharding")
      and hasattr(x.sharding, "mesh")
      and x.sharding.mesh == mesh
      and hasattr(x.sharding, "spec")
      and x.sharding.spec == pspec,
      input_data,
  )
  if all(jax.tree.leaves(is_sharded)):
    return input_data

  # Skip sharding if the input contains globally distributed arrays that are
  # not fully addressable by the local process. Attempting to recreate them via
  # `jax.make_array_from_process_local_data` would fail with a ValueError.
  is_global = jax.tree.map(
      lambda x: isinstance(x, jax.Array) and not x.is_fully_addressable,
      input_data,
  )
  if any(jax.tree.leaves(is_global)):
    return input_data

  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        ),
        input_data,
    )


def get_sharding(
    x: np.ndarray | jax.Array,
    mesh: shd.Mesh,
    pspec: shd.PartitionSpec,
    *,
    strict: bool = False,
) -> shd.NamedSharding:
  """Get a sharding for a tensor given a mesh and partition spec.

  Args:
    x: The array to determine sharding for.
    mesh: The device mesh to shard across.
    pspec: The desired partition spec.
    strict: If True, raise ``ValueError`` when the array's shape is not
      evenly divisible by the mesh axis sizes (instead of silently falling
      back to full replication).  Enable this for data batches where an
      indivisible dimension indicates a configuration bug.

  Returns:
    A ``NamedSharding`` matching ``pspec`` if the array is compatible,
    or a fully-replicated sharding if the array is scalar, low-rank,
    or (when ``strict=False``) not evenly divisible.

  Raises:
    ValueError: If ``strict=True`` and the array is not evenly divisible
      along a sharded axis.
  """
  # Only shard arrays with rank > 0.
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Don't shard if rank is not sufficient.
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Check for divisibility for all sharded axes.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        axis_size = mesh.shape[name]
        if x.shape[i] % axis_size != 0:
          if strict:
            raise ValueError(
                f"Array dimension {i} (size={x.shape[i]}) is not evenly "
                f"divisible by mesh axis '{name}' (size={axis_size}). "
                f"Array shape: {x.shape}, pspec: {pspec}. "
                f"This usually means the batch size is misconfigured."
            )
          # Replicate if not evenly divisible.
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  return shd.NamedSharding(mesh, pspec)
