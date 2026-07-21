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

"""Sharding utilities for tunix models.

This module provides helpers for placing JAX arrays with correct sharding
across devices, and for validating that arrays carry the expected shardings
before they are dispatched to JIT-compiled functions.
"""

from __future__ import annotations

from absl import logging
import jax
from jax import numpy as jnp
from jax._src.tree_util import PyTree
import jax.sharding as shd


# TODO(abheesht17): Use this function for all models and unify with the fn in
# sft/sharding_utils.py.
def shard(
    x: jax.Array,
    s: tuple[str | None, ...],
    eager: bool = False,
) -> jax.Array:
  """Shard a JAX array with the given partition spec.

  Args:
    x: The JAX array to shard.
    s: The sharding spec (axis names or None per dimension).
    eager: If ``True``, place the array onto devices immediately via
      ``jax.device_put``.  This requires a concrete physical ``Mesh`` active in
      context via ``with jax.set_mesh(mesh):``.  If only an ``AbstractMesh`` is
      available, this function raises ``RuntimeError`` rather than silently
      producing an unsharded array.

      If ``False``, the sharding is deferred via
      ``jax.lax.with_sharding_constraint``, which only takes effect inside
      a JIT-traced function body.

  Returns:
    The sharded JAX array.

  Raises:
    RuntimeError: If ``eager=True`` but no concrete physical mesh is active
      in context (only ``AbstractMesh``).
  """
  try:
    mesh = shd.get_mesh()
  except ValueError:
    mesh = shd.get_abstract_mesh()
  if mesh.empty:
    mesh = shd.get_abstract_mesh()
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return jnp.asarray(x)
  sharding = shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  if eager:
    if isinstance(mesh, shd.Mesh):
      return jax.device_put(x, sharding)
    # AbstractMesh: jax.device_put raises
    #   "is_fully_addressable is not implemented for AbstractMesh"
    # and with_sharding_constraint is a no-op outside JIT.
    # Rather than silently producing an unsharded array (which causes JIT
    # cache misses and proxy OOM), raise a loud error so the caller can fix
    # the mesh context.
    raise RuntimeError(
        f'shard(eager=True) requires a concrete jax.sharding.Mesh active in '
        f'context via `with jax.set_mesh(mesh):`, but only an AbstractMesh '
        f'was found: {mesh}. Ensure the entry point wraps execution in '
        f'`with jax.set_mesh(mesh):`.'
    )
  return jax.lax.with_sharding_constraint(x, sharding)


def validate_shardings(
    pytree: PyTree[jax.Array],
    *,
    expected_mesh: shd.Mesh | shd.AbstractMesh | None = None,
    label: str = 'pytree',
) -> None:
  """Validate that every array leaf in *pytree* has a proper NamedSharding.

  This is designed to be called **before** dispatching arrays to a
  JIT-compiled function, to detect sharding mismatches that would cause
  compilation cache misses (and therefore proxy OOM on Pathways).

  Checks performed:
    1. Every ``jax.Array`` leaf has a ``sharding`` attribute.
    2. The sharding is a ``NamedSharding`` (not ``SingleDeviceSharding`` or
       ``PmapSharding``).
    3. If *expected_mesh* is provided, the sharding's mesh matches it.

  Args:
    pytree: Arbitrary JAX pytree (e.g. a ``_SamplingState``, KV cache dict).
    expected_mesh: If provided, assert that every leaf's mesh matches.
    label: Human-readable label for error messages.

  Raises:
    ValueError: If any leaf fails validation.
  """
  leaves = jax.tree.leaves(pytree)
  errors: list[str] = []
  for i, leaf in enumerate(leaves):
    if not isinstance(leaf, jax.Array):
      continue
    leaf_sharding = getattr(leaf, 'sharding', None)
    if leaf_sharding is None:
      errors.append(
          f'  [{label} leaf {i}] shape={leaf.shape} dtype={leaf.dtype}: '
          f'no sharding attribute'
      )
      continue
    if isinstance(leaf_sharding, shd.SingleDeviceSharding):
      if expected_mesh is None or expected_mesh.size > 1:
        errors.append(
            f'  [{label} leaf {i}] shape={leaf.shape} dtype={leaf.dtype}: '
            f'expected NamedSharding, got SingleDeviceSharding '
            f'({leaf_sharding})'
        )
        continue
    elif not isinstance(leaf_sharding, shd.NamedSharding):
      errors.append(
          f'  [{label} leaf {i}] shape={leaf.shape} dtype={leaf.dtype}: '
          f'expected NamedSharding, got {type(leaf_sharding).__name__} '
          f'({leaf_sharding})'
      )
      continue
    if expected_mesh is not None and hasattr(leaf_sharding, 'mesh'):
      # Compare mesh shapes and axis names — not object identity, since
      # Mesh and AbstractMesh may differ in type but represent the same
      # logical mesh.
      leaf_mesh = leaf_sharding.mesh
      if leaf_mesh.shape != expected_mesh.shape:
        errors.append(
            f'  [{label} leaf {i}] shape={leaf.shape} '
            f'spec={getattr(leaf_sharding, "spec", None)}: '
            f'mesh shape {leaf_mesh.shape} != expected {expected_mesh.shape}'
        )
  if errors:
    raise ValueError(
        f'Sharding validation failed for {label} ({len(errors)} '
        f'leaf(s)):\n' + '\n'.join(errors)
    )


def log_sharding_summary(
    pytree: PyTree[jax.Array],
    *,
    label: str = 'pytree',
    max_leaves: int = 10,
) -> None:
  """Log a summary of the shardings in a pytree for debugging.

  Logs the shape, dtype, sharding type, and spec of the first *max_leaves*
  array leaves.
  """
  leaves = jax.tree.leaves(pytree)
  array_leaves = [
      (i, leaf) for i, leaf in enumerate(leaves)
      if isinstance(leaf, jax.Array)
  ]
  if not array_leaves:
    logging.info('[sharding_summary] %s: no array leaves', label)
    return

  logging.info(
      '[sharding_summary] %s: %d array leaves (showing first %d)',
      label, len(array_leaves), min(max_leaves, len(array_leaves)),
  )
  for i, leaf in array_leaves[:max_leaves]:
    leaf_sharding = getattr(leaf, 'sharding', None)
    if leaf_sharding is None:
      sharding_str = 'NO_SHARDING'
    elif isinstance(leaf_sharding, shd.NamedSharding):
      sharding_str = (
          f'NamedSharding(mesh={leaf_sharding.mesh.shape}, '
          f'spec={leaf_sharding.spec})'
      )
    else:
      sharding_str = f'{type(leaf_sharding).__name__}({leaf_sharding})'
    logging.info(
        '[sharding_summary]   leaf[%d] shape=%s dtype=%s sharding=%s',
        i, leaf.shape, leaf.dtype, sharding_str,
    )
