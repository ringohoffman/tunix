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

**Why this matters on Pathways / multi-controller runtimes:**

JAX offers two APIs for constraining how arrays are partitioned across devices:

  1. ``jax.device_put(x, sharding)`` — works eagerly (outside JIT), but
     requires a concrete ``jax.sharding.Mesh`` with real devices.  It raises
     ``ValueError`` if given an ``AbstractMesh`` because ``is_fully_addressable``
     is not implemented for abstract meshes.

  2. ``jax.lax.with_sharding_constraint(x, sharding)`` — emits an XLA
     sharding annotation, but is a **silent no-op outside JIT** (the returned
     array has the same sharding as the input).

On Pathways, only an ``AbstractMesh`` may be available (via ``jax.set_mesh``),
making ``device_put`` fail.  If ``eager=True`` is requested but only an
``AbstractMesh`` is present, the old code silently fell back to
``with_sharding_constraint``, which is a no-op outside JIT → arrays end up
**unsharded** → JIT compilation cache miss → proxy recompiles → OOM.

``shard()`` now raises a loud error for this case so the caller can fix the
mesh context, rather than silently producing unsharded arrays.
"""

from __future__ import annotations

from jax._src.tree_util import PyTree

from absl import logging
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd


def get_current_mesh() -> shd.Mesh | shd.AbstractMesh | None:
  """Returns the current active JAX sharding mesh, preferring physical mesh.

  Discovery order:
    1. Physical mesh from ``pxla.thread_resources`` (set by ``with mesh:``).
    2. Abstract mesh from ``jax.set_mesh()`` / ``jax.sharding.get_abstract_mesh()``.
    3. ``None`` if no mesh is active.

  This function does NOT set or modify any mesh context — it only reads.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh is not None and not mesh.empty:
    return mesh
  abstract_mesh = shd.get_abstract_mesh()
  return None if abstract_mesh.empty else abstract_mesh


def get_physical_mesh() -> shd.Mesh | None:
  """Returns the current physical (concrete) mesh, or None.

  Unlike ``get_current_mesh``, this never returns an ``AbstractMesh``.
  Use this when you need a concrete mesh for ``jax.device_put``.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh is not None and not mesh.empty:
    return mesh
  return None


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
      ``jax.device_put``.  This requires a concrete physical ``Mesh`` in the
      current thread context (set by ``with mesh:``).  If only an
      ``AbstractMesh`` is available, this function raises ``RuntimeError``
      rather than silently producing an unsharded array.

      If ``False``, the sharding is deferred via
      ``jax.lax.with_sharding_constraint``, which only takes effect inside
      a JIT-traced function body.

  Returns:
    The sharded JAX array.

  Raises:
    RuntimeError: If ``eager=True`` but no concrete physical mesh is
      available (only ``AbstractMesh``).
  """
  mesh = get_current_mesh()
  if mesh is None or mesh.empty or jax.devices()[0].platform == 'cpu':
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
        f'shard(eager=True) requires a concrete jax.sharding.Mesh in the '
        f'thread context (set by `with mesh:`), but only an AbstractMesh '
        f'was found: {mesh}.  This typically means the outermost training '
        f'loop is not using `with mesh:` alongside `jax.set_mesh(mesh)`, '
        f'or the code is running in a thread that does not inherit the '
        f'mesh context.  On Pathways, both context managers are required.'
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
    if not isinstance(leaf_sharding, shd.NamedSharding):
      errors.append(
          f'  [{label} leaf {i}] shape={leaf.shape} dtype={leaf.dtype}: '
          f'expected NamedSharding, got {type(leaf_sharding).__name__} '
          f'({leaf_sharding})'
      )
      continue
    if expected_mesh is not None:
      # Compare mesh shapes and axis names — not object identity, since
      # Mesh and AbstractMesh may differ in type but represent the same
      # logical mesh.
      leaf_mesh = leaf_sharding.mesh
      if leaf_mesh.shape != expected_mesh.shape:
        errors.append(
            f'  [{label} leaf {i}] shape={leaf.shape} '
            f'spec={leaf_sharding.spec}: '
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
