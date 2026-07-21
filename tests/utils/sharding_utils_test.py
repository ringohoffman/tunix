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

"""Tests for tunix.utils.sharding_utils."""

from absl.testing import absltest
import jax
from jax._src.mesh import use_abstract_mesh
from jax import numpy as jnp
import jax.sharding as shd
import numpy as np

from tunix.utils import sharding_utils


class GetCurrentMeshTest(absltest.TestCase):

  def test_none_when_no_mesh(self):
    """When no mesh is active, get_current_mesh returns None."""
    mesh = sharding_utils.get_current_mesh()
    self.assertIsNone(mesh)

  def test_under_set_mesh(self):
    """When jax.set_mesh is active, get_current_mesh returns active mesh."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with jax.set_mesh(mesh):
      current = sharding_utils.get_current_mesh()
      self.assertIsNotNone(current)
      self.assertEqual(current.axis_names, mesh.axis_names)

  def test_prefers_physical_mesh(self):
    """When both physical and abstract mesh are active, prefers physical."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with mesh:
      with use_abstract_mesh(mesh.abstract_mesh):
        current = sharding_utils.get_current_mesh()
        self.assertIsInstance(current, shd.Mesh)


class GetPhysicalMeshTest(absltest.TestCase):

  def test_none_when_no_mesh(self):
    self.assertIsNone(sharding_utils.get_physical_mesh())

  def test_returns_mesh_when_physical(self):
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with mesh:
      self.assertIsNotNone(sharding_utils.get_physical_mesh())
      self.assertIsInstance(sharding_utils.get_physical_mesh(), shd.Mesh)

  def test_none_when_only_abstract(self):
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with use_abstract_mesh(mesh.abstract_mesh):
      self.assertIsNone(sharding_utils.get_physical_mesh())


class ShardTest(absltest.TestCase):

  def test_fallback_returns_asarray(self):
    """When CPU or no mesh active, shard() returns plain jnp.asarray(x)."""
    x = [1.0, 2.0, 3.0]
    sharded = sharding_utils.shard(x, ('data',), eager=True)
    np.testing.assert_allclose(sharded, np.array(x))

  def test_shard_under_set_mesh(self):
    """Under active set_mesh, shard() returns properly sharded array."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with jax.set_mesh(mesh):
      x = jnp.arange(8)
      sharded_eager = sharding_utils.shard(x, ('data',), eager=True)
      if jax.devices()[0].platform != 'cpu':
        self.assertIsInstance(sharded_eager.sharding, shd.NamedSharding)
      else:
        np.testing.assert_allclose(sharded_eager, x)

      sharded_lazy = sharding_utils.shard(x, ('data',), eager=False)
      if jax.devices()[0].platform != 'cpu':
        self.assertIsInstance(sharded_lazy.sharding, shd.NamedSharding)
      else:
        np.testing.assert_allclose(sharded_lazy, x)

  def test_shard_eager_under_abstract_mesh_only_raises(self):
    """Under AbstractMesh only (no physical mesh), eager=True raises RuntimeError on non-CPU devices.

    This is the key safety check: silently producing an unsharded array
    would cause JIT cache misses and proxy OOM on Pathways.
    """
    if jax.devices()[0].platform == 'cpu':
      self.skipTest('Eager sharding validation applies to non-CPU devices')
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with use_abstract_mesh(mesh.abstract_mesh):
      x = jnp.arange(8)
      with self.assertRaises(RuntimeError) as cm:
        sharding_utils.shard(x, ('data',), eager=True)
      self.assertIn('concrete jax.sharding.Mesh', str(cm.exception))

  def test_shard_eager_with_both_meshes_uses_physical(self):
    """When both physical and abstract mesh are active, eager uses physical."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with mesh:
      with use_abstract_mesh(mesh.abstract_mesh):
        x = jnp.arange(8)
        # Should NOT raise — physical mesh is available for device_put.
        sharded = sharding_utils.shard(x, ('data',), eager=True)
        if jax.devices()[0].platform != 'cpu':
          self.assertIsInstance(sharded.sharding, shd.NamedSharding)

  def test_shard_lazy_under_abstract_mesh_does_not_raise(self):
    """Lazy (eager=False) shard with AbstractMesh does not raise."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with use_abstract_mesh(mesh.abstract_mesh):
      x = jnp.arange(8)
      # Lazy is fine — it's intended for inside JIT.
      sharded = sharding_utils.shard(x, ('data',), eager=False)
      self.assertIsNotNone(sharded)


class ValidateShardingsTest(absltest.TestCase):

  def test_valid_shardings_pass(self):
    """Correctly sharded arrays pass validation."""
    if jax.devices()[0].platform == 'cpu':
      self.skipTest('Sharding validation needs non-CPU devices')
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    sharding = shd.NamedSharding(mesh, shd.PartitionSpec('data'))
    x = jax.device_put(jnp.arange(8), sharding)
    # Should not raise.
    sharding_utils.validate_shardings(
        {'a': x}, expected_mesh=mesh, label='test'
    )

  def test_unsharded_array_fails(self):
    """Array without NamedSharding fails validation."""
    x = jnp.arange(8)  # default sharding
    if isinstance(x.sharding, shd.NamedSharding):
      self.skipTest('Default sharding is already NamedSharding on this platform')
    with self.assertRaises(ValueError) as cm:
      sharding_utils.validate_shardings({'a': x}, label='test')
    self.assertIn('expected NamedSharding', str(cm.exception))

  def test_mesh_mismatch_fails(self):
    """Array sharded on wrong mesh fails validation."""
    if jax.devices()[0].platform == 'cpu':
      self.skipTest('Sharding validation needs non-CPU devices')
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh_a = shd.Mesh(devices, ('data',))
    mesh_b = shd.Mesh(devices, ('batch',))  # different axis name
    sharding = shd.NamedSharding(mesh_a, shd.PartitionSpec('data'))
    x = jax.device_put(jnp.arange(8), sharding)
    # mesh_b has shape {'batch': 1}, mesh_a has shape {'data': 1}
    # Shapes differ in axis names.
    with self.assertRaises(ValueError):
      sharding_utils.validate_shardings(
          {'a': x}, expected_mesh=mesh_b, label='test'
      )


if __name__ == '__main__':
  absltest.main()
