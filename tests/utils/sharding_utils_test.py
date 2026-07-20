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


class ShardingUtilsTest(absltest.TestCase):

  def test_get_current_mesh_none_when_no_mesh(self):
    """When no mesh is active, get_current_mesh returns None."""
    mesh = sharding_utils.get_current_mesh()
    self.assertIsNone(mesh)

  def test_get_current_mesh_under_set_mesh(self):
    """When jax.set_mesh is active, get_current_mesh returns active mesh environment."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with jax.set_mesh(mesh):
      current = sharding_utils.get_current_mesh()
      self.assertIsNotNone(current)
      self.assertEqual(current.axis_names, mesh.axis_names)

  def test_shard_fallback_returns_asarray(self):
    """When CPU or no mesh active, shard() returns plain jnp.asarray(x)."""
    x = [1.0, 2.0, 3.0]
    sharded = sharding_utils.shard(x, ('data',), eager=True)
    np.testing.assert_allclose(sharded, np.array(x))

  def test_shard_under_set_mesh(self):
    """Under active set_mesh, shard() returns array sharded under NamedSharding or plain array on CPU."""
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

  def test_shard_eager_under_abstract_mesh_does_not_raise(self):
    """Under AbstractMesh, eager=True falls back to with_sharding_constraint instead of raising in device_put."""
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = shd.Mesh(devices, ('data',))
    with use_abstract_mesh(mesh.abstract_mesh):
      x = jnp.arange(8)
      # Should not raise ValueError: is_fully_addressable is not implemented for AbstractMesh
      sharded = sharding_utils.shard(x, ('data',), eager=True)
      self.assertIsNotNone(sharded)


if __name__ == '__main__':
  absltest.main()
