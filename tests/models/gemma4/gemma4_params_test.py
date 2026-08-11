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

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.traverse_util import flatten_dict
import flax.typing
import jax
import numpy as np
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params


class _MockDevice:

  def __init__(self, device_id: int):
    self.id = device_id
    self.platform = 'CPU'
    self.device_kind = 'CPU'
    self.client = mock.MagicMock()
    self.client.platform = 'CPU'
    self.client.platform_version = '1.0'
    self.process_index = 0

  def default_memory(self):
    mock_memory = mock.MagicMock()
    mock_memory.kind = 'device'
    return mock_memory

  def __lt__(self, other):
    return self.id < getattr(other, 'id', 0)

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return self.id == getattr(other, 'id', None)


# Small array dimensions used across all fixtures.
_V, _D, _H, _KV, _N, _F, _PLE = 5, 3, 2, 1, 4, 6, 5


def _layer_arrays(offset: int | float = 0) -> dict[str, np.ndarray]:
  """Per-layer arrays with a numeric offset to distinguish layers."""
  o = np.float32(offset)
  return {
      'gate_up': (
          np.arange(2 * _F * _D, dtype=np.float32).reshape(2, _F, _D) + o
      ),
      'down': np.arange(_F * _D, dtype=np.float32).reshape(_F, _D) + o,
      'q_w': np.arange(_N * _D * _H, dtype=np.float32).reshape(_N, _D, _H) + o,
      'kv_w': (
          np.arange(2 * _KV * _D * _H, dtype=np.float32).reshape(2, _KV, _D, _H)
          + o
      ),
      'o_w': np.arange(_N * _H * _D, dtype=np.float32).reshape(_N, _H, _D) + o,
      'pre_attn': np.arange(_D, dtype=np.float32) + o,
      'post_attn': np.arange(_D, dtype=np.float32) + o,
      'pre_ffw': np.arange(_D, dtype=np.float32) + o,
      'post_ffw': np.arange(_D, dtype=np.float32) + o,
      'skip_scale': np.array([0.5 + offset], dtype=np.float32),
      'query_norm': np.arange(_H, dtype=np.float32) + o,
      'key_norm': np.arange(_H, dtype=np.float32) + o,
  }


def _semiflat_layer(idx: int, arrs: dict[str, np.ndarray]) -> dict[str, Any]:
  """Builds semi-flat entries for one layer."""
  p = f'transformer/layer_{idx}'
  return {
      f'{p}/attn/_key_norm': {'scale': arrs['key_norm']},
      f'{p}/attn/_query_norm': {'scale': arrs['query_norm']},
      f'{p}/attn/attn_vec_einsum': {'w': arrs['o_w']},
      f'{p}/attn/kv_einsum': {'w': arrs['kv_w']},
      f'{p}/attn/q_einsum': {'w': arrs['q_w']},
      f'{p}/mlp/gating_einsum': {'w': arrs['gate_up']},
      f'{p}/mlp/linear': {'w': arrs['down']},
      f'{p}/post_attention_norm': {'scale': arrs['post_attn']},
      f'{p}/post_ffw_norm': {'scale': arrs['post_ffw']},
      f'{p}/pre_attention_norm': {'scale': arrs['pre_attn']},
      f'{p}/pre_ffw_norm': {'scale': arrs['pre_ffw']},
      f'{p}': {'skip_scale': arrs['skip_scale']},
  }


def _make_upstream_semiflat() -> dict[str, Any]:
  """Builds a multi-layer upstream checkpoint in semi-flat 2-tuple format.

  Semi-flat keys: ('transformer/layer_0/attn/q_einsum', 'w')
  Includes layer_0 and layer_1 to exercise multi-layer index parsing.
  """
  embed = np.arange(_V * _D, dtype=np.float32).reshape(_V, _D)
  final_scale = np.arange(_D, dtype=np.float32)
  per_layer_emb = np.arange(_D * _PLE, dtype=np.float32).reshape(_D, _PLE)

  d = {
      'transformer/embedder': {'input_embedding': embed},
      'transformer/embedder/per_layer_embeddings': {'w': per_layer_emb},
      'transformer/final_norm': {'scale': final_scale},
  }
  d.update(_semiflat_layer(0, _layer_arrays(offset=0)))
  d.update(_semiflat_layer(1, _layer_arrays(offset=100)))
  return d


def _make_upstream_nested() -> dict[str, Any]:
  """Builds the same checkpoint in genuinely nested N-tuple format.

  Nested keys: ('transformer', 'layer_0', 'attn', 'q_einsum', 'w')
  """
  sf = _make_upstream_semiflat()
  nested = {}
  for key, sub_dict in sf.items():
    parts = tuple(key.split('/'))
    current = nested
    for part in parts[:-1]:
      current = current.setdefault(part, {})
    leaf = parts[-1]
    # Merge into existing dict to avoid overwriting sibling sub-paths
    # (e.g., 'transformer/layer_0' shouldn't clobber
    # 'transformer/layer_0/attn').
    if leaf in current and isinstance(current[leaf], dict):
      current[leaf].update(sub_dict)
    else:
      current[leaf] = sub_dict
  return nested


def _expected_keys_and_shapes() -> dict[tuple[str, ...], tuple[int, ...]]:
  """Returns expected output keys and shapes after mapping (both layers)."""
  result = {
      ('embedder', 'input_embedding'): (_V, _D),
      ('embedder', 'per_layer_input_embedding'): (_D, _PLE),
      ('final_norm', 'scale'): (_D,),
  }
  for i in range(2):
    result.update({
        ('layers', i, 'attn', '_key_norm', 'scale'): (_H,),
        ('layers', i, 'attn', '_query_norm', 'scale'): (_H,),
        ('layers', i, 'attn', 'attn_vec_einsum', 'w'): (_N, _H, _D),
        ('layers', i, 'attn', 'kv_einsum', 'w'): (2, _KV, _D, _H),
        ('layers', i, 'attn', 'q_einsum', 'w'): (_N, _D, _H),
        ('layers', i, 'mlp', 'down_proj', 'kernel'): (_F, _D),
        ('layers', i, 'mlp', 'gate_proj', 'kernel'): (_D, _F),
        ('layers', i, 'mlp', 'up_proj', 'kernel'): (_D, _F),
        ('layers', i, 'post_attention_norm', 'scale'): (_D,),
        ('layers', i, 'post_ffw_norm', 'scale'): (_D,),
        ('layers', i, 'pre_attention_norm', 'scale'): (_D,),
        ('layers', i, 'pre_ffw_norm', 'scale'): (_D,),
        ('layers', i, 'skip_scale'): (1,),
    })
  return result


class MapFromUpstreamCheckpointTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_keys_and_shapes(self, make_fn):
    """Verifies all expected keys exist with correct shapes."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    expected = _expected_keys_and_shapes()
    for key, shape in expected.items():
      with self.subTest(key=key):
        self.assertIn(key, flat, msg=f'Missing key {key}')
        self.assertEqual(
            flat[key].shape,
            shape,
            msg=(
                f'Shape mismatch for {key}: got {flat[key].shape}, want {shape}'
            ),
        )

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_mlp_gating_transpose(self, make_fn):
    """Verifies MLP gating_einsum is split and transposed correctly."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    # Get the original gating_einsum value (layer 0).
    upstream_flat = flatten_dict(upstream)
    gate_up_key = [k for k in upstream_flat if 'gating_einsum' in str(k)][0]
    gate_up = upstream_flat[gate_up_key]

    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'gate_proj', 'kernel')],
        gate_up[0].T,
    )
    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'up_proj', 'kernel')],
        gate_up[1].T,
    )

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_passthrough_values(self, make_fn):
    """Verifies pass-through params (e.g., MLP down_proj) are not modified."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    # The down_proj kernel is passed through without transpose. Verify by
    # reconstructing the expected value from the fixture constants.
    expected_down = _layer_arrays(offset=0)['down']
    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'down_proj', 'kernel')],
        expected_down,
    )

  def test_format_parity(self):
    """Both checkpoint formats must produce identical output."""
    sf_mapped = params.map_from_upstream_checkpoint(_make_upstream_semiflat())
    nested_mapped = params.map_from_upstream_checkpoint(_make_upstream_nested())

    sf_flat = flatten_dict(sf_mapped)
    nested_flat = flatten_dict(nested_mapped)

    self.assertEqual(set(sf_flat.keys()), set(nested_flat.keys()))
    for key, sf_value in sf_flat.items():
      with self.subTest(key=key):
        np.testing.assert_array_equal(
            sf_value,
            nested_flat[key],
            err_msg=f'Value mismatch for {key} between formats',
        )

  def test_non_layer_modules_skipped(self):

    upstream = _make_upstream_semiflat()
    upstream['transformer/audio_encoder'] = {
        'weight': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    audio_keys = [k for k in flat if 'audio_encoder' in str(k)]
    self.assertEmpty(audio_keys)

  @parameterized.named_parameters(
      dict(
          testcase_name='query_norm_no_underscore',
          upstream={
              'transformer/layer_0/attn/query_norm': {
                  'scale': np.arange(2, dtype=np.float32),
              },
          },
          expected_key=('layers', 0, 'attn', '_query_norm', 'scale'),
          expected_shape=(2,),
      ),
      dict(
          testcase_name='key_norm_no_underscore',
          upstream={
              'transformer/layer_0/attn/key_norm': {
                  'scale': np.arange(2, dtype=np.float32),
              },
          },
          expected_key=('layers', 0, 'attn', '_key_norm', 'scale'),
          expected_shape=(2,),
      ),
      dict(
          testcase_name='embedder_per_layer_as_leaf',
          upstream={
              'transformer/embedder': {
                  'per_layer_input_embedding': np.ones(
                      (_D, _PLE), dtype=np.float32
                  ),
              },
          },
          expected_key=('embedder', 'per_layer_input_embedding'),
          expected_shape=(_D, _PLE),
      ),
      dict(
          testcase_name='embedder_mm_submodule',
          upstream={
              'transformer/embedder/mm_input_projection': {
                  'w': np.ones((4, 8), dtype=np.float32),
              },
          },
          expected_key=('embedder', 'mm_input_projection', 'w'),
          expected_shape=(4, 8),
      ),
  )
  def test_mapper_edge_cases(self, upstream, expected_key, expected_shape):

    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)
    self.assertIn(expected_key, flat, msg=f'Missing key {expected_key}')
    self.assertEqual(flat[expected_key].shape, expected_shape)

  def test_gating_einsum_bad_shape_raises(self):

    upstream = {
        'transformer/layer_0/mlp/gating_einsum': {
            'w': np.ones((3, _F, _D), dtype=np.float32),
        },
    }
    with self.assertRaisesRegex(ValueError, r'gating_einsum shape\[0\]=2'):
      params.map_from_upstream_checkpoint(upstream)


class PruneToModelKeysTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            params.nnx, 'to_pure_dict', side_effect=lambda x: x, autospec=True
        )
    )

  def test_extra_keys_pruned(self):

    checkpoint = {'a': {'x': np.array(1.0)}, 'b': {'y': np.array(2.0)}}
    model_state = {'a': {'x': np.array(0.0)}}

    pruned = params._prune_to_model_keys(checkpoint, model_state)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('a', 'x'), flat)
    self.assertNotIn(('b', 'y'), flat)

  def test_expected_keys_preserved(self):

    shared = {'a': {'x': np.array(1.0)}, 'b': {'y': np.array(2.0)}}

    pruned = params._prune_to_model_keys(shared, shared)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('a', 'x'), flat)
    self.assertIn(('b', 'y'), flat)

  def test_deep_nesting_with_integer_keys(self):
    """Flatten/unflatten round-trip preserves deeply nested integer keys."""
    checkpoint = {
        'layers': {0: {'mlp': {'gate_proj': {'kernel': np.array(1.0)}}}},
        'extra': {'junk': np.array(2.0)},
    }
    model_state = {
        'layers': {0: {'mlp': {'gate_proj': {'kernel': np.array(0.0)}}}},
    }

    pruned = params._prune_to_model_keys(checkpoint, model_state)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('layers', 0, 'mlp', 'gate_proj', 'kernel'), flat)
    self.assertNotIn(('extra', 'junk'), flat)


class ValidateParamShapesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            params.nnx, 'to_pure_dict', side_effect=lambda x: x, autospec=True
        )
    )

  def test_missing_keys_raises(self):

    checkpoint = {'a': {'x': np.array([1.0])}}
    model_state = {
        'a': {'x': np.array([0.0])},
        'b': {'y': np.array([0.0])},
    }

    with self.assertRaisesRegex(ValueError, 'missing keys'):
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

  def test_shape_mismatch_raises(self):

    checkpoint = {'a': {'x': np.array([1.0, 2.0])}}  # shape (2,)
    model_state = {'a': {'x': np.array([0.0])}}  # shape (1,)

    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

  def test_matching_shapes_passes(self):

    data = {'a': {'x': np.array([1.0, 2.0])}}
    params._validate_param_shapes(data, data)  # pylint: disable=protected-access

  def test_extra_keys_warns(self):
    """Extra checkpoint keys (superset of model) should log a warning."""
    checkpoint = {
        'a': {'x': np.array([1.0])},
        'b': {'y': np.array([2.0])},
    }
    model_state = {'a': {'x': np.array([0.0])}}

    with self.assertLogs(level='WARNING') as cm:
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

    self.assertTrue(
        any('extra keys' in msg.lower() for msg in cm.output),
        msg=f'Expected warning about extra keys, got: {cm.output}',
    )

  def test_non_array_values_warns(self):

    data = {'a': {'x': 'not_an_array'}}

    with self.assertLogs(level='WARNING') as cm:
      params._validate_param_shapes(data, data)  # pylint: disable=protected-access

    self.assertTrue(
        any('non-array' in msg.lower() for msg in cm.output),
        msg=f'Expected warning about non-array values, got: {cm.output}',
    )


class CreateModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_map = self.enter_context(
        mock.patch.object(params, 'map_from_upstream_checkpoint', autospec=True)
    )
    self.mock_prune = self.enter_context(
        mock.patch.object(params, '_prune_to_model_keys', autospec=True)
    )
    self.mock_validate = self.enter_context(
        mock.patch.object(params, '_validate_param_shapes', autospec=True)
    )
    self.mock_update = self.enter_context(
        mock.patch.object(params.nnx, 'update', autospec=True)
    )
    self.mock_state = self.enter_context(
        mock.patch.object(params.nnx, 'state', autospec=True)
    )
    self.mock_eval_shape = self.enter_context(
        mock.patch.object(params.nnx, 'eval_shape', autospec=True)
    )
    self.mock_ckptr_cls = self.enter_context(
        mock.patch.object(params.ocp, 'PyTreeCheckpointer', autospec=True)
    )

    self.fake_model = mock.create_autospec(object, instance=True)
    self.mock_eval_shape.return_value = self.fake_model
    self.fake_params = {'layers': {0: {'w': np.array([1.0])}}}
    self.mock_ckptr_cls.return_value.restore.return_value = self.fake_params
    self.mock_map.return_value = self.fake_params
    self.mock_prune.return_value = self.fake_params
    self.mock_state.return_value = nnx.State(self.fake_params)
    self.config = mock.create_autospec(
        gemma4_model.ModelConfig, instance=True
    )
    self.config.use_scan_layers = False
    self.checkpoint_path = '/fake/checkpoint'

  def test_create_model_restores_checkpoint(self):
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_ckptr_cls.return_value.restore.assert_called_once_with(
        self.checkpoint_path
    )

  def test_create_model_processes_params(self):
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_map.assert_called_once()
    self.mock_prune.assert_called_once()
    self.mock_validate.assert_called_once()

  def test_create_model_updates_nnx_state(self):
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_update.assert_called()

  def test_create_model_returns_model_instance(self):
    result = params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.assertIs(result, self.fake_model)

  @mock.patch.object(params.spm, 'SentencePieceProcessor', autospec=True)
  @mock.patch.object(params.epath, 'Path', autospec=True)
  def test_create_tokenizer(self, mock_path_cls, mock_spm_cls):
    """Verifies tokenizer loads bytes and initializes processor."""
    fake_bytes = b'fake-sentencepiece-model'
    mock_path_cls.return_value.read_bytes.return_value = fake_bytes
    mock_processor = mock_spm_cls.return_value

    result = params.create_tokenizer('/fake/tokenizer.model')

    mock_path_cls.assert_called_once_with('/fake/tokenizer.model')
    mock_processor.LoadFromSerializedProto.assert_called_once_with(fake_bytes)
    self.assertIs(result, mock_processor)


def _make_array_metadata_tree(
    tree: flax.typing.PyTree[params.LeafT],
) -> flax.typing.PyTree[params.ocp.metadata.ArrayMetadata]:
  """Converts a dict of numpy arrays to a dict of ocp.metadata.ArrayMetadata."""

  def _to_meta(x: Any) -> Any:
    if isinstance(x, (np.ndarray, jax.Array, jax.ShapeDtypeStruct)):
      return params.ocp.metadata.ArrayMetadata(
          name='',
          directory=None,
          shape=x.shape,
          sharding=None,
          dtype=np.dtype(x.dtype),
      )
    return x

  return jax.tree.map(
      _to_meta,
      tree,
      is_leaf=lambda x: isinstance(
          x, (np.ndarray, jax.Array, jax.ShapeDtypeStruct)
      ),
  )


class BuildShardedRestoreTargetTest(absltest.TestCase):
  """Tests for _build_sharded_restore_target."""

  def setUp(self):
    super().setUp()
    self.mock_ckptr_cls = self.enter_context(
        mock.patch.object(params.ocp, 'PyTreeCheckpointer', autospec=True)
    )
    self.mock_ckptr = self.mock_ckptr_cls.return_value

    # Upstream keys: one layer parameter, one non-layer parameter
    self.fake_upstream = {
        'layer_0': {'mlp': {'gating_einsum': {'w': np.zeros((2, 4, 8))}}},
        'embedder': {'input_embedding': np.zeros((16, 4))},
    }
    mock_meta = mock.MagicMock()
    mock_meta.item_metadata.tree = _make_array_metadata_tree(self.fake_upstream)
    self.mock_ckptr.metadata.return_value = mock_meta

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    self.mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
    self.enter_context(jax.set_mesh(self.mesh))

  def test_build_target_without_scan_layers(self):
    config = mock.create_autospec(
        gemma4_model.ModelConfig, instance=True
    )
    config.use_scan_layers = False

    # Downstream layout matches upstream map output exactly
    model_state = nnx.state(
        nnx.Dict({
            'layers': nnx.List([
                nnx.Dict({
                    'mlp': nnx.Dict({
                        'gate_proj': nnx.Dict({
                            'kernel': nnx.Param(
                                np.zeros((4, 8)),
                                sharding=jax.sharding.PartitionSpec(
                                    'fsdp', 'tp'
                                ),
                            )
                        }),
                        'up_proj': nnx.Dict({
                            'kernel': nnx.Param(
                                np.zeros((4, 8)),
                                sharding=jax.sharding.PartitionSpec(
                                    'fsdp', 'tp'
                                ),
                            )
                        }),
                    })
                })
            ]),
            'embedder': nnx.Dict({
                'input_embedding': nnx.Param(
                    np.zeros((16, 4)),
                    sharding=jax.sharding.PartitionSpec('fsdp', None),
                )
            }),
        })
    )

    target, _ = params._build_sharded_restore_target(
        '/fake/checkpoint', model_state, self.mesh, config
    )

    # For embedder: matches directly, spec is ('fsdp', None)
    self.assertEqual(
        target['embedder']['input_embedding'].sharding.spec,
        jax.sharding.PartitionSpec('fsdp', None),
    )

    # For layer 0 mlp gate_proj: upstream is gating_einsum/w.
    # Inverted spec: ('fsdp', 'tp')[::-1] -> ('tp', 'fsdp') -> prepends None -> (None, 'tp', 'fsdp')
    self.assertEqual(
        target['layer_0']['mlp']['gating_einsum']['w'].sharding.spec,
        jax.sharding.PartitionSpec(None, 'tp', 'fsdp'),
    )

  def test_build_target_with_scan_layers(self):
    config = mock.create_autospec(
        gemma4_model.ModelConfig, instance=True
    )
    config.use_scan_layers = True
    config.attention_pattern = (gemma4_model.AttentionType.GLOBAL,)
    config.num_layers = 1
    config.frac_shared_layers = 0.0

    # Downstream scan layout: scan_groups/sub_groups/0/layers/...
    model_state = nnx.state(
        nnx.Dict({
            'scan_groups': nnx.Dict({
                'sub_groups': nnx.List([
                    nnx.Dict({
                        'layers': nnx.Dict({
                            'mlp': nnx.Dict({
                                'gate_proj': nnx.Dict({
                                    'kernel': nnx.Param(
                                        np.zeros((1, 1, 4, 8)),
                                        sharding=jax.sharding.PartitionSpec(
                                            'fsdp', 'tp'
                                        ),
                                    )
                                }),
                                'up_proj': nnx.Dict({
                                    'kernel': nnx.Param(
                                        np.zeros((1, 1, 4, 8)),
                                        sharding=jax.sharding.PartitionSpec(
                                            'fsdp', 'tp'
                                        ),
                                    )
                                }),
                            })
                        })
                    })
                ])
            }),
            'embedder': nnx.Dict({
                'input_embedding': nnx.Param(
                    np.zeros((16, 4)),
                    sharding=jax.sharding.PartitionSpec('fsdp', None),
                )
            }),
        })
    )

    target, _ = params._build_sharded_restore_target(
        '/fake/checkpoint', model_state, self.mesh, config
    )

    # Embedder remains unsharded scan, spec is unmodified: ('fsdp', None)
    self.assertEqual(
        target['embedder']['input_embedding'].sharding.spec,
        jax.sharding.PartitionSpec('fsdp', None),
    )

    # For layer_0 gating_einsum/w:
    # Downstream sharding on scan_groups is ('fsdp', 'tp') (unstacked).
    # Tracer inverts: transposes -> ('tp', 'fsdp'), prepends None -> (None, 'tp', 'fsdp')
    self.assertEqual(
        target['layer_0']['mlp']['gating_einsum']['w'].sharding.spec,
        jax.sharding.PartitionSpec(None, 'tp', 'fsdp'),
    )

  def test_real_model_sharded_restore_target_with_scan_layers(self):
    config = gemma4_model.ModelConfig(
        num_layers=6,
        num_embed=256000,
        embed_dim=5376,
        hidden_dim=16384,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        attention_pattern=(
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.GLOBAL,
        ),
        use_scan_layers=True,
    )

    mock_devices = [_MockDevice(i) for i in range(32)]
    mesh = jax.sharding.Mesh(
        np.array(mock_devices).reshape(32, 1), ('fsdp', 'tp')
    )

    with nnx.use_eager_sharding(True), jax.set_mesh(mesh):
      abs_model = nnx.eval_shape(
          lambda: gemma4_model.Gemma4(config, rngs=nnx.Rngs(0))
      )
    model_state = nnx.state(abs_model)

    fake_upstream = {
        'layer_0': {'attn': {'kv_einsum': {'w': np.zeros((2, 16, 5376, 256))}}}
    }
    mock_meta = mock.MagicMock()
    mock_meta.item_metadata.tree = _make_array_metadata_tree(fake_upstream)
    self.mock_ckptr.metadata.return_value = mock_meta

    target, _ = params._build_sharded_restore_target(
        '/fake/checkpoint', model_state, mesh, config
    )

    # Let's check if calling shard_shape raises an error or passes
    sharding = target['layer_0']['attn']['kv_einsum']['w'].sharding
    global_shape = (2, 16, 5376, 256)
    # This should succeed without IndivisibleError.
    shard_shape = sharding.shard_shape(global_shape)
    self.assertEqual(shard_shape, (2, 16, 168, 256))

  def test_create_model_from_checkpoint_with_scan_layers(self):
    config = gemma4_model.ModelConfig(
        num_layers=6,
        num_embed=256000,
        embed_dim=5376,
        hidden_dim=16384,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        global_key_size=256,
        num_global_kv_heads=16,
        attention_pattern=(
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.GLOBAL,
        ),
        use_scan_layers=True,
    )

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    fake_upstream = {}
    for i in range(6):
      h_dim = 256
      attn_dict: dict[str, Any] = {
          'kv_einsum': {'w': np.zeros((2, 16, 5376, h_dim))},
          'q_einsum': {'w': np.zeros((32, 5376, h_dim))},
          'attn_vec_einsum': {'w': np.zeros((32, h_dim, 5376))},
          'query_norm': {'scale': np.zeros((h_dim,))},
          'key_norm': {'scale': np.zeros((h_dim,))},
      }
      fake_upstream[f'layer_{i}'] = {
          'attn': attn_dict,
          'mlp': {
              'gating_einsum': {'w': np.zeros((2, 16384, 5376))},
              'linear': {'w': np.zeros((16384, 5376))},
          },
          'pre_attention_norm': {'scale': np.zeros((5376,))},
          'post_attention_norm': {'scale': np.zeros((5376,))},
          'pre_ffw_norm': {'scale': np.zeros((5376,))},
          'post_ffw_norm': {'scale': np.zeros((5376,))},
          'skip_scale': np.zeros((1,)),
      }
    fake_upstream['embedder'] = {'input_embedding': np.zeros((256000, 5376))}
    fake_upstream['final_norm'] = {'scale': np.zeros((5376,))}

    mock_meta = mock.MagicMock()
    mock_meta.item_metadata.tree = _make_array_metadata_tree(fake_upstream)
    self.mock_ckptr.metadata.return_value = mock_meta
    self.mock_ckptr.restore.return_value = fake_upstream

    # This should load successfully without IndivisibleError or other exceptions.
    model = params.create_model_from_checkpoint(
        '/fake/checkpoint', config, mesh=mesh
    )
    self.assertIsInstance(model, gemma4_model.Gemma4)

    # Verify that the stacked parameters have the correct sharding spec
    state = nnx.state(model)
    self.assertEqual(
        state.scan_groups.sub_groups[0].layers.attn.kv_einsum.w.sharding.spec,
        jax.sharding.PartitionSpec(),
    )

  def test_map_from_upstream_checkpoint_fine_tuned_layout(self):
    """Verifies key remapping for fine-tuned Tunix/Linen checkpoint formats."""
    fine_tuned_raw = {
        'token_embedder': {'embedding': {'value': np.zeros((10, 8))}},
        'decoder': {
            'decoder_norm': {'scale': {'value': np.zeros((8,))}},
            'layers_0': {
                'pre_self_attention_norm': {'scale': {'value': np.zeros((8,))}},
                'self_attention': {
                    'query': {'kernel': {'value': np.zeros((8, 8))}},
                    'key': {'kernel': {'value': np.zeros((8, 8))}},
                    'value': {'kernel': {'value': np.zeros((8, 8))}},
                    'out': {'kernel': {'value': np.zeros((8, 8))}},
                },
                'mlp': {
                    'wi_0': {'kernel': {'value': np.zeros((8, 16))}},
                    'wi_1': {'kernel': {'value': np.zeros((8, 16))}},
                    'wo': {'kernel': {'value': np.zeros((16, 8))}},
                },
            },
        },
    }
    mapped = params.map_from_upstream_checkpoint(fine_tuned_raw)
    flat_mapped = flatten_dict(mapped)

    self.assertIn(('embedder', 'input_embedding'), flat_mapped)
    self.assertIn(('final_norm', 'scale'), flat_mapped)
    self.assertIn(('layers', 0, 'pre_attention_norm', 'scale'), flat_mapped)
    self.assertIn(('layers', 0, 'attn', 'q_einsum', 'w'), flat_mapped)
    self.assertIn(('layers', 0, 'attn', 'k_einsum', 'w'), flat_mapped)
    self.assertIn(('layers', 0, 'attn', 'v_einsum', 'w'), flat_mapped)
    self.assertIn(('layers', 0, 'attn', 'attn_vec_einsum', 'w'), flat_mapped)
    self.assertIn(('layers', 0, 'mlp', 'gate_proj', 'kernel'), flat_mapped)
    self.assertIn(('layers', 0, 'mlp', 'up_proj', 'kernel'), flat_mapped)
    self.assertIn(('layers', 0, 'mlp', 'down_proj', 'kernel'), flat_mapped)

  def test_create_model_from_fine_tuned_checkpoint_with_scan_layers(self):
    """Full pipeline: Linen fine-tuned checkpoint → scan-layer model.

    Exercises the exact checkpoint format produced by the Linen-based trainer
    (the yg-balance-v2-sft checkpoint), with:
    - Linen key names (decoder/layers_N/self_attention/query/kernel/value)
    - kernel→w transpose for 3D einsum tensors
    - k_eq_v_global=True (GLOBAL layers have only 'key', no 'value')
    - Attention pattern adapter (separate k/v → kv_einsum for LOCAL_SLIDING)
    - Scan layer stacking across 2 groups of 6 sub-layers
    """
    _E, _H, _KV, _HD = 64, 4, 2, 16  # embed, heads, kv_heads, head_dim
    _GKV, _GHD = 1, 32  # global kv_heads, global head_dim
    _F = 128  # hidden_dim

    config = gemma4_model.ModelConfig(
        num_layers=12,
        num_embed=128,
        embed_dim=_E,
        hidden_dim=_F,
        num_heads=_H,
        num_kv_heads=_KV,
        head_dim=_HD,
        global_key_size=_GHD,
        num_global_kv_heads=_GKV,
        sliding_window_size=16,
        attention_pattern=gemma4_model.GEMMA4_ATTENTION_PATTERN,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        k_eq_v_global=True,
        use_scan_layers=True,
        use_flash_attention=False,
    )

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    # Build a fake checkpoint in Linen format (genuinely nested)
    fine_tuned_raw = {
        'token_embedder': {'embedding': {'value': np.ones((_E * 2, _E))}},
        'decoder': {
            'decoder_norm': {'scale': {'value': np.ones((_E,))}},
        },
    }
    for i in range(12):
      is_global = i % 6 == 5
      if is_global:
        kv_heads = _GKV
        head_dim = _GHD
        # k_eq_v_global: only 'key', no 'value'
        attn_dict = {
            'query': {'kernel': {'value': np.ones((_E, _H, head_dim))}},
            'key': {'kernel': {'value': np.ones((_E, kv_heads, head_dim))}},
            'out': {'kernel': {'value': np.ones((_H, head_dim, _E))}},
            'query_norm': {'scale': {'value': np.ones((head_dim,))}},
            'key_norm': {'scale': {'value': np.ones((head_dim,))}},
        }
      else:
        kv_heads = _KV
        head_dim = _HD
        # LOCAL_SLIDING: separate 'key' and 'value'
        attn_dict = {
            'query': {'kernel': {'value': np.ones((_E, _H, head_dim))}},
            'key': {'kernel': {'value': np.ones((_E, kv_heads, head_dim))}},
            'value': {'kernel': {'value': np.ones((_E, kv_heads, head_dim))}},
            'out': {'kernel': {'value': np.ones((_H, head_dim, _E))}},
            'query_norm': {'scale': {'value': np.ones((head_dim,))}},
            'key_norm': {'scale': {'value': np.ones((head_dim,))}},
        }
      fine_tuned_raw['decoder'][f'layers_{i}'] = {
          'self_attention': attn_dict,
          'mlp': {
              'wi_0': {'kernel': {'value': np.ones((_E, _F))}},
              'wi_1': {'kernel': {'value': np.ones((_E, _F))}},
              'wo': {'kernel': {'value': np.ones((_F, _E))}},
          },
          'pre_self_attention_norm': {'scale': {'value': np.ones((_E,))}},
          'post_self_attention_norm': {'scale': {'value': np.ones((_E,))}},
          'pre_ffw_norm': {'scale': {'value': np.ones((_E,))}},
          'post_ffw_norm': {'scale': {'value': np.ones((_E,))}},
          'layer_scalar': {'value': np.ones((1,))},
      }

    mock_meta = mock.MagicMock()
    mock_meta.item_metadata.tree = _make_array_metadata_tree(fine_tuned_raw)
    self.mock_ckptr.metadata.return_value = mock_meta
    self.mock_ckptr.restore.return_value = fine_tuned_raw

    model = params.create_model_from_checkpoint(
        '/fake/checkpoint', config, mesh=mesh
    )
    self.assertIsInstance(model, gemma4_model.Gemma4)

    # Verify scan group structure exists
    self.assertTrue(hasattr(model, 'scan_groups'))
    self.assertEqual(model.num_scan_groups, 2)  # 12 layers / 6 pattern

    # Verify stacked parameter shapes have the scan group axis prepended
    state = nnx.state(model)
    flat_state = flatten_dict(nnx.to_pure_dict(state))

    # LOCAL_SLIDING sub_group (index 0): in sub_groups[0].layers
    kv_key = ('scan_groups', 'sub_groups', 0, 'layers', 'attn', 'kv_einsum', 'w')
    self.assertIn(kv_key, flat_state)
    # Shape: (num_groups=2, count=5, 2, kv_heads, embed, head_dim)
    self.assertEqual(flat_state[kv_key].shape, (2, 5, 2, _KV, _E, _HD))

    # GLOBAL sub_group (index 1): in sub_groups[1].layers
    k_key = ('scan_groups', 'sub_groups', 1, 'layers', 'attn', 'k_einsum', 'w')
    self.assertIn(k_key, flat_state)
    # Shape: (num_groups=2, count=1, global_kv_heads, embed, global_head_dim)
    self.assertEqual(flat_state[k_key].shape, (2, 1, _GKV, _E, _GHD))

  def test_stack_layers_for_scan_int_layer_keys(self):
    """Tests _stack_layers_for_scan with integer layer keys.

    _stack_layers_for_scan always receives integer layer keys because
    map_from_upstream_checkpoint constructs them as ('layers', int(...)).
    """
    dummy_params = {
        'layers': {
            0: {'attn': {'w': np.ones((4, 4))}},
            1: {'attn': {'w': np.ones((4, 4)) * 2}},
        }
    }
    stacked = params._stack_layers_for_scan(
        dummy_params,
        num_layers=2,
        pattern=(
            gemma4_model.AttentionType.LOCAL_SLIDING,
            gemma4_model.AttentionType.GLOBAL,
        ),
        frac_shared_layers=0.0,
    )
    self.assertIn('scan_groups', stacked)
    self.assertIn('sub_groups', stacked['scan_groups'])
    # Stacked shape for sub_groups[0] (local) should be (num_groups=1, count=1, 4, 4)
    self.assertEqual(
        stacked['scan_groups']['sub_groups'][0]['layers']['attn']['w'].shape,
        (1, 1, 4, 4),
    )
    # Stacked shape for sub_groups[1] (global) should be (num_groups=1, count=1, 4, 4)
    self.assertEqual(
        stacked['scan_groups']['sub_groups'][1]['layers']['attn']['w'].shape,
        (1, 1, 4, 4),
    )


if __name__ == '__main__':
  absltest.main()
