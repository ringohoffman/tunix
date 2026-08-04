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

"""Tests for Gemma 4 classification model."""

from __future__ import annotations

import dataclasses

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import classification as cls_lib
from tunix.models.gemma4 import model as model_lib


def _create_small_model_config() -> model_lib.ModelConfig:
  config = model_lib.ModelConfig.gemma4_e2b()
  config.num_embed = 256
  config.num_layers = 1
  config.use_scan_layers = False
  config.embed_dim = 64
  config.hidden_dim = 128
  config.num_heads = 2
  config.head_dim = 32
  config.num_kv_heads = 1
  config.frac_shared_layers = 0.0
  return config


def _create_small_classification_config(
    head_type: cls_lib.HeadType = cls_lib.HeadType.MULTILABEL,
    num_classes: int = 3,
    class_names: tuple[str, ...] = (),
    max_examples_per_packed_sequence: int | None = None,
    pool_strategy: cls_lib.PoolStrategy = cls_lib.PoolStrategy.LAST_TOKEN,
) -> cls_lib.ClassificationModelConfig:
  config = _create_small_model_config()
  fields_dict = {
      f.name: getattr(config, f.name) for f in dataclasses.fields(config)
  }
  return cls_lib.ClassificationModelConfig(
      head_type=head_type,
      num_classes=num_classes,
      class_names=class_names,
      max_examples_per_packed_sequence=max_examples_per_packed_sequence,
      pool_strategy=pool_strategy,
      **fields_dict,
  )


class Gemma4ForClassificationTest(absltest.TestCase):

  def test_unpacked_default_mask(self):
    """Test unpacked classification when no input_mask is provided."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=3,
        class_names=("cat_a", "cat_b", "cat_c"),
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(config=config, rngs=rngs)

    batch_size, seq_len = 2, 8
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.num_embed
    )

    out = model(tokens)

    self.assertEqual(out.logits.shape, (batch_size, 3))
    self.assertEqual(out.hidden_states.shape, (batch_size, config.embed_dim))

  def test_unpacked_explicit_input_mask(self):
    """Test unpacked classification with an explicit attention_mask (padding)."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=2,
        pool_strategy=cls_lib.PoolStrategy.LAST_TOKEN,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(
        config=config,
        rngs=rngs,
    )

    batch_size, seq_len = 2, 8
    tokens = jnp.array([
        [10, 20, 30, 40, 50, 0, 0, 0],
        [10, 20, 30, 40, 50, 60, 70, 80],
    ])
    input_mask = jnp.array([
        [True, True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True, True],
    ])
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    valid = input_mask[..., :, None] & input_mask[..., None, :]
    attn_mask = causal[None, ...] & valid

    out = model(tokens, attention_mask=attn_mask)

    self.assertEqual(out.logits.shape, (batch_size, 2))
    self.assertEqual(out.hidden_states.shape, (batch_size, config.embed_dim))

  def test_unpacked_explicit_positions_and_attn_mask(self):
    """Test passing pre-computed positions and attention_mask."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=2,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(config=config, rngs=rngs)

    batch_size, seq_len = 1, 4
    tokens = jnp.array([[10, 20, 30, 0]])
    positions = jnp.array([[0, 1, 2, 0]])
    attn_mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))[None, ...]

    out = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
    )

    self.assertEqual(out.logits.shape, (1, 2))

  def test_unpacked_mean_pooling(self):
    """Test MEAN pooling strategy."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=2,
        pool_strategy=cls_lib.PoolStrategy.MEAN,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(
        config=config,
        rngs=rngs,
    )

    tokens = jnp.array([[10, 20, 30, 0]])
    input_mask = jnp.array([[True, True, True, False]])
    causal = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))
    valid = input_mask[..., :, None] & input_mask[..., None, :]
    attn_mask = causal[None, ...] & valid

    out = model(tokens, attention_mask=attn_mask)

    self.assertEqual(out.logits.shape, (1, 2))
    self.assertEqual(out.hidden_states.shape, (1, config.embed_dim))

  def test_binary_head(self):
    """Test BINARY classification head producing [B] shaped logits."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.BINARY,
        num_classes=1,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(config=config, rngs=rngs)

    tokens = jnp.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    out = model(tokens)

    self.assertEqual(out.logits.shape, (2,))

  def test_packed_mode_success(self):
    """Test packed sequence classification."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=3,
        max_examples_per_packed_sequence=2,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(
        config=config,
        rngs=rngs,
    )

    batch_size, seq_len = 2, 10
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.num_embed
    )
    positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
    # Segment layout: 0 = prefix, 1 = example 1, 2 = example 2
    segment_ids = jnp.array([
        [0, 0, 1, 1, 1, 2, 2, 2, 2, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    ])

    out = model(tokens, positions=positions, segment_ids=segment_ids)

    # Output shape should be [B * N, C] = [2 * 2, 3] = [4, 3]
    self.assertEqual(
        out.logits.shape,
        (batch_size * config.max_examples_per_packed_sequence, 3),
    )
    self.assertEqual(
        out.hidden_states.shape,
        (
            batch_size * config.max_examples_per_packed_sequence,
            config.embed_dim,
        ),
    )

  def test_packed_mode_missing_positions_error(self):
    """Test that packed mode raises ValueError if positions is omitted."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=2,
        max_examples_per_packed_sequence=2,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(
        config=config,
        rngs=rngs,
    )

    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    segment_ids = jnp.array([[0, 1, 1, 1, 2, 2, 0, 0]])

    with self.assertRaisesRegex(ValueError, "positions must be provided"):
      model(tokens, segment_ids=segment_ids)

  def test_packed_mode_missing_max_examples_error(self):
    """Test that packed mode raises ValueError if max_examples_per_packed_sequence is None."""
    config = _create_small_classification_config(
        head_type=cls_lib.HeadType.MULTILABEL,
        num_classes=2,
    )
    rngs = nnx.Rngs(0)
    model = cls_lib.Gemma4ForClassification(config=config, rngs=rngs)

    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    positions = jnp.arange(8)[None, :]
    segment_ids = jnp.array([[0, 1, 1, 1, 2, 2, 0, 0]])

    with self.assertRaisesRegex(
        ValueError, "max_examples_per_packed_sequence must be specified"
    ):
      model(tokens, positions=positions, segment_ids=segment_ids)

  def test_classification_config_validation(self):
    """Test ClassificationModelConfig error conditions."""
    with self.assertRaisesRegex(
        ValueError, "BINARY head requires num_classes=1"
    ):
      _create_small_classification_config(
          head_type=cls_lib.HeadType.BINARY, num_classes=2
      )

    with self.assertRaisesRegex(ValueError, "class_names length"):
      _create_small_classification_config(
          head_type=cls_lib.HeadType.MULTILABEL,
          num_classes=2,
          class_names=("only_one",),
      )


if __name__ == "__main__":
  absltest.main()
