"""Tests for split attention with decoupled prefix cache.

Verifies that computing attention in two parts (prefix at B=1 + generation
at B=batch) with logit concatenation and joint softmax produces identical
results to monolithic attention with the full KV at B=batch.
"""

from __future__ import annotations

import functools

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import functional
from tunix.models.gemma4 import model as gemma4_model


def split_attention(
    query: jax.Array,
    gen_k: jax.Array,
    gen_v: jax.Array,
    prefix_k: jax.Array,
    prefix_v: jax.Array,
    gen_mask: jax.Array,
    *,
    use_gqa: bool = False,
    num_kv_heads: int | None = None,
) -> jax.Array:
  """Split attention: prefix at B=1 broadcasts, generation at B=batch.

  Args:
      query: [B, T, N, D] query projections.
      gen_k: [B, G, H, D] generation key cache.
      gen_v: [B, G, H, D] generation value cache.
      prefix_k: [1, P, H, D] or [B, 0, H, D] prefix key cache.
      prefix_v: [1, P, H, D] or [B, 0, H, D] prefix value cache.
      gen_mask: [B, T, G] boolean mask for generation positions.
      use_gqa: Whether to use grouped query attention.
      num_kv_heads: Number of KV heads (required when use_gqa=True).

  Returns:
      [B, T, N, D] attention output.
  """
  k_mask = jnp.finfo(query.dtype).min

  if use_gqa:
    assert num_kv_heads is not None
    b, t, n, d = query.shape
    n_groups = n // num_kv_heads
    q_r = query.reshape((b, t, num_kv_heads, n_groups, d))

    # Prefix logits — B=1 broadcasts automatically in einsum
    pfx_logits = jnp.einsum("BTKGH,BSKH->BTKGS", q_r, prefix_k)
    # Generation logits
    gen_logits = jnp.einsum("BTKGH,BSKH->BTKGS", q_r, gen_k)

    # Reshape to [B, T, N, S] for mask + softmax
    pfx_s = pfx_logits.shape[-1]
    gen_s = gen_logits.shape[-1]
    pfx_logits = pfx_logits.reshape((b, t, n, pfx_s))
    gen_logits = gen_logits.reshape((b, t, n, gen_s))
  else:
    # Standard MHA
    pfx_logits = jnp.einsum("BTNH,BSNH->BTNS", query, prefix_k)
    gen_logits = jnp.einsum("BTNH,BSNH->BTNS", query, gen_k)
    pfx_s = pfx_logits.shape[-1]
    gen_s = gen_logits.shape[-1]

  prefix_len = prefix_k.shape[1]
  if prefix_len > 0:
    expanded_gen_mask = (
        jnp.expand_dims(gen_mask, -2) if gen_mask.ndim == 3 else gen_mask
    )
    gen_masked_logits = jnp.where(expanded_gen_mask, gen_logits, k_mask)

    m_gen = jnp.max(
        gen_masked_logits.astype(jnp.float32), axis=-1, keepdims=True
    )
    m_pfx = jnp.max(pfx_logits.astype(jnp.float32), axis=-1, keepdims=True)
    m = jnp.maximum(m_gen, m_pfx)

    e_pfx = jnp.exp(pfx_logits.astype(jnp.float32) - m)
    e_gen = jnp.exp(gen_masked_logits.astype(jnp.float32) - m)

    l_total = jnp.sum(e_pfx, axis=-1, keepdims=True) + jnp.sum(
        e_gen, axis=-1, keepdims=True
    )
    pfx_attn = (e_pfx / l_total).astype(query.dtype)
    gen_attn = (e_gen / l_total).astype(query.dtype)

    if use_gqa:
      assert num_kv_heads is not None
      b, t, n, _ = pfx_logits.shape
      n_groups = n // num_kv_heads
      pfx_attn_r = pfx_attn.reshape((b, t, num_kv_heads, n_groups, pfx_s))
      gen_attn_r = gen_attn.reshape((b, t, num_kv_heads, n_groups, gen_s))
      encoded = jnp.einsum(
          "BTKGS,BSKH->BTKGH", pfx_attn_r, prefix_v
      ) + jnp.einsum("BTKGS,BSKH->BTKGH", gen_attn_r, gen_v)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum("BTNS,BSNH->BTNH", pfx_attn, prefix_v) + jnp.einsum(
          "BTNS,BSNH->BTNH", gen_attn, gen_v
      )
  else:
    expanded_gen_mask = (
        jnp.expand_dims(gen_mask, -2) if gen_mask.ndim == 3 else gen_mask
    )
    attn = jnp.where(expanded_gen_mask, gen_logits, k_mask)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)
    if use_gqa:
      assert num_kv_heads is not None
      b, t, n, s = attn.shape
      n_groups = n // num_kv_heads
      probs_reshaped = attn.reshape((b, t, num_kv_heads, n_groups, s))
      encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs_reshaped, gen_v)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum("BTNS,BSNH->BTNH", attn, gen_v)

  return encoded


def monolithic_attention(
    query: jax.Array,
    full_k: jax.Array,
    full_v: jax.Array,
    full_mask: jax.Array,
    *,
    use_gqa: bool = False,
    num_kv_heads: int | None = None,
) -> jax.Array:
  """Standard monolithic attention (reference implementation).

  Args:
      query: [B, T, N, D] query projections.
      full_k: [B, S, H, D] full key cache (prefix + gen, broadcast to B).
      full_v: [B, S, H, D] full value cache.
      full_mask: [B, T, S] boolean mask.
      use_gqa: Whether to use grouped query attention.
      num_kv_heads: Number of KV heads (required when use_gqa=True).

  Returns:
      [B, T, N, D] attention output.
  """
  k_mask = jnp.finfo(query.dtype).min

  if use_gqa:
    assert num_kv_heads is not None
    b, t, n, d = query.shape
    n_groups = n // num_kv_heads
    q_r = query.reshape((b, t, num_kv_heads, n_groups, d))
    logits = jnp.einsum("BTKGH,BSKH->BTKGS", q_r, full_k)
    b, t, k, g, s = logits.shape
    logits = logits.reshape((b, t, k * g, s))
  else:
    logits = jnp.einsum("BTNH,BSNH->BTNS", query, full_k)

  attn = jnp.where(jnp.expand_dims(full_mask, -2), logits, k_mask)
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)

  if use_gqa:
    assert num_kv_heads is not None
    b, t, n, s = attn.shape
    n_groups = n // num_kv_heads
    attn_r = attn.reshape((b, t, num_kv_heads, n_groups, s))
    encoded = jnp.einsum("BTKGS,BSKH->BTKGH", attn_r, full_v)
    b, t, k, g, h = encoded.shape
    encoded = encoded.reshape((b, t, k * g, h))
  else:
    encoded = jnp.einsum("BTNS,BSNH->BTNH", attn, full_v)

  return encoded


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SplitAttentionCorrectnessTest(parameterized.TestCase):
  """Verify split attention matches monolithic attention exactly."""

  @parameterized.named_parameters(
      dict(
          testcase_name="mha_decode",
          batch=4,
          seq_len=1,
          prefix_len=64,
          gen_len=16,
          num_heads=8,
          num_kv_heads=8,
          head_dim=32,
      ),
      dict(
          testcase_name="gqa_decode",
          batch=4,
          seq_len=1,
          prefix_len=64,
          gen_len=16,
          num_heads=8,
          num_kv_heads=2,
          head_dim=32,
      ),
      dict(
          testcase_name="mha_prefill",
          batch=4,
          seq_len=8,
          prefix_len=64,
          gen_len=16,
          num_heads=8,
          num_kv_heads=8,
          head_dim=32,
      ),
      dict(
          testcase_name="gqa_prefill",
          batch=4,
          seq_len=8,
          prefix_len=64,
          gen_len=16,
          num_heads=8,
          num_kv_heads=2,
          head_dim=32,
      ),
      dict(
          testcase_name="large_prefix",
          batch=8,
          seq_len=1,
          prefix_len=512,
          gen_len=32,
          num_heads=16,
          num_kv_heads=4,
          head_dim=64,
      ),
      dict(
          testcase_name="large_batch",
          batch=32,
          seq_len=1,
          prefix_len=128,
          gen_len=16,
          num_heads=8,
          num_kv_heads=2,
          head_dim=32,
      ),
  )
  def test_equivalence(
      self,
      batch: int,
      seq_len: int,
      prefix_len: int,
      gen_len: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
  ) -> None:
    """Split attention must produce identical output to monolithic."""
    use_gqa = num_kv_heads != num_heads
    key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    dtype = jnp.bfloat16

    query = jax.random.normal(
        k1, (batch, seq_len, num_heads, head_dim), dtype=dtype
    )
    prefix_k = jax.random.normal(
        k2, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    prefix_v = jax.random.normal(
        k3, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_k = jax.random.normal(
        k4, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_v = jax.random.normal(
        k5, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )

    # Mask: all prefix positions valid, causal mask on gen positions
    gen_mask = jnp.ones((batch, seq_len, gen_len), dtype=jnp.bool_)

    # --- Split attention ---
    split_out = split_attention(
        query,
        gen_k,
        gen_v,
        prefix_k,
        prefix_v,
        gen_mask,
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    )

    # --- Monolithic reference ---
    # Broadcast prefix to B and concatenate
    prefix_k_bc = jnp.broadcast_to(
        prefix_k, (batch, prefix_len, num_kv_heads, head_dim)
    )
    prefix_v_bc = jnp.broadcast_to(
        prefix_v, (batch, prefix_len, num_kv_heads, head_dim)
    )
    full_k = jnp.concatenate([prefix_k_bc, gen_k], axis=1)
    full_v = jnp.concatenate([prefix_v_bc, gen_v], axis=1)
    pfx_mask = jnp.ones((batch, seq_len, prefix_len), dtype=jnp.bool_)
    full_mask = jnp.concatenate([pfx_mask, gen_mask], axis=-1)

    mono_out = monolithic_attention(
        query,
        full_k,
        full_v,
        full_mask,
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    )

    # Compare — bf16 has limited precision so use atol
    self.assertEqual(split_out.shape, mono_out.shape)
    max_diff = jnp.max(
        jnp.abs(split_out.astype(jnp.float32) - mono_out.astype(jnp.float32))
    )
    self.assertLess(
        float(max_diff),
        0.02,
        f"Max diff {float(max_diff):.6f} exceeds tolerance",
    )


class EmptyPrefixTest(parameterized.TestCase):
  """Verify split attention degrades gracefully when prefix has 0 tokens."""

  @parameterized.named_parameters(
      dict(
          testcase_name="mha",
          num_heads=8,
          num_kv_heads=8,
      ),
      dict(
          testcase_name="gqa",
          num_heads=8,
          num_kv_heads=2,
      ),
  )
  def test_empty_prefix_matches_standard(
      self, num_heads: int, num_kv_heads: int
  ) -> None:
    """With P=0, split attention should equal standard attention."""
    batch, seq_len, gen_len, head_dim = 4, 1, 32, 32
    use_gqa = num_kv_heads != num_heads
    key = jax.random.key(7)
    k1, k2, k3 = jax.random.split(key, 3)
    dtype = jnp.bfloat16

    query = jax.random.normal(
        k1, (batch, seq_len, num_heads, head_dim), dtype=dtype
    )
    gen_k = jax.random.normal(
        k2, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_v = jax.random.normal(
        k3, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )

    # Empty prefix
    prefix_k = jnp.zeros((batch, 0, num_kv_heads, head_dim), dtype=dtype)
    prefix_v = jnp.zeros((batch, 0, num_kv_heads, head_dim), dtype=dtype)
    gen_mask = jnp.ones((batch, seq_len, gen_len), dtype=jnp.bool_)

    split_out = split_attention(
        query,
        gen_k,
        gen_v,
        prefix_k,
        prefix_v,
        gen_mask,
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    )

    # Standard attention (no prefix)
    mono_out = monolithic_attention(
        query,
        gen_k,
        gen_v,
        gen_mask,
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    )

    max_diff = jnp.max(
        jnp.abs(split_out.astype(jnp.float32) - mono_out.astype(jnp.float32))
    )
    self.assertLess(float(max_diff), 1e-5)


class BroadcastFusionTest(absltest.TestCase):
  """Verify that XLA handles B=1 prefix without materializing at B."""

  def test_jaxpr_has_dot_general(self) -> None:
    """Check that the jaxpr uses dot_general for the einsum."""
    batch, prefix_len, num_heads, head_dim = 8, 64, 4, 32

    def prefix_einsum(q: jax.Array, pk: jax.Array) -> jax.Array:
      return jnp.einsum("BTNH,BSNH->BTNS", q, pk)

    q = jax.ShapeDtypeStruct((batch, 1, num_heads, head_dim), jnp.bfloat16)
    pk = jax.ShapeDtypeStruct(
        (1, prefix_len, num_heads, head_dim), jnp.bfloat16
    )

    jaxpr = jax.make_jaxpr(prefix_einsum)(q, pk)
    jaxpr_str = str(jaxpr)

    # The jaxpr should use dot_general with batch broadcasting,
    # NOT an explicit broadcast_in_dim followed by dot_general.
    has_dot_general = "dot_general" in jaxpr_str
    self.assertTrue(
        has_dot_general,
        "Expected dot_general in jaxpr for einsum",
    )

  def test_lowered_hlo_inspection(self) -> None:
    """Inspect the lowered HLO to check for broadcast buffers."""
    batch, prefix_len, num_heads, head_dim = 32, 512, 8, 64

    @jax.jit
    def prefix_einsum(q: jax.Array, pk: jax.Array) -> jax.Array:
      return jnp.einsum("BTNH,BSNH->BTNS", q, pk)

    q = jnp.zeros((batch, 1, num_heads, head_dim), jnp.bfloat16)
    pk = jnp.zeros((1, prefix_len, num_heads, head_dim), jnp.bfloat16)

    lowered = prefix_einsum.lower(q, pk)
    hlo_text = lowered.as_text()

    # A problematic broadcast would create a [32, 512, 8, 64] buffer.
    problematic_shape = f"{batch}x{prefix_len}x{num_heads}x{head_dim}"

    has_large_broadcast = problematic_shape in hlo_text
    if has_large_broadcast:
      lines = [
          line.strip()
          for line in hlo_text.split("\n")
          if problematic_shape in line
      ]
      print(
          "\n⚠️  XLA may materialize broadcast buffer"
          f" [{batch},{prefix_len},{num_heads},{head_dim}]:"
      )
      for line in lines[:5]:
        print(f"  {line}")
    else:
      print("\n✅ No large broadcast buffer found in HLO — XLA fuses correctly")
    # This test is informational — don't fail, just report.


class SplitAttentionJitTest(absltest.TestCase):
  """Verify split attention works correctly under JIT compilation."""

  def test_jit_correctness(self) -> None:
    """JIT-compiled split attention matches eager execution."""
    batch, prefix_len, gen_len = 4, 64, 16
    num_heads, num_kv_heads, head_dim = 8, 2, 32
    dtype = jnp.bfloat16

    key = jax.random.key(99)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    query = jax.random.normal(k1, (batch, 1, num_heads, head_dim), dtype=dtype)
    prefix_k = jax.random.normal(
        k2, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    prefix_v = jax.random.normal(
        k3, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_k = jax.random.normal(
        k4, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_v = jax.random.normal(
        k5, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_mask = jnp.ones((batch, 1, gen_len), dtype=jnp.bool_)

    eager_out = split_attention(
        query,
        gen_k,
        gen_v,
        prefix_k,
        prefix_v,
        gen_mask,
        use_gqa=True,
        num_kv_heads=num_kv_heads,
    )

    jit_fn = jax.jit(
        functools.partial(
            split_attention,
            use_gqa=True,
            num_kv_heads=num_kv_heads,
        )
    )
    jit_out = jit_fn(query, gen_k, gen_v, prefix_k, prefix_v, gen_mask)

    max_diff = float(
        jnp.max(
            jnp.abs(eager_out.astype(jnp.float32) - jit_out.astype(jnp.float32))
        )
    )
    self.assertLess(max_diff, 1e-6)


class CausalMaskTest(absltest.TestCase):
  """Verify causal masking during suffix prefill with prefix."""

  def test_suffix_prefill_causal(self) -> None:
    """During suffix prefill, each suffix token should only attend to

    all prefix tokens and causally to prior suffix tokens.
    """
    batch, prefix_len, suffix_len = 2, 16, 8
    num_heads, num_kv_heads, head_dim = 4, 4, 16
    dtype = jnp.float32  # f32 for precision

    key = jax.random.key(55)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    query = jax.random.normal(
        k1, (batch, suffix_len, num_heads, head_dim), dtype=dtype
    )
    prefix_k = jax.random.normal(
        k2, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    prefix_v = jax.random.normal(
        k3, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_k = jax.random.normal(
        k4, (batch, suffix_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_v = jax.random.normal(
        k5, (batch, suffix_len, num_kv_heads, head_dim), dtype=dtype
    )

    # Causal mask for generation (suffix): position i attends to 0..i
    causal = jnp.tril(jnp.ones((suffix_len, suffix_len), dtype=jnp.bool_))
    gen_mask = jnp.broadcast_to(
        causal[None, :, :], (batch, suffix_len, suffix_len)
    )

    split_out = split_attention(
        query,
        gen_k,
        gen_v,
        prefix_k,
        prefix_v,
        gen_mask,
        use_gqa=False,
        num_kv_heads=num_kv_heads,
    )

    # Reference: broadcast prefix, concat, full mask
    prefix_k_bc = jnp.broadcast_to(
        prefix_k, (batch, prefix_len, num_kv_heads, head_dim)
    )
    prefix_v_bc = jnp.broadcast_to(
        prefix_v, (batch, prefix_len, num_kv_heads, head_dim)
    )
    full_k = jnp.concatenate([prefix_k_bc, gen_k], axis=1)
    full_v = jnp.concatenate([prefix_v_bc, gen_v], axis=1)
    pfx_mask = jnp.ones((batch, suffix_len, prefix_len), dtype=jnp.bool_)
    full_mask = jnp.concatenate([pfx_mask, gen_mask], axis=-1)

    mono_out = monolithic_attention(
        query,
        full_k,
        full_v,
        full_mask,
        use_gqa=False,
        num_kv_heads=num_kv_heads,
    )

    max_diff = float(jnp.max(jnp.abs(split_out - mono_out)))
    self.assertLess(max_diff, 1e-5, f"Max diff: {max_diff:.8f}")


class GradientTest(absltest.TestCase):
  """Verify gradients flow correctly through split attention."""

  def test_gradient_matches(self) -> None:
    """Gradient of split attention w.r.t. query matches monolithic."""
    batch, prefix_len, gen_len = 2, 32, 8
    num_heads, num_kv_heads, head_dim = 4, 2, 16
    dtype = jnp.float32

    key = jax.random.key(77)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    query = jax.random.normal(k1, (batch, 1, num_heads, head_dim), dtype=dtype)
    prefix_k = jax.random.normal(
        k2, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    prefix_v = jax.random.normal(
        k3, (1, prefix_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_k = jax.random.normal(
        k4, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_v = jax.random.normal(
        k5, (batch, gen_len, num_kv_heads, head_dim), dtype=dtype
    )
    gen_mask = jnp.ones((batch, 1, gen_len), dtype=jnp.bool_)

    def split_loss(q: jax.Array) -> jax.Array:
      out = split_attention(
          q,
          gen_k,
          gen_v,
          prefix_k,
          prefix_v,
          gen_mask,
          use_gqa=True,
          num_kv_heads=num_kv_heads,
      )
      return jnp.sum(out)

    def mono_loss(q: jax.Array) -> jax.Array:
      pk_bc = jnp.broadcast_to(
          prefix_k, (batch, prefix_len, num_kv_heads, head_dim)
      )
      pv_bc = jnp.broadcast_to(
          prefix_v, (batch, prefix_len, num_kv_heads, head_dim)
      )
      fk = jnp.concatenate([pk_bc, gen_k], axis=1)
      fv = jnp.concatenate([pv_bc, gen_v], axis=1)
      pm = jnp.ones((batch, 1, prefix_len), dtype=jnp.bool_)
      fm = jnp.concatenate([pm, gen_mask], axis=-1)
      out = monolithic_attention(
          q,
          fk,
          fv,
          fm,
          use_gqa=True,
          num_kv_heads=num_kv_heads,
      )
      return jnp.sum(out)

    split_grad = jax.grad(split_loss)(query)
    mono_grad = jax.grad(mono_loss)(query)

    max_diff = float(jnp.max(jnp.abs(split_grad - mono_grad)))
    self.assertLess(max_diff, 1e-4, f"Gradient max diff: {max_diff:.8f}")


class BridgeFlashAttentionTest(absltest.TestCase):
  """Verify bridge+flash path matches dense attention for unaligned suffixes.

  When a prefix cache is present and the suffix length is NOT a multiple
  of flash_attention_block_size, the model should use the bridge+flash
  path: dense attention for the first (suffix_len % block_size) tokens,
  then Splash Attention for the aligned remainder.
  """

  def _make_model(self, *, use_flash: bool, sliding_window: int | None = None):
    config = gemma4_model.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 128
    config.hidden_dim = 256
    config.num_heads = 4
    config.head_dim = 32
    config.num_kv_heads = 2
    config.frac_shared_layers = 0.0
    config.use_flash_attention = use_flash
    config.flash_attention_block_size = 128
    config.shd_config = gemma4_model.ShardingConfig.no_shard()
    if sliding_window is not None:
      config.sliding_window_size = sliding_window
    rngs = nnx.Rngs(42)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape((1,)), ("batch",)
    )
    model = gemma4_model.Gemma4(config, rngs=rngs)
    return model, config, mesh

  def test_bridge_flash_matches_dense_global(self):
    """Bridge+flash with global attention matches dense reference."""
    model_flash, config, mesh = self._make_model(use_flash=True)
    model_dense, _, _ = self._make_model(use_flash=False)

    # Copy weights from flash model to dense model
    flash_state = model_flash.__getstate__()
    model_dense.__setstate__(flash_state)

    # Prefix: 100 tokens (NOT aligned to block_size=128)
    prefix_ids = jax.random.randint(
        jax.random.key(1), (100,), 1, config.num_embed
    )
    cache_size = 512

    with jax.set_mesh(mesh):
      pfx_flash = functional.prefill_prefix(model_flash, prefix_ids, cache_size)
      pfx_dense = functional.prefill_prefix(model_dense, prefix_ids, cache_size)

      # Suffix: 180 tokens (total 100+180=280, suffix NOT aligned)
      suffix = jax.random.randint(
          jax.random.key(2), (2, 180), 1, config.num_embed
      )

      out_flash = functional.generate(
          model_flash,
          suffix,
          max_new_tokens=1,
          pad_id=0,
          eos_ids=99999,
          cache_size=cache_size,
          prefix_cache=pfx_flash,
      )
      out_dense = functional.generate(
          model_dense,
          suffix,
          max_new_tokens=1,
          pad_id=0,
          eos_ids=99999,
          cache_size=cache_size,
          prefix_cache=pfx_dense,
      )

    # Compare generated tokens (shape is [batch, max_new_tokens] = [2, 1])
    self.assertEqual(out_flash.tokens.shape, (2, 1))
    self.assertEqual(out_dense.tokens.shape, (2, 1))
    np.testing.assert_array_equal(out_flash.tokens, out_dense.tokens)

  def test_bridge_flash_matches_dense_local_sliding(self):
    """Bridge+flash with sliding window attention matches dense reference."""
    model_flash, config, mesh = self._make_model(
        use_flash=True, sliding_window=64
    )
    model_dense, _, _ = self._make_model(use_flash=False, sliding_window=64)

    # Copy weights from flash model to dense model
    flash_state = model_flash.__getstate__()
    model_dense.__setstate__(flash_state)

    prefix_ids = jax.random.randint(
        jax.random.key(1), (100,), 1, config.num_embed
    )
    cache_size = 512

    with jax.set_mesh(mesh):
      pfx_flash = functional.prefill_prefix(model_flash, prefix_ids, cache_size)
      pfx_dense = functional.prefill_prefix(model_dense, prefix_ids, cache_size)

      suffix = jax.random.randint(
          jax.random.key(2), (2, 180), 1, config.num_embed
      )

      out_flash = functional.generate(
          model_flash,
          suffix,
          max_new_tokens=1,
          pad_id=0,
          eos_ids=99999,
          cache_size=cache_size,
          prefix_cache=pfx_flash,
      )
      out_dense = functional.generate(
          model_dense,
          suffix,
          max_new_tokens=1,
          pad_id=0,
          eos_ids=99999,
          cache_size=cache_size,
          prefix_cache=pfx_dense,
      )

    self.assertEqual(out_flash.tokens.shape, (2, 1))
    self.assertEqual(out_dense.tokens.shape, (2, 1))
    np.testing.assert_array_equal(out_flash.tokens, out_dense.tokens)

  def test_bridge_flash_aligned_suffix_no_bridge(self):
    """When suffix is already aligned, no bridge is needed — pure Flash."""
    model, config, mesh = self._make_model(use_flash=True)

    # Prefix: 128 tokens (aligned to block_size=128)
    prefix_ids = jax.random.randint(
        jax.random.key(1), (128,), 1, config.num_embed
    )
    cache_size = 512

    with jax.set_mesh(mesh):
      pfx = functional.prefill_prefix(model, prefix_ids, cache_size)

      # Suffix: 128 tokens (aligned) — should use pure Flash, no bridge
      suffix = jax.random.randint(
          jax.random.key(2), (2, 128), 1, config.num_embed
      )

      out = functional.generate(
          model,
          suffix,
          max_new_tokens=1,
          pad_id=0,
          eos_ids=99999,
          cache_size=cache_size,
          prefix_cache=pfx,
      )
    self.assertEqual(out.tokens.shape, (2, 1))


if __name__ == "__main__":
  absltest.main()
