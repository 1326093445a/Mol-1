"""Attention layout, masking and broadcast-storage regressions."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from mol.model.modules.primitives import _attention, _attention_4d


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(6, 9, 16), (2, 5, 4, 9, 16), (2, 3, 4, 5, 9, 16)])
@pytest.mark.parametrize(
    "bias_kind", ["none", "shared", "full", "broadcast", "expanded"]
)
def test_layout_matches_attention_with_masks(device, shape, bias_kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(59)
    q = torch.randn(*shape, device=device) / 4
    kv_shape = (*shape[:-2], 13, shape[-1])
    k, v = [torch.randn(*kv_shape, device=device) for _ in range(2)]
    bias = None
    if bias_kind != "none":
        bias_shape = (*shape[:-1], 13)
        if bias_kind == "shared":
            bias_shape = (9, 13)
        elif bias_kind in ("broadcast", "expanded") and len(shape) > 3:
            bias_shape = (shape[0], *((1,) * (len(shape) - 4)), *bias_shape[-3:])
        # Noncontiguous mask, distinct per-head/input biases, and padded keys.
        bias = torch.randn(*bias_shape[:-1], 26, device=device)[..., ::2]
        bias[..., -3:] = -1e10
        if bias_kind == "expanded":
            bias = bias.expand(*shape[:-1], 13)
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        actual = _attention_4d(q, k, v, bias)
        if device == "cuda":
            torch.testing.assert_close(
                _attention(q, k, v, bias), actual, atol=0, rtol=0
            )
    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_shared_bias_storage_is_not_repeated_for_samples():
    q, k, v = [torch.randn(2, 5, 4, 9, 16) for _ in range(3)]
    bias = torch.randn(2, 1, 4, 9, 9)
    storage = bias.untyped_storage().data_ptr()
    calls = []
    original = F.scaled_dot_product_attention

    def record(q, k, v, *, attn_mask, **kwargs):
        assert q.ndim == k.ndim == v.ndim == 4
        assert attn_mask.untyped_storage().data_ptr() == storage
        calls.append(q.shape)
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    with torch.no_grad(), patch.object(F, "scaled_dot_product_attention", record):
        output = _attention_4d(q, k, v, bias)
    assert output.shape == q.shape
    assert len(calls) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grad_enabled_path_is_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k, v = [
        torch.randn(2, 3, 4, 7, 16, device=device, requires_grad=True) for _ in range(3)
    ]
    with patch(
        "mol.model.modules.primitives._attention_4d",
        side_effect=AssertionError("inference only"),
    ):
        result = _attention(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        result.sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autocast_path_is_unchanged():
    q, k, v = [torch.randn(2, 3, 4, 8, 16, device="cuda") for _ in range(3)]
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        patch(
            "mol.model.modules.primitives._attention_4d",
            side_effect=AssertionError("FP32 only"),
        ),
    ):
        result = _attention(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v, scale=1.0)
    torch.testing.assert_close(result, expected.float(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_broadcast_queries_and_empty_axes_keep_existing_path():
    with (
        torch.no_grad(),
        patch(
            "mol.model.modules.primitives._attention_4d",
            side_effect=AssertionError("native broadcasting"),
        ),
    ):
        for n_query in (0, 7):
            q = torch.randn(1, 3, 4, n_query, 16, device="cuda")
            k, v = [torch.randn(2, 3, 4, 8, 16, device="cuda") for _ in range(2)]
            result = _attention(q, k, v)
            expected = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_validation_uses_boolean_ca_mask():
    from types import SimpleNamespace

    import numpy as np

    from scripts.validate_attention_9fm7 import protein_ca_mask

    atoms = SimpleNamespace(
        is_protein=np.array([1, 1, 0, 1], dtype=np.int64),
        atom_name=np.array(["N", "CA", "CA", "CA"]),
    )
    mask = protein_ca_mask(atoms)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, False, True]


def test_validation_alignment_removes_rigid_motion_only():
    from scripts.validate_attention_9fm7 import aligned_rmsd

    x = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=torch.float64)
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    y = x @ rotation + 7
    assert aligned_rmsd(x, y) < 1e-12
    y[0, 0] += 1
    assert aligned_rmsd(x, y) > 0.1
