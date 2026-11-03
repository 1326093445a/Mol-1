"""Stable atom-to-token inference reductions and unchanged training behavior."""

from unittest.mock import patch

import pytest
import torch

from mol.model.utils import (
    _aggregate_atom_to_token_grouped,
    aggregate_atom_to_token,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("prefix", [(), (1,), (5,), (2, 3)])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_grouped_reduction_handles_unsorted_indices_and_empty_tokens(
    device, prefix, reduce
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(97)
    x = torch.randn(*prefix, 7, 8, device=device)
    index = torch.tensor([2, 0, 2, 0, 3, 2, 3], device=device)
    expected = torch.zeros(*prefix, 6, 8, device=device)
    for token in [0, 2, 3]:
        values = x[..., index == token, :]
        expected[..., token, :] = (
            values.mean(-2) if reduce == "mean" else values.sum(-2)
        )
    with torch.no_grad():
        actual = _aggregate_atom_to_token_grouped(x, index, 6, reduce)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ordinary_inference_is_repeatable_without_global_flag_changes(dtype):
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(71)
        index = torch.arange(644, device="cuda").repeat_interleave(8)
        x = torch.randn(5, len(index), 128, dtype=dtype, device="cuda")
        with torch.no_grad():
            expected = aggregate_atom_to_token(x, index, 644)
            for _ in range(4):
                assert torch.equal(aggregate_atom_to_token(x, index, 644), expected)
        assert not torch.are_deterministic_algorithms_enabled()
        assert expected.dtype == dtype
    finally:
        torch.use_deterministic_algorithms(old)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients_keep_scatter_path(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(7, 4, device=device, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 2], device=device)
    with patch(
        "mol.model.utils._aggregate_atom_to_token_grouped", side_effect=AssertionError
    ):
        aggregate_atom_to_token(x, index, 3).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_grouped_reduction_rejects_out_of_range_token():
    with pytest.raises(RuntimeError, match="exceeds n_token"):
        _aggregate_atom_to_token_grouped(
            torch.ones(2, 3), torch.tensor([0, 2]), 2, "mean"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_deterministic_mode_keeps_the_existing_path():
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x = torch.randn(12, 8, device="cuda")
        index = torch.arange(3, device="cuda").repeat_interleave(4)
        with (
            torch.no_grad(),
            patch(
                "mol.model.utils._aggregate_atom_to_token_grouped",
                side_effect=AssertionError,
            ),
        ):
            actual = aggregate_atom_to_token(x, index, 3)
        torch.testing.assert_close(actual, x.reshape(3, 4, 8).mean(1))
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(old)
