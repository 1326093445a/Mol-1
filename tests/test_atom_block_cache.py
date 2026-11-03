"""Request-local atom caches preserve arithmetic, gradients and sample chunking."""

import copy
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch

from mol.model.modules.transformer import AtomTransformer
from mol.model.mol import Mol
from tests.test_input_batching import activate_residuals, tiny_config, tiny_features


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk", [None, 3])
def test_atom_cache_is_exact_and_immutable(device, chunk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(53)
    model = (
        AtomTransformer(
            c_atom=32, c_atompair=16, n_blocks=2, n_heads=4, n_queries=4, n_keys=8
        )
        .to(device)
        .eval()
    )
    activate_residuals(model)
    q = torch.randn(5, 55, 32, device=device)
    c = torch.randn(1, 55, 32, device=device)
    p = torch.randn(1, 14, 4, 8, 16, device=device)
    with torch.no_grad():
        cache = model.prepare_inference_cache(c, p)
        saved = []
        for block in cache:
            for field in fields(block):
                value = getattr(block, field.name)
                values = value if isinstance(value, tuple) else (value,)
                saved.extend((t, t.clone()) for t in values)
        expected = model(q, c, p, chunk_size=chunk)
        actual = model(q, c, p, chunk_size=chunk, inference_cache=cache)
        assert torch.equal(actual, expected)
        changed_q = q + 0.7
        assert torch.equal(
            model(changed_q, c, p, chunk_size=chunk, inference_cache=cache),
            model(changed_q, c, p, chunk_size=chunk),
        )
        for current, snapshot in saved:
            assert torch.equal(current, snapshot)
        assert all("cache" not in name for name in model.state_dict())


def test_atom_cache_is_ignored_with_gradients_and_training():
    torch.manual_seed(71)
    model = AtomTransformer(
        c_atom=16, c_atompair=8, n_blocks=1, n_heads=2, n_queries=4, n_keys=8
    ).eval()
    activate_residuals(model)
    q, c, p = torch.randn(2, 7, 16), torch.randn(1, 7, 16), torch.randn(1, 2, 4, 8, 8)
    with torch.no_grad():
        cache = model.prepare_inference_cache(c, p)
    model.train()
    c, p = c.requires_grad_(), p.requires_grad_()
    assert model.prepare_inference_cache(c, p) is None
    actual = model(q, c, p, inference_cache=cache)
    expected = model(q, c, p)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert c.grad is not None and p.grad is not None
    assert torch.isfinite(c.grad).all() and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("mode", ["full", "shared"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_complete_batch_matches_uncached_and_builds_once_per_request(mode, chunk):
    torch.manual_seed(11)
    cfg = tiny_config()
    cfg.diffusion_cache_mode = mode
    cfg.infer_setting.atom_block_cache = True
    cfg.sample_diffusion.N_sample = 3
    cfg.infer_setting.sample_diffusion_chunk_size = chunk
    model = Mol(cfg).eval()
    activate_residuals(model)
    inputs = [tiny_features(3, 101), tiny_features(5, 103)]
    original = AtomTransformer.prepare_inference_cache
    calls = []

    def capture(self, c, p):
        calls.append((id(self), c.shape, p.shape))
        return original(self, c, p)

    with torch.no_grad():
        with patch.object(
            AtomTransformer, "prepare_inference_cache", return_value=None
        ):
            expected = model.forward_batch(copy.deepcopy(inputs), [101, 103])
        with patch.object(AtomTransformer, "prepare_inference_cache", capture):
            actual = model.forward_batch(copy.deepcopy(inputs), [101, 103])
    assert len(calls) == 4  # encoder + decoder, once for each independent input
    for got, want in zip(actual, expected):
        for name in ["coordinate", "pae", "pde", "plddt", "resolved"]:
            assert torch.equal(got[name], want[name]), name


@pytest.mark.parametrize("enabled,mode", [(False, "full"), (True, "off")])
def test_atom_cache_is_opt_in_and_respects_cache_off(enabled, mode):
    cfg = tiny_config()
    cfg.diffusion_cache_mode = mode
    cfg.infer_setting.atom_block_cache = enabled
    model = Mol(cfg).eval()
    with (
        torch.no_grad(),
        patch.object(
            AtomTransformer, "prepare_inference_cache", side_effect=AssertionError
        ),
    ):
        model.forward_batch([tiny_features(3, 101), tiny_features(5, 103)], [101, 103])
