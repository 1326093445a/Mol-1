"""Inference reuse, sample chunking, and fused mixed-length padding regressions."""

import copy
import os
from unittest.mock import patch

import pytest
import torch

from tests.test_input_batching import activate_residuals, tiny_config, tiny_features
from mol.config.inference import validate_input_batching
from mol.model import batching
from mol.model.modules.pairformer import PairformerStack
from mol.model.mol import Mol


@pytest.mark.parametrize("confidence_chunk", [2, 3, None])
def test_confidence_chunks_preserve_sample_axis_and_scores(confidence_chunk):
    torch.manual_seed(71)
    cfg = tiny_config()
    model = Mol(cfg).eval()
    activate_residuals(model)
    features = [tiny_features(3, 101), tiny_features(5, 103)]
    with torch.inference_mode():
        expected = model.forward_batch(copy.deepcopy(features), [101, 103])
        cfg.infer_setting.confidence_sample_chunk_size = confidence_chunk
        actual = model.forward_batch(copy.deepcopy(features), [101, 103])
    for got, want in zip(actual, expected):
        for key in ("coordinate", "pae", "pde", "plddt", "resolved"):
            torch.testing.assert_close(got[key], want[key], atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("cache_mode", ["full", "shared"])
def test_pair_masks_and_conditioning_are_reused_without_mutation(cache_mode):
    cfg = tiny_config()
    cfg.diffusion_cache_mode = cache_mode
    cfg.model.N_cycle = 10
    model = Mol(cfg).eval()
    activate_residuals(model)
    caches = []
    original = model.prepare_inference_cache
    encoder = model.diffusion_module.atom_attention_encoder
    prepare_conditioned = encoder.prepare_conditioned_cache

    def cache(*args, **kwargs):
        result = original(*args, **kwargs)
        tensors = [result["pair_z"], *result["p_lm/c_l"]]
        caches.extend((t, t.clone()) for t in tensors)
        return result

    def conditioned(*args, **kwargs):
        result = prepare_conditioned(*args, **kwargs)
        caches.extend((t, t.clone()) for t in result)
        return result

    with (
        torch.inference_mode(),
        patch.object(model, "prepare_inference_cache", cache),
        patch.object(batching, "token_masks", wraps=batching.token_masks) as masks,
        patch.object(
            encoder, "prepare_conditioned_cache", side_effect=conditioned
        ) as atoms,
    ):
        model.forward_batch([tiny_features(3, 101), tiny_features(5, 103)], [101, 103])
        # One mask for all ten recycles, one for all diffusion steps/chunks.
        assert masks.call_count == 2
        assert atoms.call_count == 2
        for tensor, saved in caches:
            torch.testing.assert_close(tensor, saved, atol=0, rtol=0)


def test_batching_accepts_fused_backends():
    cfg = tiny_config()
    cfg.triangle_attention = "cuequivariance"
    for backend in ("auto", "torch", "cuequivariance", "flashpairformer"):
        cfg.triangle_multiplicative = backend
        validate_input_batching(cfg, batch_size=4)


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_BATCH_KERNEL_TESTS") != "1",
    reason="opt-in CUDA/cuEquivariance/opt_core regression",
)
@pytest.mark.parametrize("backend", ["cuequivariance", "flashpairformer"])
@pytest.mark.parametrize("width", [128, 320, 448])
def test_fused_batched_trunk_matches_masked_singletons(backend, width):
    assert torch.cuda.is_available()
    torch.manual_seed(71)
    model = (
        PairformerStack(
            n_blocks=2,
            n_pairmixer_blocks=1,
            c_s=64,
            c_z=width,
            c_z_pairmixer=width,
            c_z_pairformer=width,
            n_heads=4,
        )
        .cuda()
        .eval()
    )
    activate_residuals(model)
    lengths = [111, 129]
    singles = [torch.randn(n, 64, device="cuda") for n in lengths]
    pairs = [torch.randn(n, n, width, device="cuda") for n in lengths]
    kernels = dict(triangle_multiplicative=backend, triangle_attention="cuequivariance")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Use identically padded singleton calls to isolate B-axis dispatch
        # from the fused kernel's length-dependent accumulation order.
        s = batching.pad_token_tensors(singles)
        z = batching.pad_token_tensors(pairs, pair=True)
        mask, bias = batching.token_masks(lengths, z)
        expected = [
            model(
                s[i : i + 1].clone(),
                z[i : i + 1].clone(),
                pair_mask=mask[i : i + 1],
                extra_attn_bias=bias[i : i + 1],
                inplace_safe=True,
                chunk_size=32,
                **kernels,
            )
            for i in range(2)
        ]
        actual = batching.pairformer_batch(
            model, singles, pairs, chunk_size=32, **kernels
        )
    for i, (got_s, got_z) in enumerate(zip(*actual)):
        n = lengths[i]
        assert torch.isfinite(got_s).all() and torch.isfinite(got_z).all()
        for got, want in (
            (got_s, expected[i][0][0, :n]),
            (got_z, expected[i][1][0, :n, :n]),
        ):
            # BF16 GEMM batch shapes change accumulation/rounding. Bound the
            # RMS as well as pointwise error; use FP32 below to check masks.
            assert (got.float() - want.float()).square().mean().sqrt() < 0.01
            torch.testing.assert_close(got, want, atol=0.0625, rtol=0.03125)


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_BATCH_KERNEL_TESTS") != "1", reason="opt-in CUDA regression"
)
@pytest.mark.parametrize("lengths", [(7, 13), (7, 23), (23, 37), (111, 129)])
def test_fused_fp32_padding_matches_unpadded_torch(lengths):
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        PairformerStack(
            n_blocks=2,
            n_pairmixer_blocks=1,
            c_s=64,
            c_z=128,
            c_z_pairmixer=128,
            c_z_pairformer=128,
            n_heads=4,
        )
        .cuda()
        .eval()
    )
    activate_residuals(model)
    singles = [torch.randn(n, 64, device="cuda") for n in lengths]
    pairs = [torch.randn(n, n, 128, device="cuda") for n in lengths]
    with torch.inference_mode():
        expected = [
            model(
                s.clone(),
                z.clone(),
                pair_mask=None,
                inplace_safe=True,
                chunk_size=8,
                triangle_multiplicative="torch",
                triangle_attention="torch",
            )
            for s, z in zip(singles, pairs)
        ]
        actual = batching.pairformer_batch(
            model,
            singles,
            pairs,
            chunk_size=8,
            triangle_multiplicative="cuequivariance",
            triangle_attention="cuequivariance",
        )
    for got_s, got_z, (want_s, want_z) in zip(*actual, expected):
        torch.testing.assert_close(got_s, want_s, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(got_z, want_z, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_BATCH_KERNEL_TESTS") != "1", reason="opt-in CUDA regression"
)
@pytest.mark.parametrize("cache_mode", ["full", "shared", "off"])
def test_bf16_diffusion_cache_policy_is_finite(cache_mode):
    torch.manual_seed(71)
    cfg = tiny_config()
    cfg.skip_amp.sample_diffusion = False
    cfg.skip_amp.confidence_head = False
    cfg.diffusion_cache_mode = cache_mode
    cfg.infer_setting.confidence_sample_chunk_size = 3
    model = Mol(cfg).cuda().eval()
    activate_residuals(model)
    features = [
        {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in tiny_features(n, seed).items()
        }
        for n, seed in ((3, 101), (5, 103))
    ]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = model.forward_batch(features, [101, 103])
    for pred in predictions:
        for key in ("coordinate", "pae", "pde", "plddt", "resolved"):
            assert torch.isfinite(pred[key]).all(), (cache_mode, key)
