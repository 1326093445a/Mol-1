# Copyright (c) 2026 Aureka AI Research
# Modifications Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenDDE's tests/test_input_batching.py.
"""Mixed-length inputs must retain their own samples, atoms and confidence."""

import copy
import os

os.environ["LAYERNORM_TYPE"] = "torch"

import pytest
import torch
import torch.nn.functional as F

from mol.config.base import configs as base_configs
from mol.config.data import data_configs
from mol.config.inference import inference_configs, validate_input_batching
from mol.config.parser import parse_configs
from mol.data.utils import make_dummy_feature
from mol.model.batching import (
    pad_token_tensors,
    pairformer_batch,
    sample_diffusion_batch,
)
from mol.model.generator import InferenceNoiseScheduler
from mol.model.modules.pairformer import PairformerStack
from mol.model.mol import Mol
from mol.runner.batched_prediction import validate_prediction


def tiny_config():
    return parse_configs(
        copy.deepcopy({**base_configs, "data": data_configs, **inference_configs}),
        arg_str=" ".join(
            [
                "--use_msa false --use_template false --use_rna_msa false --dtype fp32",
                "--triangle_multiplicative torch --triangle_attention torch",
                "--c_s 16 --c_z 16 --c_z_pairmixer 16 --c_z_pairformer 12",
                "--c_token 16 --c_atom 16 --c_atompair 8 --c_s_inputs 81",
                "--n_blocks 2 --n_pairmixer_blocks 1 --hidden_scale_up false --model.N_cycle 2",
                "--model.pairformer.n_heads 4",
                "--model.msa_module.c_m 16 --model.msa_module.n_blocks 1",
                "--model.template_embedder.n_blocks 1 --model.template_embedder.c 16",
                "--model.diffusion_module.c_token 32",
                "--model.diffusion_module.atom_encoder.n_blocks 1",
                "--model.diffusion_module.transformer.n_blocks 2",
                "--model.diffusion_module.transformer.n_heads 4",
                "--model.diffusion_module.atom_decoder.n_blocks 1",
                "--model.confidence_head.n_blocks 1",
                "--sample_diffusion.N_sample 3 --sample_diffusion.N_step 4",
                "--infer_setting.chunk_size 2 --infer_setting.sample_diffusion_chunk_size 2",
            ]
        ),
        fill_required_with_null=True,
    )


def tiny_features(n_token, seed):
    """Two chains and four atoms per residue, without external CCD assets."""
    g = torch.Generator().manual_seed(seed)
    residue = torch.arange(n_token)
    parent = residue.repeat_interleave(4)
    slot = torch.arange(4).repeat(n_token)
    asym = (residue >= n_token // 2).long()
    feat = {
        "token_index": residue,
        "residue_index": residue,
        "asym_id": asym,
        "entity_id": asym,
        "sym_id": torch.zeros_like(residue),
        "restype": F.one_hot(residue % 20, 32).float(),
        "token_bonds": torch.zeros(n_token, n_token),
        "ref_pos": torch.randn(n_token * 4, 3, generator=g),
        "ref_charge": torch.zeros(n_token * 4),
        "ref_mask": torch.ones(n_token * 4),
        "ref_space_uid": parent,
        "ref_element": F.one_hot(torch.full((n_token * 4,), 6), 128),
        "ref_atom_name_chars": F.one_hot(
            torch.zeros(n_token * 4, 4, dtype=torch.long), 64
        ),
        "atom_to_token_idx": parent,
        "atom_to_tokatom_idx": slot,
        "has_frame": torch.ones(n_token, dtype=torch.long),
        "frame_atom_index": residue[:, None] * 4 + torch.arange(3)[None, :],
        "distogram_rep_atom_mask": slot == 1,
        "pae_rep_atom_mask": slot == 1,
        "is_protein": torch.ones(n_token * 4, dtype=torch.long),
        "is_ligand": torch.zeros(n_token * 4, dtype=torch.long),
        "is_dna": torch.zeros(n_token * 4, dtype=torch.long),
        "is_rna": torch.zeros(n_token * 4, dtype=torch.long),
        "inference_seed": torch.tensor(seed),
    }
    return make_dummy_feature(feat, dummy_feats=("msa", "template"))


def activate_residuals(model):
    # Zero-initialized output projections would hide missing attention masks.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1 and not parameter.count_nonzero():
                parameter.normal_(std=0.02)


def test_padding_only_changes_named_token_axes():
    values = [torch.randn(3, n, n, 3) for n in (2, 5, 3)]
    padded = pad_token_tensors(values, pair=True)
    assert padded.shape == (3, 3, 5, 5, 3)
    for i, value in enumerate(values):
        n = value.shape[-2]
        torch.testing.assert_close(padded[i, :, :n, :n], value, atol=0, rtol=0)
        assert not padded[i, :, n:].any()
        assert not padded[i, :, :, n:].any()


@pytest.mark.parametrize("lengths", [(3, 3), (2, 5), (1, 7)])
def test_mixed_width_trunk_matches_singletons(lengths):
    torch.manual_seed(31)
    model = PairformerStack(
        n_blocks=2,
        n_pairmixer_blocks=1,
        c_s=16,
        c_z=16,
        c_z_pairmixer=16,
        c_z_pairformer=12,
        n_heads=2,
    ).eval()
    activate_residuals(model)
    singles = [torch.randn(n, 16) for n in lengths]
    pairs = [torch.randn(n, n, 16) for n in lengths]
    with torch.no_grad():
        expected = [
            model(
                s.clone(),
                z.clone(),
                pair_mask=None,
                inplace_safe=True,
                triangle_multiplicative="torch",
                triangle_attention="torch",
                chunk_size=2,
            )
            for s, z in zip(singles, pairs)
        ]
        observed = pairformer_batch(model, singles, pairs, chunk_size=2)
    for s, z, (want_s, want_z) in zip(*observed, expected):
        torch.testing.assert_close(s, want_s, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(z, want_z, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("chunk", [None, 1, 2, 5])
def test_sampler_streams_do_not_depend_on_batch_membership(chunk):
    states = [
        ({"atom_to_token_idx": torch.zeros(n, dtype=torch.long)}, torch.zeros(2, 4))
        for n in (3, 7, 3)
    ]
    schedules = [InferenceNoiseScheduler()(N_step=3) for _ in states]
    seeds = [101, 102, 103]
    settings = dict(
        n_sample=3,
        chunk_size=chunk,
        gamma0=0.8,
        gamma_min=1,
        noise_scale_lambda=1.003,
        step_scale_eta=1.5,
    )
    denoise = lambda xs, ts: [x * 0.5 for x in xs]
    actual = sample_diffusion_batch(
        states, denoise, schedules=schedules, seeds=seeds, **settings
    )
    for i in (2, 0, 1):
        single = sample_diffusion_batch(
            [states[i]], denoise, schedules=[schedules[i]], seeds=[seeds[i]], **settings
        )[0]
        torch.testing.assert_close(actual[i], single, atol=0, rtol=0)


@pytest.mark.parametrize(
    "cache,fusion", [("full", False), ("full", True), ("shared", True), ("off", False)]
)
def test_full_model_preserves_inputs_samples_and_confidence(cache, fusion):
    torch.manual_seed(42)
    cfg = tiny_config()
    cfg.diffusion_cache_mode = cache
    cfg.enable_efficient_fusion = fusion
    model = Mol(cfg).eval()
    activate_residuals(model)
    seeds = [101, 102, 103]
    features = [tiny_features(n, seed) for n, seed in zip((3, 5, 5), seeds)]
    with torch.no_grad():
        expected = [
            model(copy.deepcopy(feat), None, None, mode="inference")[0]
            for feat in features
        ]
        trunks, denoisers = [], []
        trunk_hook = model.pairformer_stack.register_forward_pre_hook(
            lambda m, a, kw: trunks.append(kw["s"].shape[0]), with_kwargs=True
        )
        diffusion_hook = (
            model.diffusion_module.diffusion_transformer.register_forward_pre_hook(
                lambda m, a, kw: denoisers.append(kw["a"].shape[:2]), with_kwargs=True
            )
        )
        try:
            actual = model.forward_batch(copy.deepcopy(features), seeds)
        finally:
            trunk_hook.remove()
            diffusion_hook.remove()
    assert trunks == [3, 3]
    assert denoisers and set(denoisers) == {(3, 2), (3, 1)}
    for observed, wanted, feat in zip(actual, expected, features):
        validate_prediction(
            observed,
            n_atom=len(feat["ref_pos"]),
            n_token=len(feat["token_index"]),
            n_sample=3,
        )
        for key in ("coordinate", "plddt", "pae", "pde", "resolved", "contact_probs"):
            torch.testing.assert_close(observed[key], wanted[key], atol=3e-4, rtol=3e-4)
        for obs, want in zip(
            observed["summary_confidence"], wanted["summary_confidence"]
        ):
            for key in ("plddt", "ptm", "iptm", "ranking_score"):
                torch.testing.assert_close(obs[key], want[key], atol=3e-4, rtol=3e-4)
        for full in observed["full_data"]:
            torch.testing.assert_close(
                full["atom_to_token_idx"], feat["atom_to_token_idx"]
            )
            torch.testing.assert_close(full["token_asym_id"], feat["asym_id"])


def test_batch_configuration_rejects_unsupported_modes():
    cfg = tiny_config()
    validate_input_batching(cfg, batch_size=2)
    for key, value in (
        ("use_msa", True),
        ("use_template", True),
        ("use_rna_msa", True),
        ("triangle_attention", "cuequivariance"),
        ("num_workers", 2),
        ("mc_dropout_apply_rate", 0.4),
    ):
        changed = copy.deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError, match=key):
            validate_input_batching(changed, batch_size=2)
    with pytest.raises(ValueError, match="positive"):
        validate_input_batching(cfg, batch_size=0)


def test_triangle_chunking_does_not_materialize_cubic_broadcast(monkeypatch):
    from mol.model.triangular import triangular

    torch.manual_seed(23)
    layer = triangular.TriangleAttention(c_in=8, c_hidden=4, no_heads=2).eval()
    activate_residuals(layer)
    x = torch.randn(2, 7, 7, 8)
    with torch.no_grad():
        expected = layer(x, chunk_size=None)
        shapes = []
        hook = layer.mha.register_forward_pre_hook(
            lambda m, a, kw: shapes.append((kw["q_x"].shape, kw["biases"][1].shape)),
            with_kwargs=True,
        )

        def forbid_flattening(*args, **kwargs):
            raise AssertionError("Batched triangle bias must not expand/flatten")

        monkeypatch.setattr(triangular, "chunk_layer", forbid_flattening)
        try:
            actual = layer(x, chunk_size=3, inplace_safe=True)
        finally:
            hook.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert [shape[0][-3] for shape in shapes] == [3, 3, 1]
    assert all(shape[1] == (2, 1, 2, 7, 7) for shape in shapes)


def test_batched_triangle_chunking_preserves_gradients():
    from mol.model.triangular.triangular import TriangleAttention

    torch.manual_seed(23)
    layer = TriangleAttention(c_in=8, c_hidden=4, no_heads=2)
    activate_residuals(layer)
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 5, 5, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    layer(x, chunk_size=2).square().sum().backward()
    reference(y, chunk_size=None).square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)
    for got, want in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(got.grad, want.grad, atol=2e-5, rtol=2e-5)
