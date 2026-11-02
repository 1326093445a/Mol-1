# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Real collectives: one complex partitioned across two and three CPU workers."""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LAYERNORM_TYPE"] = "torch"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mol.config.inference import validate_context_parallel
from mol.model.context_parallel import active_context, context_parallel
from mol.model.modules.pairformer import PairformerStack
from mol.model.mol import Mol
from mol.utils.seed import seed_everything
from tests.test_input_batching import activate_residuals, tiny_config, tiny_features
from tests.test_batched_prediction import prediction


def _check_shared_scheduler(rank):
    from mol.runner import context_parallel_prediction as scheduler

    config = tiny_config()
    config.use_seeds_in_json = True
    calls, writes = [], []

    class Dataset:
        def __init__(self, configs):
            assert rank == 0, "Only the output rank should featurize"
            self.inputs = [
                {"name": "a", "modelSeeds": [7, 9]},
                {"name": "b", "modelSeeds": [11]},
            ]

        def __getitem__(self, index):
            return (
                {
                    "sample_name": self.inputs[index]["name"],
                    "N_token": torch.tensor(5),
                    "input_feature_dict": {},
                    "entity_poly_type": {"1": "polypeptide(L)"},
                },
                [None] * 20,
                "",
            )

    def predict(data):
        seed = int(data["input_feature_dict"]["inference_seed"])
        calls.append((data["sample_name"], seed))
        return prediction(5, 3, seed)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        update_model_configs=lambda config: None,
        predict=predict,
        last_context_parallel_stats={},
        dumper=SimpleNamespace(
            dump=lambda **kwargs: writes.append((kwargs["pdb_id"], kwargs["seed"]))
        ),
    )
    with (
        patch.object(scheduler, "InferenceDataset", Dataset),
        patch(
            "mol.runner.inference.fix_cterminal_carboxyl_oxygens",
            lambda coordinates, atoms: coordinates,
        ),
    ):
        scheduler.infer_predict_context_parallel(runner, config)
    expected = [("a", 7), ("a", 9), ("b", 11)]
    assert calls == expected  # every rank processes every input, not a job shard
    assert writes == (expected if rank == 0 else [])

    def fail():
        raise ValueError("injected input/output failure")

    with pytest.raises(RuntimeError, match="injected input/output failure"):
        scheduler.rank_zero_action(fail)


def _worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=90),
    )
    try:
        with torch.no_grad():
            # Exercise transposes, width adapters, masked pairs and empty ranks.
            for n in (1, 5):
                seed_everything(41, deterministic=True)
                stack = PairformerStack(
                    n_blocks=3,
                    n_pairmixer_blocks=1,
                    c_s=16,
                    c_z=16,
                    c_z_pairmixer=16,
                    c_z_pairformer=12,
                    n_heads=4,
                ).eval()
                activate_residuals(stack)
                s, z = torch.randn(n, 16), torch.randn(n, n, 16)
                mask = (torch.rand(n, n) > 0.2).float()
                bias = torch.randn(n, n) * 0.1
                args = dict(pair_mask=mask, chunk_size=2, extra_attn_bias=bias)
                expected = stack(s.clone(), z.clone(), **args)
                with context_parallel() as ctx:
                    observed = stack(s.clone(), z.clone(), **args)
                for want, got in zip(expected, observed):
                    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
                assert ctx.stats["last_pair_rows"] == (n + size - 1) // size
                assert active_context() is None

            # Full molecule: atom mappings, multiple samples, confidence, caches,
            # and efficient fusion must all survive the partition boundaries.
            for cache_mode, fusion in (
                ("full", True),
                ("shared", False),
                ("off", True),
            ):
                seed_everything(19, deterministic=True)
                config = tiny_config()
                config.diffusion_cache_mode = cache_mode
                config.enable_diffusion_shared_vars_cache = cache_mode != "off"
                config.enable_efficient_fusion = fusion
                model = Mol(config).eval()
                activate_residuals(model)
                features = tiny_features(5, 97)
                expected = model(
                    copy.deepcopy(features),
                    None,
                    None,
                    mode="inference",
                    mc_dropout_apply_rate=0,
                )[0]
                with context_parallel() as ctx:
                    observed = model(
                        copy.deepcopy(features),
                        None,
                        None,
                        mode="inference",
                        mc_dropout_apply_rate=0,
                    )[0]
                for key in (
                    "coordinate",
                    "plddt",
                    "pae",
                    "pde",
                    "resolved",
                    "contact_probs",
                ):
                    torch.testing.assert_close(
                        observed[key], expected[key], atol=3e-4, rtol=3e-4
                    )
                assert (
                    ctx.stats["pairformer_calls"]
                    >= 2 + config.sample_diffusion.N_sample
                )
                assert ctx.stats["diffusion_calls"] == 8  # 4 steps x 2 sample chunks
                assert ctx.stats["last_diffusion_queries"] == (5 + size - 1) // size

            # One rank's allocation/computation error must reach every peer.
            with context_parallel() as ctx:

                def fail():
                    if rank == 1:
                        raise RuntimeError("injected peer failure")

                with pytest.raises(RuntimeError, match="injected peer failure"):
                    ctx.local(fail)
            _check_shared_scheduler(rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 3])
def test_context_parallel_matches_singleton(tmp_path, size):
    mp.spawn(_worker, args=(size, str(tmp_path / "rendezvous")), nprocs=size, join=True)


def test_context_parallel_config_and_cli(monkeypatch):
    from click.testing import CliRunner
    from mol.runner.batch_inference import predict

    config = tiny_config()
    config.context_parallel_size = 2
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        validate_context_parallel(config)
    monkeypatch.setenv("WORLD_SIZE", "2")
    validate_context_parallel(config)
    for key, value in (
        ("batch_size", 2),
        ("use_msa", True),
        ("use_template", True),
        ("dtype", "fp16"),
        ("device", "cuda:0"),
    ):
        altered = copy.deepcopy(config)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_context_parallel(altered)
    result = CliRunner().invoke(predict, ["--help"])
    assert result.exit_code == 0
    assert "--context_parallel_size" in result.output


def test_context_parallel_rejects_autograd():
    with pytest.raises(ValueError, match="inference-only"):
        with context_parallel():
            pass
