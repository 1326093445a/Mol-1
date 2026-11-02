# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Opt-in checkpoint test; launch this test with torchrun on two or more GPUs."""

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_CP_SMOKE") != "1",
    reason="requires torchrun, cached checkpoint, and free CUDA GPUs",
)
def test_context_parallel_checkpoint(tmp_path):
    import gemmi

    from mol.config.inference import inference_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.runner.batch_inference import get_default_runner
    from mol.runner.batched_prediction import validate_prediction
    from mol.runner.context_parallel_prediction import rank_zero_action
    from mol.runner.inference import infer_predict, update_inference_configs
    from mol.utils.seed import seed_everything

    size = int(os.environ["WORLD_SIZE"])
    assert size >= 2 and torch.cuda.is_available()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    checkpoint = Path(os.environ["MOL_ROOT_DIR"]) / "checkpoint/mol-1-base.pt"
    assert checkpoint.is_file()
    with patch.dict(inference_configs, {"dump_dir": str(tmp_path)}):
        runner = get_default_runner(
            context_parallel_size=size,
            use_msa=False,
            dtype="fp32",
            device="cuda",
            n_cycle=2,
            n_step=8,
            n_sample=2,
            need_atom_confidence=True,
            load_checkpoint_path=str(checkpoint),
            memory_settings={"infer_setting.sample_diffusion_chunk_size": 2},
        )
    try:
        runner.configs.input_json_path = str(
            Path(__file__).resolve().parents[1] / "examples/no_msa_batch/inputs.json"
        )
        runner.configs.deterministic = True
        seed_everything(101, deterministic=True)

        def featurize():
            data, atoms, error = InferenceDataset(runner.configs)[1]
            assert not error, error
            data["input_feature_dict"]["inference_seed"] = torch.tensor(101)
            return data

        data = rank_zero_action(featurize)
        runner.configs.context_parallel_size = 1
        expected = runner.predict(copy.deepcopy(data))
        runner.configs.context_parallel_size = size
        observed = runner.predict(copy.deepcopy(data))
        error = observed["coordinate"] - expected["coordinate"]
        print(
            f"rank {dist.get_rank()} FP32 coordinate max={float(error.abs().max()):.8f} A RMSE={float(error.square().mean().sqrt()):.8f} A; {runner.last_context_parallel_stats}",
            flush=True,
        )
        assert float(error.square().mean().sqrt()) < 0.002
        torch.testing.assert_close(
            observed["coordinate"], expected["coordinate"], atol=0.02, rtol=0
        )
        for key in ("plddt", "pae", "pde", "resolved", "contact_probs"):
            torch.testing.assert_close(
                observed[key], expected[key], atol=0.003, rtol=0.0003
            )
        # Empty-template features still run the checkpoint's template stacks.
        template_calls = 2 * data["input_feature_dict"]["template_aatype"].shape[0]
        assert (
            runner.last_context_parallel_stats["pairformer_calls"] == 4 + template_calls
        )
        assert runner.last_context_parallel_stats["diffusion_calls"] == 8
        assert (
            runner.last_context_parallel_stats["last_diffusion_queries"]
            == (64 + size - 1) // size
        )
        reference = observed["coordinate"].clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(observed["coordinate"], reference, atol=0, rtol=0)

        runner.configs.dtype = "bf16"
        runner.update_model_configs(
            update_inference_configs(copy.deepcopy(runner.configs), 64)
        )
        observed = runner.predict(copy.deepcopy(data))
        validate_prediction(
            observed, n_atom=int(data["N_atom"].item()), n_token=64, n_sample=2
        )
        print(
            f"rank {dist.get_rank()} BF16: finite outputs and correct axes", flush=True
        )

        # Exercise the actual shared-job scheduler and rank-zero-only output.
        runner.configs.dtype = "fp32"
        infer_predict(runner, runner.configs)

        def check_outputs():
            assert len(list(tmp_path.rglob("*.cif"))) == 6
            for n, n_atom in ((32, 262), (64, 530), (96, 796)):
                name = f"synthetic_{n}"
                directory = tmp_path / name / "seed_101/predictions"
                scores = []
                for sample in range(2):
                    cif = gemmi.cif.read_file(
                        str(directory / f"{name}_sample_{sample}.cif")
                    )
                    assert len(cif.sole_block().find_values("_atom_site.id")) == n_atom
                    full = json.loads(
                        (
                            directory / f"{name}_full_data_sample_{sample}.json"
                        ).read_text()
                    )
                    assert len(full["atom_plddt"]) == n_atom
                    assert len(full["token_pair_pae"]) == n
                    assert all(len(row) == n for row in full["token_pair_pae"])
                    summary = json.loads(
                        (
                            directory
                            / f"{name}_summary_confidence_sample_{sample}.json"
                        ).read_text()
                    )
                    scores.append(summary["ranking_score"])
                assert scores == sorted(scores, reverse=True)

        rank_zero_action(check_outputs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
