# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, local-checkpoint CUDA regression; never downloads model assets."""

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_BATCH_SMOKE") != "1",
    reason="set MOL_RUN_BATCH_SMOKE=1 with cached assets and a free CUDA GPU",
)
def test_released_checkpoint_batch_matches_seeded_singletons(tmp_path):
    import gemmi

    from mol.config.inference import inference_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.runner.batch_inference import get_default_runner
    from mol.runner.batched_prediction import validate_prediction
    from mol.runner.inference import (
        fix_cterminal_carboxyl_oxygens,
        update_inference_configs,
    )
    from mol.utils.seed import seed_everything

    root = Path(os.environ["MOL_ROOT_DIR"])
    checkpoint = root / "checkpoint/mol-1-base.pt"
    assert checkpoint.is_file(), "Checkpoint must be cached before running this test"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    assert torch.cuda.is_available(), (
        "This opt-in test requires an available CUDA device"
    )
    with patch.dict(inference_configs, {"dump_dir": str(tmp_path)}):
        runner = get_default_runner(
            batch_size=2,
            use_msa=False,
            n_cycle=2,
            n_step=8,
            n_sample=2,
            dtype="fp32",
            device="cuda",
            load_checkpoint_path=str(checkpoint),
            need_atom_confidence=True,
            memory_settings={"infer_setting.sample_diffusion_chunk_size": 2},
        )
    runner.configs.input_json_path = str(
        Path(__file__).resolve().parents[1] / "examples/no_msa_batch/inputs.json"
    )
    dataset = InferenceDataset(runner.configs)
    inputs, atoms, seeds = [], [], [101, 103, 107]
    for i, seed in enumerate(seeds):
        seed_everything(seed, deterministic=True)
        data, atom_array, error = dataset[i]
        assert not error, error
        data["input_feature_dict"]["inference_seed"] = torch.tensor(seed)
        inputs.append(data)
        atoms.append(atom_array)
    expected = [runner.predict(copy.deepcopy(data)) for data in inputs]
    repeated = runner.predict(copy.deepcopy(inputs[-1]))
    torch.testing.assert_close(
        repeated["coordinate"], expected[-1]["coordinate"], atol=0, rtol=0
    )
    print("Deterministic singleton repeat: bitwise identical coordinates")
    calls = []
    hook = runner.model.pairformer_stack.register_forward_pre_hook(
        lambda m, a, kw: calls.append(kw["s"].shape[0]), with_kwargs=True
    )
    try:
        observed = runner.predict_batch(copy.deepcopy(inputs[:2]), seeds[:2])
        observed += runner.predict_batch(copy.deepcopy(inputs[2:]), seeds[2:])
    finally:
        hook.remove()
    assert calls == [2, 2, 1, 1], (
        "Expected two batched recycles and a final partial batch"
    )
    for data, atom_array, seed, want, got in zip(
        inputs, atoms, seeds, expected, observed
    ):
        n_token = int(data["N_token"].item())
        validate_prediction(got, n_atom=len(atom_array), n_token=n_token, n_sample=2)
        # Different GEMM batch shapes accumulate FP32 rounding differently.
        # Bound both RMS and worst-case coordinate differences in angstroms;
        # relative errors on coordinates near zero are not a useful metric.
        error = got["coordinate"] - want["coordinate"]
        assert float(error.square().mean().sqrt()) < 1e-3
        torch.testing.assert_close(
            got["coordinate"], want["coordinate"], atol=1e-2, rtol=0
        )
        for key in ("plddt", "pae", "pde", "resolved", "contact_probs"):
            torch.testing.assert_close(got[key], want[key], atol=2e-3, rtol=2e-4)
        print(
            data["sample_name"],
            "max coordinate difference:",
            float(error.abs().max()),
            "coordinate RMSE:",
            float(error.square().mean().sqrt()),
        )
        got["coordinate"] = fix_cterminal_carboxyl_oxygens(
            got["coordinate"], atom_array
        )
        runner.dumper.dump(
            dataset_name="",
            pdb_id=data["sample_name"],
            seed=seed,
            pred_dict=got,
            atom_array=atom_array,
            entity_poly_type={
                k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"
            },
        )
        directory = tmp_path / data["sample_name"] / f"seed_{seed}" / "predictions"
        assert len(list(directory.glob("*.cif"))) == 2
        scores = []
        for sample in range(2):
            stem = data["sample_name"]
            cif = gemmi.cif.read_file(str(directory / f"{stem}_sample_{sample}.cif"))
            assert len(cif.sole_block().find_values("_atom_site.id")) == len(atom_array)
            full = json.loads(
                (directory / f"{stem}_full_data_sample_{sample}.json").read_text()
            )
            assert len(full["atom_plddt"]) == len(atom_array)
            assert len(full["token_pair_pae"]) == n_token
            assert all(len(row) == n_token for row in full["token_pair_pae"])
            summary = json.loads(
                (
                    directory / f"{stem}_summary_confidence_sample_{sample}.json"
                ).read_text()
            )
            scores.append(summary["ranking_score"])
        assert scores == sorted(scores, reverse=True)

    # Exercise the CLI's default precision with the scheduler's policy for
    # small inputs. BF16 is checked for valid outputs, not FP32 equality.
    runner.configs.dtype = "bf16"
    runner.update_model_configs(
        update_inference_configs(copy.deepcopy(runner.configs), 64)
    )
    mixed_precision = runner.predict_batch(copy.deepcopy(inputs[:2]), seeds[:2])
    for data, atom_array, got in zip(inputs[:2], atoms[:2], mixed_precision):
        validate_prediction(
            got, n_atom=len(atom_array), n_token=int(data["N_token"].item()), n_sample=2
        )
    print("BF16 batch: two mixed-length inputs, two samples each, finite outputs")
