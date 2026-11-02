# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Opt-in released-checkpoint test for native ipSAE on mixed-length batches."""

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("MOL_RUN_IPSAE_SMOKE") != "1",
    reason="requires cached checkpoint/assets and a free CUDA GPU",
)
def test_native_ipsae_checkpoint(tmp_path):
    import gemmi

    from mol.config.inference import inference_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.runner.batch_inference import get_default_runner
    from mol.runner.dumper import DataDumper
    from mol.utils.seed import seed_everything
    from tests.test_ipsae import reference_direction

    checkpoint = Path(os.environ["MOL_ROOT_DIR"]) / "checkpoint/mol-1-base.pt"
    assert checkpoint.is_file() and torch.cuda.is_available()
    jobs = []
    for lengths in ((64, 64), (96, 48)):
        jobs.append(
            {
                "name": f"synthetic_{lengths[0]}_{lengths[1]}",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": ("ACDEFGHIKLMNPQRSTVWY" * 6)[:length],
                            "count": 1,
                            "id": [chain],
                        }
                    }
                    for length, chain in zip(lengths, ("T", "B"))
                ],
            }
        )
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps(jobs))
    with patch.dict(inference_configs, {"dump_dir": str(tmp_path)}):
        runner = get_default_runner(
            batch_size=2,
            use_msa=False,
            n_cycle=2,
            n_step=8,
            n_sample=2,
            dtype="bf16",
            device="cuda",
            load_checkpoint_path=str(checkpoint),
        )
    runner.configs.input_json_path = str(inputs_path)
    dataset = InferenceDataset(runner.configs)
    inputs, atoms = [], []
    for index in range(2):
        seed_everything(101, deterministic=False)
        data, atom_array, error = dataset[index]
        assert not error, error
        inputs.append(data)
        atoms.append(atom_array)
    predictions = runner.predict_batch(inputs, [101, 101])
    for full in (False, True):
        dumper = DataDumper(str(tmp_path / str(full)), need_atom_confidence=full)
        for data, atom_array, prediction in zip(inputs, atoms, predictions):
            dumper.dump(
                "",
                data["sample_name"],
                101,
                copy.deepcopy(prediction),
                atom_array,
                data["entity_poly_type"],
            )
    for job, atom_array in zip(jobs, atoms):
        name = job["name"]
        short_dir = tmp_path / "False" / name / "seed_101/predictions"
        full_dir = tmp_path / "True" / name / "seed_101/predictions"
        assert len(list(short_dir.glob("*.cif"))) == 2
        assert not list(short_dir.glob("*full_data*"))
        for sample in range(2):
            filename = f"{name}_summary_confidence_sample_{sample}.json"
            short = json.loads((short_dir / filename).read_text())
            summary = json.loads((full_dir / filename).read_text())
            assert short == summary
            full = json.loads(
                (full_dir / f"{name}_full_data_sample_{sample}.json").read_text()
            )
            pae = np.asarray(full["token_pair_pae"], dtype=np.float32)
            mapping = np.asarray(full["atom_to_token_idx"])
            representatives = atom_array.atom_name == "CA"
            target = mapping[representatives & (atom_array.chain_id == "T")]
            binder = mapping[representatives & (atom_array.chain_id == "B")]
            for clip, suffix in ((27, ""), (26, "_d0ref")):
                expected = min(
                    reference_direction(pae, target, binder, clip),
                    reference_direction(pae, binder, target, clip),
                )
                assert summary["ipsae_min_10" + suffix] == pytest.approx(
                    expected, abs=1e-14, rel=0
                )
            cif = gemmi.cif.read_file(str(short_dir / f"{name}_sample_{sample}.cif"))
            assert len(cif.sole_block().find_values("_atom_site.id")) == len(atom_array)
    print(
        f"Verified four batched predictions, both output modes, native/file ipSAE parity: {tmp_path}"
    )
