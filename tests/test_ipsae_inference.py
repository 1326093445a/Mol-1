# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Per-job interface selection reaches summaries through every scheduler."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from mol.data.inference import infer_dataloader
from mol.runner import context_parallel_prediction, inference
from mol.runner.dumper import DataDumper
from tests.test_batched_prediction import prediction
from tests.test_input_batching import tiny_config, tiny_features


@pytest.fixture
def interface_inputs(tmp_path, monkeypatch):
    # Stub only atomic feature construction, avoiding external CCD assets.
    # Use the real dataset, scheduler, writer and confidence serialization.
    class Features:
        entity_poly_type_and_seqs = {"entity_poly_type": {"1": "polypeptide(L)"}}

        def __init__(self, job):
            pass

        def get_feature_dict(self):
            feat = tiny_features(6, 101)
            feat["asym_id"] = torch.arange(3).repeat_interleave(2)
            atoms = AtomArray(24)
            atoms.chain_id = np.repeat(["A", "B", "C"], 8)
            atoms.atom_name = np.tile(["N", "CA", "C", "O"], 6)
            atoms.res_id = np.repeat(np.tile([1, 2], 3), 4)
            atoms.set_annotation("asym_id_int", np.repeat(np.arange(3), 8))
            atoms.set_annotation("is_protein", np.ones(24, dtype=bool))
            atoms.set_annotation("distogram_rep_atom_mask", atoms.atom_name == "CA")
            atoms.set_annotation("centre_atom_mask", atoms.atom_name == "CA")
            return feat, atoms, np.arange(6)

    monkeypatch.setattr(infer_dataloader, "SampleDictToFeatures", Features)
    jobs = [
        {"name": "grouped", "ipsae_interface": [["A"], ["B", "C"]]},
        {"name": "subset", "ipsae_interface": [["B"], ["C"]]},
        {"name": "default"},
    ]
    for job in jobs:
        job["sequences"] = [{"proteinChain": {"sequence": "AA", "count": 3}}]
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(jobs))
    cfg = tiny_config()
    cfg.input_json_path = str(path)
    cfg.dump_dir = str(tmp_path)
    cfg.num_workers = 0
    cfg.seeds = [101]
    cfg.use_seeds_in_json = False
    return cfg


@pytest.mark.parametrize("mode", ["ordinary", "batch", "background", "context"])
def test_interface_survives_each_inference_path(
    tmp_path, monkeypatch, interface_inputs, mode
):
    cfg = interface_inputs
    if mode in ("batch", "background"):
        cfg.batch_size = 2
        cfg.infer_setting.background_write = mode == "background"
    elif mode == "context":
        cfg.context_parallel_size = 2
        # Exercise rank-zero plumbing without starting distributed workers.
        monkeypatch.setattr(
            context_parallel_prediction, "rank_zero_action", lambda fn: fn()
        )
        monkeypatch.setattr(context_parallel_prediction.dist, "get_rank", lambda: 0)

    def predict(data):
        result = prediction(6, 3, 0)
        pae = torch.zeros(6, 6)
        pae[:2, 2:] = pae[2:, :2] = 20
        result["summary_confidence"] = [{"ranking_score": i / 10} for i in range(3)]
        result["full_data"] = [
            {
                "token_pair_pae": pae.clone(),
                "atom_to_token_idx": torch.arange(6).repeat_interleave(4),
            }
            for _ in range(3)
        ]
        return result

    dumper = DataDumper(str(tmp_path))
    monkeypatch.setattr(dumper, "_save_structure", Mock())
    monkeypatch.setattr(inference, "fix_cterminal_carboxyl_oxygens", lambda x, atoms: x)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "ERR"),
        update_model_configs=lambda config: None,
        predict=predict,
        predict_batch=lambda inputs, seeds: [predict(data) for data in inputs],
        last_context_parallel_stats={},
        dumper=dumper,
    )
    inference.infer_predict(runner, cfg)
    for name, expected in (("grouped", 0), ("subset", 1), ("default", None)):
        folder = tmp_path / name / "seed_101/predictions"
        files = list(folder.glob("*summary_confidence*.json"))
        assert len(files) == 3
        assert not list(folder.glob("*full_data*"))
        for path in files:
            summary = json.loads(path.read_text())
            assert summary["ipsae_min_10"] == expected
            assert summary["ipsae_min_10_d0ref"] == expected
            assert len(summary["ipsae_chain_pairs"]) == 3
            if expected is None:
                assert "ipsae_interface" not in summary
            else:
                selected = summary["ipsae_interface"]
                assert selected["chains_1"] == ["A" if name == "grouped" else "B"]
                assert selected["chains_2"] == (
                    ["B", "C"] if name == "grouped" else ["C"]
                )


def test_invalid_selection_fails_during_data_loading(monkeypatch, interface_inputs):
    dataset = infer_dataloader.InferenceDataset(interface_inputs)
    dataset.inputs[0]["ipsae_interface"] = [["A"], ["missing"]]
    template = Mock()
    monkeypatch.setattr(
        infer_dataloader.InferenceTemplateFeaturizer, "make_template_feature", template
    )
    data, atoms, error = dataset[0]
    assert data["sample_name"] == "grouped"
    assert atoms is None
    assert "ipsae_interface chain 'missing' has no protein CA tokens" in error
    template.assert_not_called()
