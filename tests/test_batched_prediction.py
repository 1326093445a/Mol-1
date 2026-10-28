# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Exercise partial batches, seeds, rank ownership and failure accounting."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner

from mol.runner import batch_inference, batched_prediction, inference
from tests.test_input_batching import tiny_config


def prediction(n_token, n_sample, marker):
    n_atom = n_token * 4
    return {
        "coordinate": torch.full((n_sample, n_atom, 3), float(marker)),
        "plddt": torch.zeros(n_sample, n_atom, 4),
        "resolved": torch.zeros(n_sample, n_atom, 2),
        "pae": torch.zeros(n_sample, n_token, n_token, 4),
        "pde": torch.zeros(n_sample, n_token, n_token, 4),
        "contact_probs": torch.zeros(n_token, n_token),
        "summary_confidence": [{} for _ in range(n_sample)],
        "full_data": [{} for _ in range(n_sample)],
    }


def run_scheduler(
    tmp_path,
    monkeypatch,
    *,
    count=3,
    rank=0,
    world=1,
    per_job_seeds=None,
    oom=False,
    featurization_failure=None,
    writer_failure=None,
    wrong_count=False,
):
    cfg = tiny_config()
    cfg.batch_size = 2
    cfg.input_json_path = "unused.json"
    cfg.use_seeds_in_json = per_job_seeds is not None
    calls, writes = [], []
    jobs = [
        {
            "name": f"job{i}",
            "sequences": [{"proteinChain": {"sequence": "A" * (i + 2)}}],
        }
        for i in range(count)
    ]
    if per_job_seeds:
        for job, seeds in zip(jobs, per_job_seeds):
            job["modelSeeds"] = seeds

    class Dataset:
        def __init__(self, configs):
            self.inputs = jobs

        def __getitem__(self, index):
            n_token = index + 2
            data = {
                "sample_name": f"job{index}",
                "N_token": torch.tensor([n_token]),
                "entity_poly_type": {"1": "polypeptide(L)"},
                "marker": index,
            }
            return (
                data,
                [None] * (n_token * 4),
                ("bad features" if index == featurization_failure else ""),
            )

    def predict(inputs, seeds):
        calls.append([(data["sample_name"], seed) for data, seed in zip(inputs, seeds)])
        if oom and len(inputs) > 1:
            raise torch.cuda.OutOfMemoryError("synthetic forward OOM")
        outputs = [
            prediction(int(data["N_token"].item()), 3, data["marker"] * 1000 + seed)
            for data, seed in zip(inputs, seeds)
        ]
        return outputs[:-1] if wrong_count else outputs

    def dump(**kwargs):
        if kwargs["pdb_id"] == writer_failure:
            raise OSError("writer failure")
        writes.append(
            (
                kwargs["pdb_id"],
                kwargs["seed"],
                kwargs["pred_dict"]["coordinate"][0, 0, 0].item(),
            )
        )

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "ERR"),
        update_model_configs=lambda c: None,
        predict_batch=predict,
        dumper=SimpleNamespace(dump=dump),
    )
    monkeypatch.setattr(batched_prediction, "InferenceDataset", Dataset)
    monkeypatch.setattr(
        batched_prediction, "DIST_WRAPPER", SimpleNamespace(rank=rank, world_size=world)
    )
    monkeypatch.setattr(inference, "fix_cterminal_carboxyl_oxygens", lambda x, atoms: x)
    if featurization_failure is not None or writer_failure is not None or wrong_count:
        with pytest.raises(RuntimeError, match="prediction"):
            inference.infer_predict(runner, cfg)
    else:
        inference.infer_predict(runner, cfg)
    return calls, writes


def test_partial_batch_keeps_every_input_and_seed(tmp_path, monkeypatch):
    calls, writes = run_scheduler(
        tmp_path, monkeypatch, per_job_seeds=[[7], [11], [11]]
    )
    assert calls == [[("job0", 7), ("job1", 11)], [("job2", 11)]]
    assert writes == [("job0", 7, 7), ("job1", 11, 1011), ("job2", 11, 2011)]


def test_multiple_seeds_produce_distinct_output_sets(tmp_path, monkeypatch):
    _, writes = run_scheduler(
        tmp_path, monkeypatch, per_job_seeds=[[7, 11], [11], [7, 11]]
    )
    assert {(name, seed) for name, seed, _ in writes} == {
        ("job0", 7),
        ("job0", 11),
        ("job1", 11),
        ("job2", 7),
        ("job2", 11),
    }
    assert len(writes) == 5


def test_oom_retries_smaller_batches_without_losing_inputs(tmp_path, monkeypatch):
    calls, writes = run_scheduler(tmp_path, monkeypatch, oom=True)
    assert [len(call) for call in calls] == [2, 1, 1, 1]
    assert [name for name, _, _ in writes] == ["job0", "job1", "job2"]


@pytest.mark.parametrize(
    "rank,expected", [(0, ["job0", "job2"]), (1, ["job1"]), (3, [])]
)
def test_rank_ownership_has_no_duplicate_padding(tmp_path, monkeypatch, rank, expected):
    _, writes = run_scheduler(
        tmp_path, monkeypatch, rank=rank, world=2 if rank < 2 else 4
    )
    assert [name for name, _, _ in writes] == expected


def test_feature_errors_fail_the_run_but_other_inputs_finish(tmp_path, monkeypatch):
    _, writes = run_scheduler(tmp_path, monkeypatch, featurization_failure=1)
    assert [name for name, _, _ in writes] == ["job0", "job2"]
    assert "bad features" in (tmp_path / "ERR/job1.txt").read_text()


def test_writer_failure_does_not_repeat_forward(tmp_path, monkeypatch):
    calls, writes = run_scheduler(tmp_path, monkeypatch, writer_failure="job1")
    assert [len(call) for call in calls] == [2, 1]
    assert [name for name, _, _ in writes] == ["job0", "job2"]


def test_wrong_output_count_is_a_failure_for_every_batch_member(tmp_path, monkeypatch):
    _, writes = run_scheduler(tmp_path, monkeypatch, wrong_count=True)
    assert not writes
    assert all((tmp_path / f"ERR/job{i}.txt").exists() for i in range(3))


def test_cli_exposes_independent_batch_size(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text("[]")
    with patch.object(batch_inference, "inference_jsons") as run:
        result = CliRunner().invoke(
            batch_inference.mol_cli,
            [
                "pred",
                "-i",
                str(path),
                "--batch_size",
                "2",
                "--use_msa",
                "false",
                "--sample",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["batch_size"] == 2
        assert run.call_args.kwargs["n_sample"] == 3
    result = CliRunner().invoke(
        batch_inference.mol_cli, ["pred", "-i", str(path), "--batch_size", "0"]
    )
    assert result.exit_code == 2


def test_unsupported_batch_mode_fails_before_asset_download():
    with patch.object(batch_inference, "download_inference_cache") as download:
        with pytest.raises(ValueError, match="use_msa"):
            batch_inference.get_default_runner(batch_size=2)
        download.assert_not_called()


def test_output_validation_rejects_confused_sample_axes():
    good = prediction(5, 3, 1)
    batched_prediction.validate_prediction(good, n_atom=20, n_token=5, n_sample=3)
    for key, value in (
        ("coordinate", torch.zeros(2, 3, 20, 3)),
        ("pae", torch.zeros(3, 3, 3, 4)),
        ("full_data", [{}]),
    ):
        with pytest.raises(ValueError):
            batched_prediction.validate_prediction(
                good | {key: value}, n_atom=20, n_token=5, n_sample=3
            )
