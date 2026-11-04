# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""The CPU pipeline preserves seeds, owns its buffers and propagates errors."""

import os
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mol.runner.inference_pipeline import PredictionWriter, prepared_jobs


class RandomDataset:
    def __getitem__(self, index):
        if index == 9:
            raise RuntimeError("feature worker failed")
        data = [index, random.random(), float(np.random.rand()), torch.rand(2)]
        return data, os.getpid(), "invalid input" if index == 2 else ""


def collect(prefetch):
    result = []
    with prepared_jobs(
        RandomDataset(),
        [(0, 101), (2, 103), (1, 101)],
        deterministic=False,
        prefetch=prefetch,
    ) as jobs:
        for seed, data, pid, error in jobs:
            result.append(
                (
                    seed,
                    data,
                    pid,
                    error,
                    random.random(),
                    float(np.random.rand()),
                    torch.rand(2),
                )
            )
    return result


def test_spawned_features_and_parent_rng_match_serial():
    serial, parallel = collect(0), collect(2)
    for expected, actual in zip(serial, parallel, strict=True):
        assert expected[0] == actual[0]
        assert expected[1][:3] == actual[1][:3]
        assert torch.equal(expected[1][3], actual[1][3])
        assert expected[2] == os.getpid() != actual[2]
        assert expected[3:6] == actual[3:6]
        assert torch.equal(expected[6], actual[6])


def test_worker_failure_is_not_silently_discarded():
    with pytest.raises(RuntimeError, match="feature worker failed"):
        with prepared_jobs(
            RandomDataset(), [(9, 101)], deterministic=False, prefetch=1
        ) as jobs:
            list(jobs)


def test_prefetch_queue_is_bounded_and_cancels_on_early_exit(monkeypatch):
    from concurrent.futures import Future

    from mol.runner import inference_pipeline

    futures, outstanding = [], []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, task):
            future = Future()
            futures.append(future)
            outstanding.append(task)
            if len(futures) == 1:
                future.set_result(fn(task))
            return future

    monkeypatch.setattr(inference_pipeline, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(inference_pipeline, "_worker_dataset", RandomDataset())
    monkeypatch.setattr(inference_pipeline, "_worker_deterministic", False)
    with prepared_jobs(
        RandomDataset(), [(0, i) for i in range(20)], deterministic=False, prefetch=2
    ) as jobs:
        assert len(outstanding) == 2
        next(jobs)
        assert len(outstanding) == 3  # One consumed, two pending.
    assert all(f.cancelled() for f in futures[1:])


def test_writer_owns_cpu_buffers_and_drains_before_exit():
    started, release = threading.Event(), threading.Event()
    writes = []

    def dump(**kwargs):
        started.set()
        assert release.wait(10)
        writes.append(kwargs)

    pred = {
        "coordinate": torch.ones(2, 3),
        "summary_confidence": [{"score": 2.0}],
        "full_data": [{"pae": torch.ones(2, 2)}],
        "unused_logits": object(),
    }
    atoms = np.array([7])
    with PredictionWriter(SimpleNamespace(dump=dump), None, background=True) as writer:
        writer.submit(pred_dict=pred, atom_array=atoms, pdb_id="real", seed=7)
        assert started.wait(10)
        # Submission returned while writing is in progress. Mutations cannot
        # affect the writer, and it never retains unused prediction buffers.
        pred["coordinate"].zero_()
        pred["full_data"][0]["pae"].zero_()
        pred["summary_confidence"][0]["score"] = -1
        atoms[0] = -1
        release.set()
    assert len(writes) == 1
    result = writes[0]
    assert set(result["pred_dict"]) == {"coordinate", "summary_confidence", "full_data"}
    assert torch.equal(result["pred_dict"]["coordinate"], torch.ones(2, 3))
    assert result["pred_dict"]["full_data"][0]["pae"].sum() == 4
    assert result["pred_dict"]["summary_confidence"][0]["score"] == 2
    assert result["atom_array"][0] == 7


def test_last_background_write_error_is_reported_on_main_thread():
    errors = []

    def dump(**kwargs):
        raise OSError("disk full")

    def report(*args):
        errors.append((threading.get_ident(), args))

    with PredictionWriter(
        SimpleNamespace(dump=dump), report, background=True
    ) as writer:
        writer.submit(
            pred_dict={
                "coordinate": torch.ones(1),
                "summary_confidence": [],
                "full_data": [],
            },
            atom_array=[],
            pdb_id="last",
            seed=7,
        )
    assert errors == [
        (
            threading.get_ident(),
            ({"sample_name": "last"}, 7, "Output writing failed: disk full"),
        )
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_background_writer_receives_only_detached_cpu_tensors():
    received = []

    def dump(**kwargs):
        pred = kwargs["pred_dict"]
        for tensor in (
            pred["coordinate"],
            pred["summary_confidence"][0]["score"],
            pred["full_data"][0]["pae"],
        ):
            assert tensor.device.type == "cpu"
            assert not tensor.requires_grad
        received.append(True)

    tensor = torch.ones(2, device="cuda", requires_grad=True)
    errors = []
    with PredictionWriter(
        SimpleNamespace(dump=dump), lambda *args: errors.append(args), background=True
    ) as writer:
        writer.submit(
            pred_dict={
                "coordinate": tensor,
                "summary_confidence": [{"score": tensor}],
                "full_data": [{"pae": tensor}],
            },
            atom_array=[],
            pdb_id="cuda",
            seed=7,
        )
    assert not errors
    assert received == [True]
