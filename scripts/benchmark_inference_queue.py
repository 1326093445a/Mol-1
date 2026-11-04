# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Validate and time a fixed real-example queue at 5 samples / 10 cycles / 200 steps.

Uses local assets only. Batch screening is exploratory, not a production optimum.
The pipeline comparison includes worker startup, featurization and drained output.
It uses a loaded model and warm parent CCD caches; each spawned worker starts cold.
Prediction hashing adds the same validation work to all pipeline passes.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.metrics.ipsae import ipsae_summary, protein_token_groups
from mol.runner.batch_inference import get_default_runner
from mol.runner.batched_prediction import (
    _exceeds_batch_budget,
    infer_predict_batched,
    validate_prediction,
)
from mol.runner.inference import update_inference_configs
from mol.runner.inference_pipeline import prepared_jobs
from mol.utils.seed import seed_everything


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def features_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            features_equal(v, right[k]) for k, v in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            features_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-scheduler", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["all", "batching", "pipeline"], default="all"
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.baseline_scheduler.is_file():
        parser.error("Both checkpoint and baseline scheduler must exist locally")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    jobs = [
        json.loads((root / "examples/example.json").read_text())[0],
        json.loads((root / "examples/example_without_msa.json").read_text())[0],
        json.loads(
            (root / "examples/examples_with_template/example_9fm7.json").read_text()
        )[0],
        json.loads((root / "examples/example_without_msa.json").read_text())[1],
    ]
    seeds = [101, 103, 107, 109]
    for job, seed in zip(jobs, seeds):
        job["modelSeeds"] = [seed]
    inputs = out / "inputs.json"
    inputs.write_text(json.dumps(jobs, indent=2))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with patch.dict(
        inference_configs, {"dump_dir": str(out), "input_json_path": str(inputs)}
    ):
        runner = get_default_runner(
            batch_size=2,
            use_msa=False,
            n_cycle=10,
            n_step=200,
            n_sample=5,
            dtype="bf16",
            device="cuda",
            trimul_kernel="flashpairformer",
            triatt_kernel="cuequivariance",
            load_checkpoint_path=str(args.checkpoint.resolve()),
            need_atom_confidence=False,
            memory_settings={
                "infer_setting.sample_diffusion_chunk_size": 5,
                "infer_setting.confidence_sample_chunk_size": 5,
                "infer_setting.atom_block_cache": False,
            },
        )
    cfg = runner.configs
    cfg.input_json_path = str(inputs)
    cfg.use_template, cfg.use_seeds_in_json, cfg.deterministic = False, True, False
    dataset = InferenceDataset(cfg)
    prepared, atoms = [], []
    for index, seed in enumerate(seeds):
        seed_everything(seed, deterministic=False)
        data, atom_array, error = dataset[index]
        if error:
            raise RuntimeError(error)
        prepared.append(data)
        atoms.append(atom_array)
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "settings": {
            "samples": 5,
            "cycles": 10,
            "steps": 200,
            "sample_chunk": 5,
            "confidence_chunk": 5,
            "atom_block_cache": False,
            "tf32": False,
            "msa": False,
            "templates": False,
            "dtype": "bf16",
            "trimul": "flashpairformer",
            "triatt": "cuequivariance",
        },
        "inputs": [
            {
                "name": data["sample_name"],
                "tokens": int(data["N_token"]),
                "atoms": len(at),
                "seed": seed,
            }
            for data, at, seed in zip(prepared, atoms, seeds)
        ],
        "batching": [],
        "pipeline": [],
        "timing_scope": {
            "batching": "Prepared features, model forward only; B2 warmup, exploratory screen then best/B2 confirmation",
            "pipeline": "Loaded model, warm parent CCD cache; includes fresh worker startup, features, prediction hashes, and all writes",
            "baseline_scheduler_sha256": hashlib.sha256(
                args.baseline_scheduler.read_bytes()
            ).hexdigest(),
        },
    }

    def save():
        (out / "report.json").write_text(
            json.dumps(report, indent=2, default=json_value)
        )

    # Check the actual featurizer under spawn, including random reference conformers.
    with prepared_jobs(
        dataset, list(enumerate(seeds)), deterministic=False, prefetch=2
    ) as stream:
        for index, (_, data, atom_array, error) in enumerate(stream):
            assert not error, error
            assert features_equal(prepared[index], data), data["sample_name"]
            assert atoms[index] == atom_array, data["sample_name"]
    report["spawned_features_exact"] = True
    print(
        json.dumps({"inputs": report["inputs"], "spawned_features_exact": True}),
        flush=True,
    )
    save()

    def groups(size, ratio):
        batches, pending = [], []
        for index in sorted(
            range(len(prepared)), key=lambda i: int(prepared[i]["N_token"])
        ):
            if pending and _exceeds_batch_budget(
                [int(prepared[i]["N_token"]) for i in pending + [index]],
                {"batch_max_padding_ratio": ratio},
            ):
                batches.append(pending)
                pending = []
            pending.append(index)
            if len(pending) == size:
                batches.append(pending)
                pending = []
        if pending:
            batches.append(pending)
        return batches

    def batch_run(label, size, ratio):
        record = {
            "label": label,
            "batch_size": size,
            "padding_ratio": ratio,
            "groups": groups(size, ratio),
            "model_seconds": 0.0,
            "peak_gib": 0.0,
            "summaries": {},
        }
        for indices in record["groups"]:
            runner.update_model_configs(
                update_inference_configs(
                    copy.deepcopy(cfg),
                    max(int(prepared[i]["N_token"]) for i in indices),
                )
            )
            seed_everything(101, deterministic=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                predictions = runner.predict_batch(
                    [copy.deepcopy(prepared[i]) for i in indices],
                    [seeds[i] for i in indices],
                )
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                record["oom"] = True
                torch.cuda.empty_cache()
                break
            record["model_seconds"] += time.perf_counter() - start
            record["peak_gib"] = max(
                record["peak_gib"], torch.cuda.max_memory_allocated() / 2**30
            )
            for index, pred in zip(indices, predictions, strict=True):
                validate_prediction(
                    pred,
                    n_atom=len(atoms[index]),
                    n_token=int(prepared[index]["N_token"]),
                    n_sample=5,
                )
                protein_groups = protein_token_groups(
                    atoms[index],
                    pred["full_data"][0]["atom_to_token_idx"],
                    int(prepared[index]["N_token"]),
                )
                summaries = []
                for full, summary in zip(pred["full_data"], pred["summary_confidence"]):
                    summaries.append(
                        {
                            "ranking_score": summary["ranking_score"],
                            **ipsae_summary(full["token_pair_pae"], protein_groups),
                        }
                    )
                record["summaries"][prepared[index]["sample_name"]] = summaries
            del predictions
        report["batching"].append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k != "summaries"}),
            flush=True,
        )
        save()
        return record

    if args.phase in ("all", "batching"):
        batch_run("warm_B2", 2, 0)
        screens = [
            batch_run(label, size, ratio)
            for label, size, ratio in [
                ("screen_B1", 1, 0),
                ("screen_B2", 2, 0),
                ("screen_B4", 4, 0),
                ("screen_B4_ratio1.4", 4, 1.4),
            ]
        ]
        best = min(
            (r for r in screens if not r.get("oom")), key=lambda r: r["model_seconds"]
        )
        batch_run("confirm_best", best["batch_size"], best["padding_ratio"])
        batch_run("confirm_B2", 2, 0)
        report["screen_winner"] = {
            k: best[k] for k in ("label", "batch_size", "padding_ratio")
        }
        save()

    if args.phase in ("all", "pipeline"):
        spec = importlib.util.spec_from_file_location(
            "baseline_scheduler", args.baseline_scheduler
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        cfg.batch_size = 2
        cfg.infer_setting.batch_max_padding_ratio = 1.4
        original_predict = runner.predict_batch
        captures = {}

        def capture_predict(data_list, seed_list):
            predictions = original_predict(data_list, seed_list)
            for data, pred in zip(data_list, predictions, strict=True):
                captures[data["sample_name"]] = {
                    "coordinates": digest(pred["coordinate"]),
                    "pae": [digest(f["token_pair_pae"]) for f in pred["full_data"]],
                    "plddt": [digest(f["atom_plddt"]) for f in pred["full_data"]],
                }
            return predictions

        runner.predict_batch = capture_predict
        for label in ["serial_1", "overlap_1", "overlap_2", "serial_2"]:
            background = label.startswith("overlap")
            cfg.infer_setting.prefetch_inputs = 2 if background else 0
            cfg.infer_setting.background_write = background
            runner.dumper.base_dir = str(out / label)
            runner.error_dir = str(out / label / "ERR")
            captures.clear()
            seed_everything(101, deterministic=False)
            torch.cuda.synchronize()
            start = time.perf_counter()
            (infer_predict_batched if background else baseline.infer_predict_batched)(
                runner, cfg
            )
            torch.cuda.synchronize()
            record = {
                "label": label,
                "end_to_end_seconds": time.perf_counter() - start,
                "raw_prediction_hashes": copy.deepcopy(captures),
            }
            report["pipeline"].append(record)
            print(
                json.dumps(
                    {k: v for k, v in record.items() if k != "raw_prediction_hashes"}
                ),
                flush=True,
            )
            save()
        reference = report["pipeline"][0]["raw_prediction_hashes"]
        report["pipeline_raw_predictions_exact"] = all(
            r["raw_prediction_hashes"] == reference for r in report["pipeline"]
        )
        reference_files = {
            str(p.relative_to(out / "serial_1")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (out / "serial_1").rglob("*")
            if p.suffix in (".cif", ".json")
        }
        report["pipeline_file_count"] = len(reference_files)
        report["pipeline_output_files_exact"] = {}
        for label in ["overlap_1", "overlap_2", "serial_2"]:
            files = {
                str(p.relative_to(out / label)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (out / label).rglob("*")
                if p.suffix in (".cif", ".json")
            }
            report["pipeline_output_files_exact"][label] = files == reference_files
        report["pipeline_medians"] = {
            prefix: statistics.median(
                r["end_to_end_seconds"]
                for r in report["pipeline"]
                if r["label"].startswith(prefix)
            )
            for prefix in ("serial", "overlap")
        }
        save()
        assert report["pipeline_raw_predictions_exact"]
        assert reference_files and all(report["pipeline_output_files_exact"].values())


if __name__ == "__main__":
    main()
