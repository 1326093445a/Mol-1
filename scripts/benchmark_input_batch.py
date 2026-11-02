# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Warm mixed-length inference timing with five samples and ten recycles.

Requires cached reference data (MOL_ROOT_DIR) and a local checkpoint. Timings
include GPU transfer, inference and confidence summaries, but exclude model
loading, featurization, prediction validation and disk output.
"""

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import mol
from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.runner.batch_inference import get_default_runner
from mol.runner.inference import update_inference_configs
from mol.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 160])
    parser.add_argument(
        "--kernel",
        choices=["torch", "cuequivariance", "flashpairformer"],
        default="torch",
    )
    parser.add_argument(
        "--attention", choices=["torch", "cuequivariance"], default="torch"
    )
    parser.add_argument("--samples-per-call", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--confidence-chunk", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint must be a cached local file")
    if (
        args.repeats < 1
        or len(args.lengths) < 2
        or any(n < 2 or n % 2 for n in args.lengths)
    ):
        parser.error(
            "Need positive repeats and at least two positive even token lengths"
        )
    from mol.runner.batched_prediction import _precision_bucket

    if len({_precision_bucket(n) for n in args.lengths}) != 1:
        parser.error("Lengths must share the CLI's precision bucket")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output.with_suffix(".inputs.json")
    inputs_path.write_text(
        json.dumps(
            [
                {
                    "name": f"synthetic_{i}_{n}",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": ("ACDEFGHIKLMNPQRSTVWY" * ((n + 39) // 40))[
                                    : n // 2
                                ],
                                "count": 2,
                            }
                        }
                    ],
                }
                for i, n in enumerate(args.lengths)
            ]
        )
    )
    inference_configs["dump_dir"] = str(args.output.with_suffix(""))
    settings = {"infer_setting.sample_diffusion_chunk_size": args.samples_per_call}
    if args.confidence_chunk != 1:
        settings["infer_setting.confidence_sample_chunk_size"] = args.confidence_chunk
    runner = get_default_runner(
        batch_size=len(args.lengths),
        use_msa=False,
        n_cycle=10,
        n_step=200,
        n_sample=5,
        dtype="bf16",
        device="cuda",
        trimul_kernel=args.kernel,
        triatt_kernel=args.attention,
        load_checkpoint_path=str(args.checkpoint),
        memory_settings=settings,
    )
    runner.configs.input_json_path = str(inputs_path)
    runner.update_model_configs(
        update_inference_configs(runner.configs, max(args.lengths))
    )
    dataset = InferenceDataset(runner.configs)
    data, seeds = [], [101 + 2 * i for i in range(len(args.lengths))]
    for i, seed in enumerate(seeds):
        seed_everything(seed, deterministic=False)
        item, _, error = dataset[i]
        if error:
            raise RuntimeError(error)
        data.append(item)
    source_root = Path(mol.__file__).parent
    metadata = {
        "source": str(source_root),
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cycles": 10,
        "samples": 5,
        "steps": 200,
        "dtype": "bf16",
        "tf32": False,
        "skip_amp": dict(runner.configs.skip_amp),
        "lengths": args.lengths,
        "source_sha256": {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in (
                "model/batching.py",
                "model/modules/transformer.py",
                "model/modules/confidence.py",
            )
        },
    }
    records = []
    for run in range(args.repeats + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = runner.predict_batch(copy.deepcopy(data), seeds)
        torch.cuda.synchronize()
        record = {
            "run": run,
            "warmup": run == 0,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        for prediction in predictions:
            assert prediction["coordinate"].shape[0] == 5
            assert all(
                torch.isfinite(prediction[key]).all()
                for key in ("coordinate", "pae", "plddt")
            )
        if run == args.repeats and args.save_predictions:
            torch.save(
                [
                    {
                        k: v.cpu()
                        for k, v in pred.items()
                        if k in ("coordinate", "pae", "plddt")
                    }
                    for pred in predictions
                ],
                args.output.with_suffix(".pt"),
            )
        records.append(record)
        print(json.dumps(record), flush=True)
        del predictions
        args.output.write_text(
            json.dumps(
                {
                    **metadata,
                    "runs": records,
                    "median_seconds": statistics.median(
                        r["seconds"] for r in records if not r["warmup"]
                    )
                    if run
                    else None,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
