# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Time strict checkpoint startup and compare seeded, synthetic predictions.

Uses a cached checkpoint, never downloads assets. CPU is the default so this
can run while production GPUs are busy. Construction timings include device
transfer but exclude imports, checkpoint loading, hashing and prediction.
All modes run in one process; checkpoint/OS caches may warm across trials.
"""

import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch

import torch

from mol.config.inference import inference_configs
from mol.model.triangular import layers
from mol.runner.batch_inference import get_default_runner
from mol.utils.seed import seed_everything
from tests.test_input_batching import tiny_features


def digest(value):
    sha = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            sha.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            sha.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                sha.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                visit(element)
        else:
            sha.update(repr(item).encode())

    visit(value)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint must be an existing local checkpoint")
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "threads": args.threads,
        "checkpoint": str(args.checkpoint.resolve()),
        "timing_scope": (
            "Model construction/device transfer and checkpoint load; "
            "excludes imports and predictions"
        ),
        "prediction_settings": {
            "tokens": 5, "cycles": 1, "steps": 2, "samples": 1,
            "dtype": "fp32", "kernels": "torch", "seed": 101,
        },
        "trials": [],
    }
    original_draws = layers.truncnorm.rvs
    for repeat in range(args.repeats):
        for fast_init in ((False, True) if repeat % 2 == 0 else (True, False)):
            draws = {"calls": 0, "seconds": 0.0}

            def timed_draws(*a, **kw):
                started = time.perf_counter()
                result = original_draws(*a, **kw)
                draws["calls"] += 1
                draws["seconds"] += time.perf_counter() - started
                return result

            seed_everything(23, deterministic=False)
            print(f"Starting repeat={repeat}, fast_init={fast_init}", flush=True)
            with (
                patch.dict(inference_configs, {
                    "dump_dir": str(args.output.parent / "predictions")
                }),
                patch.object(layers.truncnorm, "rvs", timed_draws),
            ):
                runner = get_default_runner(
                    device=args.device, dtype="fp32", use_msa=False,
                    trimul_kernel="torch", triatt_kernel="torch",
                    n_cycle=1, n_step=2, n_sample=1,
                    load_checkpoint_path=str(args.checkpoint.resolve()),
                    memory_settings={"infer_setting.fast_init": fast_init},
                )
            trial = {
                "repeat": repeat, "fast_init": fast_init,
                "parameters": sum(p.numel() for p in runner.model.parameters()),
                "construction_seconds": runner.model_init_seconds,
                "checkpoint_seconds": runner.checkpoint_load_seconds,
                "checkpoint_coverage": runner.checkpoint_load_report,
                "scipy_draws": draws,
                "state_sha256": digest(runner.model.state_dict()),
            }
            seed_everything(101, deterministic=False)
            data = {"input_feature_dict": tiny_features(5, 101)}
            prediction = runner.predict(copy.deepcopy(data))
            trial["prediction_sha256"] = digest(prediction)
            report["trials"].append(trial)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(trial), flush=True)
            del runner, prediction, data
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    report["states_identical"] = len({
        trial["state_sha256"] for trial in report["trials"]
    }) == 1
    report["predictions_identical"] = len({
        trial["prediction_sha256"] for trial in report["trials"]
    }) == 1
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["states_identical"] and report["predictions_identical"], report


if __name__ == "__main__":
    main()
