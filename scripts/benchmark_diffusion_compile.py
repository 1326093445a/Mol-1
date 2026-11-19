# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Compare eager and compiled token diffusion with fixed inputs and seeds.

Default: tiny synthetic model on CPU; no downloads or GPU access. An explicitly
selected CUDA GPU must have no other compute processes unless sharing is enabled
with --allow-busy-gpu. A checkpoint benchmark
requires a local checkpoint and a JSON containing one equal-length input batch.
Timings exclude feature preparation, output comparison and file IO. First calls
are reported separately from alternating warmed passes. This is an experiment,
not a production CLI replacement or a claim of output equivalence.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import mol
from mol.config.inference import inference_configs
from mol.metrics.ipsae import ipsae_summary, protein_token_groups
from mol.model.compiled_diffusion import compiled_diffusion_blocks
from mol.model.mol import Mol
from mol.runner.batch_inference import get_default_runner
from mol.runner.inference import update_inference_configs
from mol.utils.seed import seed_everything


def require_unoccupied_gpu(*, allow_busy=False):
    """Record other GPU processes, requiring explicit permission to share."""
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise RuntimeError("CUDA runs require CUDA_VISIBLE_DEVICES=<one explicitly allocated GPU UUID>")
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    if uuid not in devices.stdout.splitlines():
        raise RuntimeError("Use the full UUID of the allocated GPU")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    conflicts = [
        row for row in result.stdout.splitlines()
        if row.split(",")[0].strip() == uuid
        and row.split(",")[1].strip() != str(os.getpid())
    ]
    if conflicts and not allow_busy:
        raise RuntimeError(f"Selected GPU has other compute processes: {conflicts}")
    return conflicts


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def tensor_digest(tensor):
    t = tensor.contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def differences(actual, expected):
    records = []
    for got, want in zip(actual, expected, strict=True):
        metrics = {}
        for key in ("coordinate", "pae", "pde", "plddt", "resolved", "token_pair_pae"):
            a, b = got[key], want[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise AssertionError(f"Changed output shape/dtype: {key}")
            error = a.double() - b.double()
            metrics[key] = {
                "exact": torch.equal(a, b),
                "max_abs": error.abs().max().item(),
                "rmse": error.square().mean().sqrt().item(),
            }
        metrics["ipsae_exact"] = got["ipsae"] == want["ipsae"]
        metrics["ranking_exact"] = got["ranking"] == want["ranking"]
        records.append(metrics)
    return records


def summarize(predictions, groups, interfaces):
    result = []
    for pred, chains, interface in zip(predictions, groups, interfaces, strict=True):
        data = {key: pred[key].detach().cpu().clone() for key in ("coordinate", "pae", "pde", "plddt", "resolved")}
        data["token_pair_pae"] = torch.stack([
            torch.as_tensor(full["token_pair_pae"]).detach().cpu()
            for full in pred["full_data"]
        ])
        if not all(torch.isfinite(t).all() for t in data.values()):
            raise AssertionError("Non-finite predictions")
        data["ipsae"] = [ipsae_summary(pae, chains, interface=interface) for pae in data["token_pair_pae"]]
        data["ranking"] = [float(s["ranking_score"]) for s in pred["summary_confidence"]]
        result.append(data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Explicitly permit competing GPU jobs; timings are provisional")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.4, help="Upper bound on this process's PyTorch CUDA allocator")
    parser.add_argument("--backend", choices=("inductor", "eager", "aot_eager"), default="inductor")
    parser.add_argument("--cuda-graphs", action="store_true", help="Compare eager, compiled, and compiled plus CUDA graph replay")
    parser.add_argument("--max-graphs", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--input-index", type=int, help="Select one job from the original example JSON without changing its contents")
    parser.add_argument("--seed", type=int, help="Use this one seed for the comparison, overriding the job's modelSeeds")
    parser.add_argument("--reference-fused-extension", type=Path, help="Optional original fused LayerNorm binary for an additional eager baseline before a stream-fix comparison")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-chunk", type=int, default=2)
    parser.add_argument("--confidence-chunk", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--trimul", choices=("torch", "flashpairformer", "cuequivariance"), default="torch")
    parser.add_argument("--triatt", choices=("torch", "cuequivariance"), default="torch")
    parser.add_argument("--atom-cache", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.tokens, args.batch_size, args.cycles,
           args.steps, args.samples, args.sample_chunk, args.confidence_chunk) < 1:
        parser.error("Counts must be positive")
    if not args.checkpoint and args.tokens < 2:
        parser.error("The synthetic two-chain fixture requires at least two tokens")
    if bool(args.checkpoint) != bool(args.input_json):
        parser.error("A checkpoint run needs both --checkpoint and --input-json")
    if not args.checkpoint and (args.input_index is not None or args.seed is not None):
        parser.error("--input-index and --seed require a checkpoint fixture")
    if args.reference_fused_extension and (not args.checkpoint or args.device != "cuda"):
        parser.error("--reference-fused-extension requires a CUDA checkpoint fixture")
    if args.device == "cpu" and args.dtype != "fp32":
        parser.error("CPU checks use FP32; BF16 validation requires CUDA")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")
    if args.max_graphs < 1:
        parser.error("--max-graphs must be positive")
    if args.cuda_graphs and (args.device != "cuda" or args.backend != "inductor"):
        parser.error("--cuda-graphs requires CUDA and the Inductor backend")
    if args.device == "cuda":
        require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    repo = Path(__file__).resolve().parents[1]
    if Path(mol.__file__).resolve().parent != repo / "mol":
        raise RuntimeError("Wrong Mol-1 import path; use PYTHONPATH pointing to the isolated checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(17, deterministic=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "mol_import": str(Path(mol.__file__).resolve()), "settings": settings,
        "fixture": "checkpoint" if args.checkpoint else "tiny_synthetic_random_weights",
        "layernorm_type": os.environ.get("LAYERNORM_TYPE", "fast_layernorm"),
        "timing_scope": "predict_batch only, including transfers/confidence, excluding feature preparation/comparison/output",
        "cuda_graphs": args.cuda_graphs, "dynamic_shapes": False, "fullgraph": True,
        "code_sha256": {
            str(path.relative_to(repo)): file_digest(path)
            for path in sorted(p for p in (repo / "mol").rglob("*") if p.suffix in (".py", ".cpp", ".cu", ".h")) + [Path(__file__).resolve()]
        },
        "passes": [],
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.checkpoint:
        from mol.data.inference.infer_dataloader import InferenceDataset

        if not args.checkpoint.is_file() or not args.input_json.is_file():
            parser.error("Checkpoint and input JSON must exist locally")
        jobs = json.loads(args.input_json.read_text())
        if not isinstance(jobs, list):
            parser.error("Input JSON must be a list of jobs")
        indices = list(range(len(jobs)))
        if args.input_index is not None:
            if not 0 <= args.input_index < len(jobs):
                parser.error("--input-index is outside the input job list")
            indices = [args.input_index]
            jobs = [jobs[args.input_index]]
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= args.batch_size:
            parser.error("Input JSON must contain one batch of jobs")
        with patch.dict(inference_configs, {"dump_dir": str(args.output / "predictions")}):
            runner = get_default_runner(
                device=args.device, dtype=args.dtype, use_msa=False,
                batch_size=max(2, args.batch_size), n_cycle=args.cycles,
                n_step=args.steps, n_sample=args.samples,
                trimul_kernel=args.trimul, triatt_kernel=args.triatt,
                load_checkpoint_path=str(args.checkpoint.resolve()),
                memory_settings={
                    "infer_setting.fast_init": True,
                    # This script owns its comparison context and mode switches.
                    "infer_setting.compile_diffusion": False,
                    "infer_setting.cuda_graphs": False,
                    "infer_setting.sample_diffusion_chunk_size": args.sample_chunk,
                    "infer_setting.confidence_sample_chunk_size": args.confidence_chunk,
                    "infer_setting.atom_block_cache": args.atom_cache,
                    "infer_setting.batch_max_padding_ratio": 1.0,
                },
            )
        runner.configs.input_json_path = str(args.input_json.resolve())
        dataset = InferenceDataset(runner.configs)
        prepared, groups, interfaces, seeds = [], [], [], []
        for i, job in zip(indices, jobs, strict=True):
            requested = [args.seed] if args.seed is not None else job.get("modelSeeds", [101 + 2 * i])
            if len(requested) != 1:
                parser.error("Each benchmark job needs exactly one modelSeed")
            seed = int(requested[0])
            seed_everything(seed, deterministic=False)
            data, atoms, error = dataset[i]
            if error:
                raise RuntimeError(error)
            prepared.append(data)
            groups.append(protein_token_groups(atoms, data["input_feature_dict"]["atom_to_token_idx"], int(data["N_token"])))
            interfaces.append(job.get("ipsae_interface"))
            seeds.append(seed)
        lengths = [int(d["N_token"]) for d in prepared]
        if len(set(lengths)) != 1:
            parser.error("Use equal-length inputs; this experiment adds no unequal-length padding")
        runner.update_model_configs(update_inference_configs(copy.deepcopy(runner.configs), lengths[0]))
        model = runner.model
        predict = lambda data: runner.predict_batch(data, seeds)
        report["checkpoint_load"] = runner.checkpoint_load_report
        report["checkpoint_sha256"] = file_digest(args.checkpoint)
        report["inputs_sha256"] = hashlib.sha256(args.input_json.read_bytes()).hexdigest()
        report["input_indices"] = indices
        report["job_names"] = [job.get("name") for job in jobs]
    else:
        if args.device != "cpu" or args.trimul != "torch" or args.triatt != "torch":
            parser.error("The tiny fixture is CPU-only; use the checkpoint fixture for CUDA validation")
        # Test helpers set LAYERNORM_TYPE for CPU fixtures. Never import them in
        # a real-checkpoint run: the selected production backend must stay intact.
        from tests.test_input_batching import activate_residuals, tiny_config, tiny_features

        cfg = tiny_config()
        cfg.model.N_cycle = args.cycles
        cfg.sample_diffusion.N_step = args.steps
        cfg.sample_diffusion.N_sample = args.samples
        cfg.infer_setting.sample_diffusion_chunk_size = args.sample_chunk
        cfg.infer_setting.confidence_sample_chunk_size = args.confidence_chunk
        cfg.infer_setting.atom_block_cache = args.atom_cache
        model = Mol(cfg).eval()
        activate_residuals(model)
        seeds = [101 + 2 * i for i in range(args.batch_size)]
        prepared = [tiny_features(args.tokens, seed) for seed in seeds]
        lengths = [args.tokens] * len(seeds)
        groups = [{"A": list(range(args.tokens // 2)), "B": list(range(args.tokens // 2, args.tokens))} for _ in seeds]
        interfaces = [None] * len(seeds)
        predict = lambda data: model.forward_batch(data, seeds)

    report["lengths"], report["seeds"] = lengths, seeds
    from mol.model.layer_norm.layer_norm import FusedLayerNorm
    from mol.model.triangular.layers import OpenFoldLayerNorm
    report["layernorm_modules"] = {
        "fused": sum(isinstance(module, FusedLayerNorm) for module in model.modules()),
        "torch": sum(isinstance(module, OpenFoldLayerNorm) for module in model.modules()),
    }
    reference_extension = None
    if args.device == "cuda" and report["layernorm_modules"]["fused"]:
        from mol.model.layer_norm import layer_norm as fused_module

        extension = fused_module._fast_layer_norm_extension()
        report["fused_extension"] = {
            "path": extension.__file__, "sha256": file_digest(Path(extension.__file__)),
        }
        if args.reference_fused_extension:
            path = args.reference_fused_extension.resolve()
            spec = importlib.util.spec_from_file_location(path.stem, path)
            reference_extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference_extension)
            report["reference_fused_extension"] = {"path": str(path), "sha256": file_digest(path)}
    elif args.reference_fused_extension:
        parser.error("--reference-fused-extension requires actual fused LayerNorm modules")
    report["effective_configs"] = model.configs.to_dict()
    # Configs include a few non-JSON scalar types in some environments.
    report["effective_configs"] = json.loads(json.dumps(report["effective_configs"], default=str))
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    reference = None
    previous_by_mode = {}

    def run(label, session, mode):
        nonlocal reference
        competing_processes = []
        if args.device == "cuda":
            competing_processes = require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
        session.enabled = mode in ("compiled", "graph")
        session.replay_enabled = mode == "graph"
        captures_before, replays_before = session.graph_captures, session.graph_replays
        # Clone and seed outside the timed region, preserving identical inputs.
        data = copy.deepcopy(prepared)
        seed_everything(101, deterministic=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = predict(data)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        record = {"label": label, "mode": mode, "compiled": session.enabled, "seconds": seconds,
                  "competing_gpu_processes_before": competing_processes}
        record["graph_captures"] = session.graph_captures - captures_before
        record["graph_replays"] = session.graph_replays - replays_before
        record["graph_fallbacks_total"] = session.graph_fallbacks
        if mode == "graph" and (not record["graph_replays"] or session.graph_fallbacks):
            raise AssertionError("Graph benchmark must replay successfully without capacity fallbacks")
        if args.device == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            record["competing_gpu_processes_after"] = require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
        result = summarize(predictions, groups, interfaces)
        del predictions
        if reference is None:
            reference = result
        record["against_eager"] = differences(result, reference)
        if mode in previous_by_mode:
            record["against_previous_same_mode"] = differences(result, previous_by_mode[mode])
        if mode == "graph":
            record["against_compiled"] = differences(result, previous_by_mode["compiled"])
        previous_by_mode[mode] = result
        record["hashes"] = [{k: tensor_digest(v) for k, v in item.items() if isinstance(v, torch.Tensor)} for item in result]
        record["ipsae"] = [item["ipsae"] for item in result]
        record["ranking"] = [item["ranking"] for item in result]
        report["passes"].append(record)
        report["session"] = asdict(session)
        save()
        print(json.dumps({"label": label, "seconds": seconds, "compiled_calls": session.compiled_calls,
                          "graph_captures": record["graph_captures"], "graph_replays": record["graph_replays"]}), flush=True)

    try:
        modes = ("eager", "compiled", "graph") if args.cuda_graphs else ("eager", "compiled")
        with torch.no_grad(), compiled_diffusion_blocks(
            model.diffusion_module.diffusion_transformer, backend=args.backend,
            cuda_graphs=args.cuda_graphs, max_graphs=args.max_graphs,
        ) as session:
            if reference_extension is not None:
                state = fused_module._extension_state
                with patch.object(fused_module, "_extension_state", (reference_extension, *state[1:])):
                    run("original_fused_eager", session, "original_fused_eager")
                report["eager_reference"] = "original_fused_eager"
            for mode in modes:
                run(f"{mode}_first", session, mode)
            for repeat in range(args.repeats):
                for mode in (modes if repeat % 2 == 0 else tuple(reversed(modes))):
                    run(f"warm_{mode}_{repeat}", session, mode)
        report["warm_medians_seconds"] = {
            mode: statistics.median(p["seconds"] for p in report["passes"] if p["label"].startswith(f"warm_{mode}_"))
            for mode in modes
        }
        report["status"] = "completed"
        report["shared_gpu_observed"] = any(
            p.get("competing_gpu_processes_before") or p.get("competing_gpu_processes_after")
            for p in report["passes"]
        )
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
