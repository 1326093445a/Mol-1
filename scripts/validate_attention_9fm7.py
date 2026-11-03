# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Compare baseline/current attention on real examples with fixed inference settings."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.metrics.ipsae import ipsae_summary, protein_token_groups
from mol.model.modules import primitives
from mol.runner.batch_inference import get_default_runner
from mol.runner.batched_prediction import validate_prediction
from mol.runner.inference import (
    fix_cterminal_carboxyl_oxygens,
    update_inference_configs,
)
from mol.utils.seed import seed_everything


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def protein_ca_mask(atom_array):
    # is_protein is an integer annotation, so explicitly make a boolean mask.
    return torch.from_numpy(
        np.asarray(atom_array.is_protein, dtype=bool) & (atom_array.atom_name == "CA")
    )


def aligned_rmsd(x, y):
    x, y = x.double(), y.double()
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vh = torch.linalg.svd(yc.T @ xc)
    sign = torch.ones(3, dtype=torch.float64)
    sign[-1] = torch.linalg.det(u @ vh)
    return float(((yc @ ((u * sign) @ vh) - xc).square().sum(-1).mean()).sqrt())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["single", "batch"], required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--baseline-source",
        type=Path,
        required=True,
        help="Unmodified primitives.py from the comparison revision.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Cached mol-1-base checkpoint; never downloaded by this script.",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.baseline_source.is_file() or not a.checkpoint.is_file():
        p.error("--baseline-source and --checkpoint must be existing local files")
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(__file__).resolve().parents[1]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(a.deterministic)
    base_spec = importlib.util.spec_from_file_location(
        "baseline_primitives", a.baseline_source
    )
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)
    original = base._attention
    optimized = primitives._attention
    jobs = [
        json.loads(
            (root / "examples/examples_with_template/example_9fm7.json").read_text()
        )[0]
    ]
    if a.case == "batch":
        jobs.append(
            json.loads((root / "examples/example_without_msa.json").read_text())[0]
        )
    # Keep all real chains and ligands. No MSA/templates are used by this workflow.
    inputs = out / "inputs.json"
    inputs.write_text(json.dumps(jobs, indent=2))
    checkpoint = str(a.checkpoint.resolve())
    with patch.dict(
        inference_configs, {"dump_dir": str(out), "input_json_path": str(inputs)}
    ):
        runner = get_default_runner(
            batch_size=2 if a.case == "batch" else 1,
            use_msa=False,
            n_cycle=10,
            n_step=200,
            n_sample=5,
            dtype="bf16",
            device="cuda",
            trimul_kernel="flashpairformer",
            triatt_kernel="cuequivariance",
            load_checkpoint_path=checkpoint,
            need_atom_confidence=False,
            memory_settings={
                "infer_setting.sample_diffusion_chunk_size": 5,
                "infer_setting.confidence_sample_chunk_size": 5,
            },
        )
    runner.configs.input_json_path = str(inputs)
    runner.configs.use_template = False
    runner.configs.deterministic = a.deterministic
    prepared = []
    atoms = []
    seeds = [101, 103][: len(jobs)]
    dataset = InferenceDataset(runner.configs)
    for i, seed in enumerate(seeds):
        seed_everything(seed, deterministic=a.deterministic)
        data, atom_array, error = dataset[i]
        if error:
            raise RuntimeError(error)
        data["input_feature_dict"]["inference_seed"] = torch.tensor(seed)
        prepared.append(data)
        atoms.append(atom_array)
    runner.update_model_configs(
        update_inference_configs(
            copy.deepcopy(runner.configs), max(int(d["N_token"]) for d in prepared)
        )
    )
    metadata = [
        dict(
            name=d["sample_name"],
            tokens=int(d["N_token"]),
            atoms=len(at),
            protein_ca=int(protein_ca_mask(at).sum()),
            seed=seed,
        )
        for d, at, seed in zip(prepared, atoms, seeds)
    ]
    report = dict(
        case=a.case,
        deterministic=a.deterministic,
        samples=5,
        cycles=10,
        steps=200,
        inputs=metadata,
        torch=torch.__version__,
        python=__import__("sys").executable,
        gpu=torch.cuda.get_device_name(),
        settings=dict(
            dtype=runner.configs.dtype,
            skip_amp=dict(runner.model.configs.skip_amp),
            diffusion_chunk=5,
            confidence_chunk=5,
            multiplication="flashpairformer",
            attention="cuequivariance",
            tf32=False,
            use_msa=False,
            use_template=False,
        ),
        baseline_sha256=hashlib.sha256((a.baseline_source).read_bytes()).hexdigest(),
        optimized_sha256=hashlib.sha256(
            (root / "mol/model/modules/primitives.py").read_bytes()
        ).hexdigest(),
        runs=[],
        comparisons={},
    )
    torch.save(
        [protein_ca_mask(atom_array) for atom_array in atoms], out / "ca_masks.pt"
    )
    outputs = {}

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cpu(x) for x in value]
        return value

    def compare(lhs, rhs):
        comparisons = []
        for left, right, atom_array, meta in zip(lhs, rhs, atoms, metadata):
            delta = right["coordinate"].double() - left["coordinate"].double()
            ca = protein_ca_mask(atom_array)
            entry = dict(
                name=meta["name"],
                coordinate_exact=torch.equal(left["coordinate"], right["coordinate"]),
                coordinate_rmse=float(delta.square().mean().sqrt()),
                coordinate_max_abs=float(delta.abs().max()),
                ca_aligned_rmsd_per_sample=[
                    aligned_rmsd(x[ca], y[ca])
                    for x, y in zip(left["coordinate"], right["coordinate"])
                ],
                all_atom_aligned_rmsd_per_sample=[
                    aligned_rmsd(x, y)
                    for x, y in zip(left["coordinate"], right["coordinate"])
                ],
            )
            for key in ["pae", "atom_plddt"]:
                changes = right[key].double() - left[key].double()
                entry[key] = dict(
                    exact=torch.equal(left[key], right[key]),
                    max_abs=float(changes.abs().max()),
                    mean_abs=float(changes.abs().mean()),
                )
            for key in ["ipsae_min_10", "iptm", "ptm", "ranking_score"]:
                vals = [
                    (before.get(key), after.get(key))
                    for before, after in zip(left["summary"], right["summary"])
                ]
                entry[key] = dict(
                    before=[before for before, _ in vals],
                    after=[after for _, after in vals],
                    max_abs=max(
                        (
                            abs(float(before) - float(after))
                            for before, after in vals
                            if before is not None and after is not None
                        ),
                        default=0.0,
                    ),
                    exact=all(before == after for before, after in vals),
                )
            entry["ranking_order_before"] = np.argsort(
                [-float(s["ranking_score"]) for s in left["summary"]]
            ).tolist()
            entry["ranking_order_after"] = np.argsort(
                [-float(s["ranking_score"]) for s in right["summary"]]
            ).tolist()
            comparisons.append(entry)
        return comparisons

    for label in [
        "warm_baseline",
        "warm_optimized",
        "baseline_1",
        "optimized_1",
        "optimized_2",
        "baseline_2",
    ]:
        primitives._attention = original if "baseline" in label else optimized
        seed_everything(101, deterministic=a.deterministic)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        predictions = (
            runner.predict_batch(copy.deepcopy(prepared), seeds)
            if a.case == "batch"
            else [runner.predict(copy.deepcopy(prepared[0]))]
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        record = dict(
            label=label,
            seconds=seconds,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        report["runs"].append(record)
        print(json.dumps(record), flush=True)
        captured = []
        for pred, data, atom_array, seed in zip(predictions, prepared, atoms, seeds):
            validate_prediction(
                pred, n_atom=len(atom_array), n_token=int(data["N_token"]), n_sample=5
            )
            pred["coordinate"] = fix_cterminal_carboxyl_oxygens(
                pred["coordinate"], atom_array
            )
            groups = protein_token_groups(
                atom_array,
                pred["full_data"][0]["atom_to_token_idx"],
                int(data["N_token"]),
            )
            for summary, full in zip(pred["summary_confidence"], pred["full_data"]):
                summary.update(ipsae_summary(full["token_pair_pae"], groups))
            if not label.startswith("warm"):
                captured.append(
                    dict(
                        coordinate=cpu(pred["coordinate"]),
                        pae=torch.stack(
                            [cpu(f["token_pair_pae"]) for f in pred["full_data"]]
                        ),
                        atom_plddt=torch.stack(
                            [cpu(f["atom_plddt"]) for f in pred["full_data"]]
                        ),
                        summary=cpu(pred["summary_confidence"]),
                    )
                )
            if a.deterministic and label in ["baseline_1", "optimized_1"]:
                runner.dumper.base_dir = str(out / label)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=data["sample_name"],
                    seed=seed,
                    pred_dict=pred,
                    atom_array=atom_array.copy(),
                    entity_poly_type={
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )
                directory = (
                    out / label / data["sample_name"] / f"seed_{seed}" / "predictions"
                )
                assert len(list(directory.glob("*.cif"))) == 5
                assert len(list(directory.glob("*summary_confidence*.json"))) == 5
        if captured:
            outputs[label] = captured
        del predictions, pred
        (out / "report.json").write_text(
            json.dumps(
                report,
                indent=2,
                default=json_value,
            )
        )
    for name, left, right in [
        ("baseline_repeat", "baseline_1", "baseline_2"),
        ("optimized_repeat", "optimized_1", "optimized_2"),
        ("optimized_vs_baseline", "baseline_1", "optimized_1"),
    ]:
        report["comparisons"][name] = compare(outputs[left], outputs[right])
    report["median_seconds"] = {
        v: statistics.median(
            r["seconds"] for r in report["runs"] if r["label"].startswith(v + "_")
        )
        for v in ["baseline", "optimized"]
    }
    report["time_reduction_percent"] = 100 * (
        1 - report["median_seconds"]["optimized"] / report["median_seconds"]["baseline"]
    )
    report["notes"] = [
        "Interleaved ABBA timings after a complete warmup of each implementation.",
        "Timings include input copy/transfer, inference and confidence summaries; native ipSAE validation and CIF/JSON writing are outside the timed region.",
        "Same loaded model, preprocessed features, per-input seeds and settings; all five samples are compared by identity before ranking.",
        "A real-case numerical comparison is not a broad prediction-quality evaluation.",
    ]
    (out / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=json_value,
        )
    )
    torch.save(outputs, out / "predictions.pt")
    print(
        json.dumps(
            dict(
                medians=report["median_seconds"],
                time_reduction_percent=report["time_reduction_percent"],
                comparisons=report["comparisons"],
            ),
            default=json_value,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
