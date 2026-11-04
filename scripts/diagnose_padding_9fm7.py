# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Local-asset diagnosis of padding versus chunk-size sensitivity on real 9fm7.

A JSON plan contains a list of cases and a `finish` boolean. Cases may be added
while the loaded model is idle. Full predictions always use 5 samples/10 cycles.
"""

import argparse
import copy
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.metrics.ipsae import ipsae_summary, protein_token_groups
from mol.model import batching
from mol.runner.batch_inference import get_default_runner
from mol.runner.batched_prediction import validate_prediction
from mol.runner.inference import (
    fix_cterminal_carboxyl_oxygens,
    update_inference_configs,
)
from mol.utils.seed import seed_everything


def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [cpu(v) for v in x]
    return x


def json_value(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def difference(a, b):
    b = b.to(a.device)
    d = a.float() - b.float()
    return {
        "exact": torch.equal(a, b),
        "max_abs": float(d.abs().max()),
        "rmse": float(d.square().mean().sqrt()),
        "reference_rms": float(b.float().square().mean().sqrt()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    assert args.checkpoint.is_file()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs.json"
    jobs = [
        json.loads(
            (root / "examples/examples_with_template/example_9fm7.json").read_text()
        )[0],
        json.loads((root / "examples/example_without_msa.json").read_text())[1],
    ]
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
    runner.configs.input_json_path = str(inputs)
    runner.configs.use_template = False
    runner.configs.deterministic = False
    dataset = InferenceDataset(runner.configs)
    prepared, atoms = [], []
    for index, seed in enumerate([107, 109]):
        seed_everything(seed, deterministic=False)
        data, atom_array, error = dataset[index]
        assert not error, error
        prepared.append(data)
        atoms.append(atom_array)
    n = int(prepared[0]["N_token"])
    model = runner.model
    original_pad, original_masks = batching.pad_token_tensors, batching.token_masks
    original_trunk, original_denoise = batching._trunk_batch, batching._denoise_batch
    references = {}
    report = {
        "settings": {
            "samples": 5,
            "cycles": 10,
            "steps": 200,
            "seed": 107,
            "tf32": False,
            "msa": False,
            "templates": False,
        },
        "cases": [],
    }

    def save():
        (out / "report.json").write_text(
            json.dumps(report, indent=2, default=json_value)
        )

    completed = 0
    while True:
        plan = json.loads(args.plan.read_text())
        if completed >= len(plan["cases"]):
            if plan.get("finish", False):
                break
            time.sleep(2)
            continue
        case = plan["cases"][completed]
        label = case["label"]
        print(json.dumps({"starting": case, "unix_time": time.time()}), flush=True)
        record = {"case": case, "stages": {}, "started_unix": time.time()}
        phase, cycle, step = "trunk", 0, 0
        handles = []

        def capture(name, value):
            if completed == 0:
                references[name] = cpu(value)
                record["stages"][name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            elif name in references:
                record["stages"][name] = difference(value, references[name])

        def padded(values, *, pair=False, out=None):
            target = int(
                case.get("pad_trunk" if phase == "trunk" else "pad_diffusion", 0)
            )
            value = (
                original_pad(values, pair=pair)
                if target
                else original_pad(values, pair=pair, out=out)
            )
            if value.shape[-2] < target:
                valid_length = value.shape[-2]
                padding = (0, 0, 0, target - value.shape[-2])
                if pair:
                    padding += (0, target - value.shape[-3])
                value = F.pad(value, padding)
                if case.get("poison_padding", False):
                    noise = torch.randn_like(value) * 10
                    if pair:
                        noise[..., :valid_length, :valid_length, :] = value[
                            ..., :valid_length, :valid_length, :
                        ]
                    else:
                        noise[..., :valid_length, :] = value[..., :valid_length, :]
                    value = noise
            return value

        def masks(lengths, reference, biases=None):
            target = int(
                case.get("pad_trunk" if phase == "trunk" else "pad_diffusion", 0)
            )
            if target > max(lengths):
                pair, bias = original_masks(
                    lengths + [target],
                    reference,
                    None if biases is None else biases + [None],
                )
                return pair[:-1], bias[:-1]
            return original_masks(lengths, reference, biases)

        def before_stack(module, positional, kwargs):
            nonlocal cycle
            cycle += 1
            capture(f"cycle{cycle}.input_s", kwargs["s"][0, :n])
            if cycle == 1:
                capture("cycle1.input_z", kwargs["z"][0, :n, :n])
                if completed == 0:
                    torch.save(
                        cpu({"s": kwargs["s"][0, :n], "z": kwargs["z"][0, :n, :n]}),
                        out / "first_trunk_input.pt",
                    )

        def after_stack(module, positional, kwargs, result):
            capture(f"cycle{cycle}.output_s", result[0][0, :n])
            if cycle in (1, 10):
                capture(f"cycle{cycle}.output_z", result[1][0, :n, :n])

        def block_hook(index):
            def hook(module, positional, kwargs, result):
                if cycle == 1:
                    capture(f"cycle1.block{index}.s", result[0][0, :n])
                    capture(f"cycle1.block{index}.z", result[1][0, :n, :n])

            return hook

        def trunk(*a, **kw):
            nonlocal phase
            phase = "trunk"
            dtype = case.get("trunk_dtype", "bf16")
            with torch.autocast("cuda", enabled=dtype != "fp32", dtype=torch.bfloat16):
                result = original_trunk(*a, **kw)
            capture("final_trunk.s_input", result[0][0])
            torch.save(cpu(result[0]), out / f"{label}_trunk.pt")
            if case.get("reuse_trunk"):
                restored = torch.load(
                    out / f"{case['reuse_trunk']}_trunk.pt", weights_only=False
                )
                result[0] = tuple(t.to("cuda") for t in restored)
            phase = "diffusion"
            return result

        def denoise(*a, **kw):
            nonlocal step
            step += 1
            if step in (1, 2, 200):
                capture(f"step{step}.noisy", a[2][0])
            result = original_denoise(*a, **kw)
            if step in (1, 2, 200):
                capture(f"step{step}.denoised", result[0])
            return result

        handles.append(
            model.pairformer_stack.register_forward_pre_hook(
                before_stack, with_kwargs=True
            )
        )
        handles.append(
            model.pairformer_stack.register_forward_hook(after_stack, with_kwargs=True)
        )
        for index in [
            0,
            1,
            model.pairformer_stack.n_pairmixer_blocks,
            len(model.pairformer_stack.blocks) - 1,
        ]:
            handles.append(
                model.pairformer_stack.blocks[index].register_forward_hook(
                    block_hook(index), with_kwargs=True
                )
            )
        cfg = update_inference_configs(copy.deepcopy(runner.configs), 1000)
        cfg.triangle_multiplicative = case.get("trimul", "flashpairformer")
        cfg.triangle_attention = case.get("triatt", "cuequivariance")
        cfg.infer_setting.chunk_size = case.get("chunk", 1024)
        runner.update_model_configs(cfg)
        count = 2 if case.get("mixed", False) else 1
        seed_everything(101, deterministic=False)
        try:
            with ExitStack() as stack:
                for name, fn in [
                    ("pad_token_tensors", padded),
                    ("token_masks", masks),
                    ("_trunk_batch", trunk),
                    ("_denoise_batch", denoise),
                ]:
                    stack.enter_context(patch.object(batching, name, fn))
                predictions = runner.predict_batch(
                    copy.deepcopy(prepared[:count]), [107, 109][:count]
                )
            pred = predictions[0]
            validate_prediction(pred, n_atom=len(atoms[0]), n_token=n, n_sample=5)
            capture("coordinates", pred["coordinate"])
            pae = torch.stack([f["token_pair_pae"] for f in pred["full_data"]])
            capture("pae", pae)
            groups = protein_token_groups(
                atoms[0], pred["full_data"][0]["atom_to_token_idx"], n
            )
            summaries = []
            for full, summary in zip(pred["full_data"], pred["summary_confidence"]):
                summary.update(ipsae_summary(full["token_pair_pae"], groups))
                summaries.append(cpu(summary))
            record["summary"] = summaries
            group_a, group_b = list(groups.values())
            record["interchain"] = []
            for values in pae.cpu().numpy():
                directions = [
                    values[np.ix_(group_a, group_b)],
                    values[np.ix_(group_b, group_a)],
                ]
                record["interchain"].append(
                    [
                        {
                            "min_pae": float(v.min()),
                            "pairs_under_10_after_rounding": int(
                                (np.round(v, 2) < 10).sum()
                            ),
                        }
                        for v in directions
                    ]
                )
            torch.save(
                cpu(
                    {"coordinate": pred["coordinate"], "pae": pae, "summary": summaries}
                ),
                out / f"{label}_prediction.pt",
            )
            pred["coordinate"] = fix_cterminal_carboxyl_oxygens(
                pred["coordinate"], atoms[0]
            )
            runner.dumper.base_dir = str(out / label)
            runner.dumper.dump(
                dataset_name="",
                pdb_id="9fm7",
                seed=107,
                pred_dict=pred,
                atom_array=atoms[0].copy(),
                entity_poly_type=prepared[0]["entity_poly_type"],
            )
            del predictions, pred, pae
            record["finished_unix"] = time.time()
            print(
                json.dumps(
                    {
                        "finished": label,
                        "ipsae": [s["ipsae_min_10"] for s in summaries],
                        "stages": record["stages"],
                    },
                    default=json_value,
                ),
                flush=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        report["cases"].append(record)
        save()
        completed += 1


if __name__ == "__main__":
    main()
