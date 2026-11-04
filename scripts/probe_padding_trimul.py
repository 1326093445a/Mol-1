# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Probe real first-block activations/weights with zero and poisoned padding."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from mol.model.triangular.triangular import TriangleMultiplicationOutgoing
from mol.model.modules.pairformer import PairMixerBlock


def difference(a, b):
    delta = a.float() - b.float()
    return {
        "exact": torch.equal(a, b),
        "max_abs": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "changed": int(torch.count_nonzero(delta)),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--activation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(101)
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["model"]
    prefix = "module.pairformer_stack.blocks.0.tri_mul_out."
    weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    width = weights["linear_a_p.weight"].shape[-1]
    module = TriangleMultiplicationOutgoing(width, width).cuda().eval()
    module.load_state_dict(weights, strict=True)
    z = torch.load(args.activation, weights_only=False)["z"].cuda()
    n = len(z)
    padded = F.pad(z, (0, 0, 0, 1000 - n, 0, 1000 - n))
    mask = torch.zeros(1000, 1000, device="cuda", dtype=z.dtype)
    mask[:n, :n] = 1
    poisoned = torch.randn_like(padded) * 10
    poisoned[:n, :n] = z
    report, outputs = {}, {}
    for name, backend, autocast, reduced in [
        ("tmk3", "flashpairformer", True, True),
        ("cueq", "cuequivariance", True, True),
        ("torch_bf16", "torch", True, True),
        ("tmk3_full_reduction", "flashpairformer", True, False),
        ("torch_fp32", "torch", False, False),
    ]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
        with (
            torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext()
        ):
            x = z if autocast else z.float()
            pad = padded if autocast else padded.float()
            poison = poisoned if autocast else poisoned.float()
            a = module(x, triangle_multiplicative=backend)
            b = module(pad, mask=mask, triangle_multiplicative=backend)[:n, :n].clone()
            c = module(poison, mask=mask, triangle_multiplicative=backend)[
                :n, :n
            ].clone()
        report[name] = {
            "unpadded_vs_padded": difference(a, b),
            "masked_poison_vs_zero": difference(b, c),
        }
        if name == "tmk3":
            outputs = {"single": a.clone(), "padded": b.clone()}
        else:
            report[name]["unpadded_vs_tmk3"] = difference(a, outputs["single"])
            report[name]["padded_vs_tmk3"] = difference(b, outputs["padded"])
        del a, b, c
        print(json.dumps({name: report[name]}), flush=True)
        args.output.write_text(json.dumps(report, indent=2))

    prefix = "module.pairformer_stack.blocks.0."
    block = PairMixerBlock(c_z=width, hidden_scale_up=True).cuda().eval()
    block.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
        strict=True,
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    for fp32 in (False, True):
        a, b, c = z.clone(), padded.clone(), poisoned.clone()
        if fp32:
            a, b, c = a.float(), b.float(), c.float()
        stages = {}
        with torch.autocast("cuda", enabled=not fp32, dtype=torch.bfloat16):
            for name in ("tri_mul_out", "tri_mul_in", "pair_transition"):
                module = getattr(block, name)
                if name == "pair_transition":
                    a = a + module(a.unsqueeze(0), chunk_size=128)[0]
                    b = b + module(b.unsqueeze(0), chunk_size=128)[0]
                    c = c + module(c.unsqueeze(0), chunk_size=128)[0]
                else:
                    kw = {
                        "triangle_multiplicative": "torch"
                        if fp32
                        else "flashpairformer",
                        "inplace_safe": True,
                        "_add_with_inplace": True,
                    }
                    a = module(
                        a.unsqueeze(0),
                        mask=torch.ones_like(mask[:n, :n]).unsqueeze(0),
                        **kw,
                    )[0]
                    b = module(b.unsqueeze(0), mask=mask.unsqueeze(0), **kw)[0]
                    c = module(c.unsqueeze(0), mask=mask.unsqueeze(0), **kw)[0]
                stages[name] = {
                    "unpadded_vs_padded": difference(a, b[:n, :n]),
                    "masked_poison_vs_zero": difference(b[:n, :n], c[:n, :n]),
                }
        report["block_chain_fp32" if fp32 else "block_chain_bf16"] = stages
        print(json.dumps({"fp32": fp32, "stages": stages}), flush=True)
        args.output.write_text(json.dumps(report, indent=2))

    # The first observed difference is in incoming multiplication, after an
    # exactly matching outgoing update. Compare the public kernels there too.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = block.tri_mul_out(
            z,
            triangle_multiplicative="flashpairformer",
            inplace_safe=True,
            _add_with_inplace=True,
        )
        b = block.tri_mul_out(
            padded,
            mask=mask,
            triangle_multiplicative="flashpairformer",
            inplace_safe=True,
            _add_with_inplace=True,
        )
        incoming = {}
        for backend in ("flashpairformer", "cuequivariance", "torch"):
            x = block.tri_mul_in(
                a.clone(),
                triangle_multiplicative=backend,
                inplace_safe=True,
                _add_with_inplace=True,
            )
            y = block.tri_mul_in(
                b.clone(),
                mask=mask,
                triangle_multiplicative=backend,
                inplace_safe=True,
                _add_with_inplace=True,
            )[:n, :n].clone()
            incoming[backend] = {"unpadded_vs_padded": difference(x, y)}
            if backend == "flashpairformer":
                ref_x, ref_y = x, y
            else:
                incoming[backend]["unpadded_vs_tmk3"] = difference(x, ref_x)
                incoming[backend]["padded_vs_tmk3"] = difference(y, ref_y)
        report["incoming_backends"] = incoming
        print(json.dumps({"incoming_backends": incoming}), flush=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
