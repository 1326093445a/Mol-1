# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Cost-inclusive cuEquivariance layout/chunk audit; not a model speedup claim."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from cuequivariance_torch.primitives.triangle import triangle_attention


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(101)
    records = []
    for batch, tokens in [(1, 245), (2, 509), (2, 644)]:
        # Match the model's linear(...).view(...).transpose(...) layout.
        q, k, v = [
            torch.randn(
                batch, tokens, tokens, 4, 32, device="cuda", dtype=torch.bfloat16
            ).transpose(-2, -3)
            for _ in range(3)
        ]
        bias = (
            torch.randn(batch, tokens, tokens, 4, device="cuda")
            .permute(0, 3, 1, 2)
            .unsqueeze(1)
        )
        mask = torch.ones(batch, tokens, 1, 1, tokens, device="cuda", dtype=torch.bool)

        def invoke(mode):
            if mode == "contiguous":
                return triangle_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(), bias, mask
                )
            if mode.startswith("rows"):
                size = int(mode[4:])
                return torch.cat(
                    [
                        triangle_attention(
                            q[:, i : i + size],
                            k[:, i : i + size],
                            v[:, i : i + size],
                            bias,
                            mask[:, i : i + size],
                        )
                        for i in range(0, tokens, size)
                    ],
                    dim=1,
                )
            if mode == "split_inputs":
                return torch.cat(
                    [
                        triangle_attention(
                            q[i : i + 1],
                            k[i : i + 1],
                            v[i : i + 1],
                            bias[i : i + 1],
                            mask[i : i + 1],
                        )
                        for i in range(batch)
                    ]
                )
            return triangle_attention(q, k, v, bias, mask)

        modes = [
            "current",
            "contiguous",
            "rows64",
            "rows128",
            "rows256",
            "split_inputs",
        ]
        reference = invoke("current")
        rows = {mode: {"milliseconds": []} for mode in modes}
        for mode in modes:
            actual = invoke(mode)
            rows[mode]["exact"] = torch.equal(reference, actual)
            rows[mode]["max_abs"] = float((reference - actual).abs().max())
            for _ in range(3):
                invoke(mode)
        for repeat in range(6):
            for mode in modes if repeat % 2 == 0 else list(reversed(modes)):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    invoke(mode)
                end.record()
                end.synchronize()
                rows[mode]["milliseconds"].append(start.elapsed_time(end) / 10)
        for result in rows.values():
            result["median_ms"] = statistics.median(result["milliseconds"])
        records.append(
            {"batch": batch, "tokens": tokens, "q_stride": q.stride(), "modes": rows}
        )
        print(json.dumps(records[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "dtype": "bf16",
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
