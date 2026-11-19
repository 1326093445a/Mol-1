# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Inference-only compiler boundary around the existing fused CUDA kernel.

Imported by the diffusion compilation context. The custom operator
keeps the existing kernel, casts and epsilon; it does not substitute torch
LayerNorm. Ordinary eager calls and gradient-enabled calls use the original
module forward.
"""

import torch

from mol.model.layer_norm.layer_norm import FusedLayerNormAffineFunction


@torch.library.custom_op("mol_inference::fused_layer_norm", mutates_args=(), device_types="cuda")
def fused_layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> torch.Tensor:
    return FusedLayerNormAffineFunction.apply(
        input, weight, bias, torch.Size(normalized_shape), eps
    )


@fused_layer_norm.register_fake
def _fake_fused_layer_norm(input, weight, bias, normalized_shape, eps):
    # The original forward makes the input contiguous before calling CUDA.
    return torch.empty_like(input, memory_format=torch.contiguous_format)


def compile_compatible_forward(module, original):
    """Return a temporary instance forward; eager/training behavior is retained."""
    def forward(input):
        if (
            not torch.compiler.is_compiling()
            or module.training
            or torch.is_grad_enabled()
            or not input.is_cuda
        ):
            return original(input)
        return fused_layer_norm(
            input, module.weight, module.bias, list(module.normalized_shape), module.eps
        )

    return forward
