# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Scoped compilation of token diffusion blocks, also used by CUDA inference.

Call the context manager after loading an eval model. It neither pads inputs nor changes the sampling
loop, random streams, precision, conditioning caches or registered parameters.
CUDA graph replay surrounds the whole token stack when requested.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from mol.model.modules.transformer import DiffusionTransformer


@dataclass
class CompilationSession:
    """Switch eager/compiled execution without rebuilding lazy compiled callables."""

    enabled: bool
    block_count: int
    compiled_calls: int = 0
    eager_calls: int = 0
    replay_enabled: bool = False
    graph_captures: int = 0
    graph_replays: int = 0
    graph_fallbacks: int = 0
    graph_cache_entries: int = 0
    fused_layer_norm_modules: int = 0


def _dispatch(block, original, compiled, session):
    def forward(*args, **kwargs):
        if not session.enabled or block.training or torch.is_grad_enabled():
            session.eager_calls += 1
            return original(*args, **kwargs)
        session.compiled_calls += 1
        # Compilation/runtime failures propagate: no silent eager fallback that
        # could turn a benchmark into a measurement of the wrong implementation.
        return compiled(*args, **kwargs)

    return forward


@contextmanager
def compiled_diffusion_blocks(
    transformer: DiffusionTransformer,
    *,
    backend: str | Callable = "inductor",
    enabled: bool = True,
    cuda_graphs: bool = False,
    max_graphs: int = 2,
    dynamic_shapes: bool = False,
    compile_threads: int = 1,
) -> Iterator[CompilationSession]:
    """Compile token blocks only, without graph breaks or added padding.

    Pass ``model.diffusion_module.diffusion_transformer``. The outer transformer
    keeps validation, conditioning preparation and orchestration eager. Bound
    forward methods are compiled off the module tree, preserving state_dict keys
    and Parameter identities. Training/gradient-enabled calls remain eager.
    Fused CUDA LayerNorm is exposed to the compiler as an opaque custom operator
    that calls the original kernel, preserving the selected LayerNorm backend.

    This temporarily changes instance forwards and requires exclusive ownership
    of this model instance. Do not concurrently serve predictions or change the
    device/dtype, replace blocks, or reload weights inside the context. Separate
    processes and other model instances are unaffected. All forwards are restored
    on exit, including exceptional exits.

    ``cuda_graphs=True`` captures the compiled token stack in a private CUDA
    graph per exact input signature. New input values are copied every call;
    returned outputs own their storage. At ``max_graphs`` signatures, additional
    shapes use compiled execution and increment ``graph_fallbacks``. Captures
    require CUDA/Inductor and exclusive use of the model on one caller stream.
    ``dynamic_shapes=True`` supports varying token lengths without specializing
    every length. CUDA graphs still use exact tensor signatures.
    """
    if not isinstance(transformer, DiffusionTransformer):
        raise TypeError("Expected a DiffusionTransformer")
    if transformer.training:
        raise ValueError("Put the model in eval mode before enabling compilation")
    if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs < 1:
        raise ValueError("max_graphs must be a positive integer")
    if isinstance(compile_threads, bool) or not isinstance(compile_threads, int) or compile_threads < 1:
        raise ValueError("compile_threads must be a positive integer")
    if cuda_graphs and (backend != "inductor" or not next(transformer.parameters()).is_cuda):
        raise ValueError("CUDA graph replay requires the Inductor backend and a CUDA model")
    if getattr(transformer, "_compilation_session", None) is not None:
        raise RuntimeError("This transformer already has a compilation session")
    if not transformer.blocks or any(
        b.attention_pair_bias.cross_attention_mode for b in transformer.blocks
    ):
        raise ValueError("This experiment supports nonempty token diffusion blocks only")
    from mol.model.context_parallel import active_context

    if active_context() is not None:
        raise ValueError("Context-parallel inference is outside this experiment")

    session = CompilationSession(enabled=enabled, block_count=len(transformer.blocks), replay_enabled=cuda_graphs)
    transformer._compilation_session = session
    restored = []
    missing = object()
    replay = None
    previous_forward = transformer.__dict__.get("forward", missing)
    try:
        from mol.model.layer_norm.layer_norm import FusedLayerNorm

        fused_norms = [module for module in transformer.modules() if isinstance(module, FusedLayerNorm)]
        if fused_norms:
            from mol.model.compiled_layer_norm import compile_compatible_forward

            for module in fused_norms:
                original = module.forward
                restored.append((module, module.__dict__.get("forward", missing)))
                module.forward = compile_compatible_forward(module, original)
            session.fused_layer_norm_modules = len(fused_norms)
        for block in transformer.blocks:
            original = block.forward
            options = {"triton.cudagraphs": False, "compile_threads": compile_threads} if backend == "inductor" else None
            compiled = torch.compile(
                original, backend=backend, fullgraph=True, dynamic=dynamic_shapes, options=options
            )
            restored.append((block, block.__dict__.get("forward", missing)))
            block.forward = _dispatch(block, original, compiled, session)
        if cuda_graphs:
            from mol.model.cuda_graph_replay import ReplayCache

            replay = ReplayCache(transformer.forward, transformer, session, max_graphs)
            transformer.forward = replay
        yield session
    finally:
        if replay is not None:
            if previous_forward is missing:
                del transformer.forward
            else:
                transformer.forward = previous_forward
        for block, previous in restored:
            if previous is missing:
                del block.forward
            else:
                block.forward = previous
        del transformer._compilation_session
        if replay is not None:
            replay.close()
