# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA replay for an exclusively owned inference callable.

Inputs are copied into owned, stride-preserving buffers on every invocation.
Outputs are cloned outside capture so later replays cannot overwrite them.
The prediction runner owns the cache for the lifetime of its loaded model.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import inspect

import torch


@dataclass
class EagerReplaySession:
    """Runner-compatible counters for replay without compilation."""

    enabled: bool = True
    replay_enabled: bool = True
    compiled_calls: int = 0
    graph_captures: int = 0
    graph_replays: int = 0
    graph_fallbacks: int = 0
    graph_cache_entries: int = 0
    fused_layer_norm_modules: int = 0


def tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: tree_map(fn, getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {key: tree_map(fn, v) for key, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported graph argument: {type(value).__name__}")


def tree_key(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.stride(), value.dtype, value.device)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, tree_key(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, dict):
        return (dict, tuple((key, tree_key(v)) for key, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(tree_key(v) for v in value))
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    raise TypeError(f"Unsupported graph argument: {type(value).__name__}")


def tensor_leaves(value):
    leaves = []
    tree_map(lambda t: leaves.append(t) or t, value)
    return leaves


def owned_buffer(tensor):
    # Expanded read-only inputs keep zero strides. Copy only one representative
    # of each broadcast dimension, then expose the original logical shape.
    shape = tuple(1 if stride == 0 and n else n for n, stride in zip(tensor.shape, tensor.stride()))
    backing = torch.empty_strided(shape, tensor.stride(), device=tensor.device, dtype=tensor.dtype)
    return backing.as_strided(tensor.shape, tensor.stride())


def copy_inputs(destinations, sources):
    for dest, src in zip(destinations, sources, strict=True):
        slices = tuple(slice(0, 1) if stride == 0 and n else slice(None) for n, stride in zip(src.shape, src.stride()))
        dest[slices].copy_(src[slices])


class ReplayCache:
    """One private graph/pool per exact signature, with an explicit size limit."""

    def __init__(self, original, module, session, max_graphs):
        self.original = original
        self.signature = inspect.signature(original)
        self.module = module
        self.session = session
        self.max_graphs = max_graphs
        self.entries = {}

    def __call__(self, *args, **kwargs):
        session = self.session
        if not session.enabled or not session.replay_enabled or self.module.training or torch.is_grad_enabled():
            return self.original(*args, **kwargs)
        from mol.model.context_parallel import active_context

        if active_context() is not None:
            raise ValueError("CUDA graph replay does not support context-parallel inference")
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        if arguments.get("n_queries") is not None or arguments.get("n_keys") is not None or arguments.get("atom_block_caches") is not None:
            raise ValueError("CUDA graph replay supports token diffusion only")
        sources = tensor_leaves(arguments)
        device = sources[0].device
        if device.type != "cuda" or any(t.device != device for t in sources):
            raise ValueError("Graph inputs must all be on one CUDA device")
        key = (
            tree_key(arguments), torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"), torch.is_inference_mode_enabled(),
            torch.backends.cuda.matmul.allow_tf32,
        )
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                session.graph_fallbacks += 1
                return self.original(*args, **kwargs)
            static = tree_map(owned_buffer, arguments)
            destinations = tensor_leaves(static)
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    copy_inputs(destinations, sources)
                    self.original(**static)
            # All compilation and library initialization finishes before capture.
            stream.synchronize()
            copy_inputs(destinations, sources)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(device), torch.cuda.graph(graph, stream=stream):
                output = self.original(**static)
            entry = (graph, static, destinations, output)
            self.entries[key] = entry
            session.graph_captures += 1
            session.graph_cache_entries = len(self.entries)
        graph, static, destinations, output = entry
        copy_inputs(destinations, sources)
        graph.replay()
        session.graph_replays += 1
        # Retaining an earlier result across another replay is safe, including
        # across requests with new masks, conditioning and pair projections.
        return tree_map(lambda t: t.clone(), output)

    def close(self):
        devices = {t.device for _, _, leaves, _ in self.entries.values() for t in leaves}
        for device in devices:
            torch.cuda.synchronize(device)
        for graph, _, _, _ in self.entries.values():
            graph.reset()
        self.entries.clear()
        self.session.graph_cache_entries = 0


@contextmanager
def eager_diffusion_replay(transformer, *, max_graphs=2):
    """Capture the original token stack, preserving kernels and exact shapes.

    No compilation or LayerNorm substitution is performed. Inputs are refreshed
    on every call and returned outputs own their storage. Once the cache holds
    ``max_graphs`` signatures, unseen shapes run eagerly. This context requires
    exclusive model ownership on one CUDA caller stream; close before changing
    model weights, device or dtype. Training and gradient-enabled calls stay
    eager. Instance methods are restored on normal and exceptional exits.
    """
    from mol.model.context_parallel import active_context
    from mol.model.layer_norm.layer_norm import FusedLayerNorm
    from mol.model.modules.transformer import DiffusionTransformer

    if not isinstance(transformer, DiffusionTransformer):
        raise TypeError("Expected a DiffusionTransformer")
    if transformer.training:
        raise ValueError("Put the model in eval mode before enabling replay")
    if not next(transformer.parameters()).is_cuda:
        raise ValueError("Eager CUDA graph replay requires a CUDA model")
    if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs < 1:
        raise ValueError("max_graphs must be a positive integer")
    if not transformer.blocks or any(
        block.attention_pair_bias.cross_attention_mode for block in transformer.blocks
    ):
        raise ValueError("Replay supports nonempty token diffusion blocks only")
    if active_context() is not None:
        raise ValueError("Context-parallel inference does not support CUDA replay")
    if getattr(transformer, "_compilation_session", None) is not None:
        raise RuntimeError("This transformer already has an acceleration session")

    session = EagerReplaySession(fused_layer_norm_modules=sum(
        isinstance(module, FusedLayerNorm) for module in transformer.modules()
    ))
    missing = object()
    previous = transformer.__dict__.get("forward", missing)
    replay = ReplayCache(transformer.forward, transformer, session, max_graphs)
    transformer._compilation_session = session
    transformer.forward = replay
    try:
        yield session
    finally:
        if previous is missing:
            del transformer.forward
        else:
            transformer.forward = previous
        del transformer._compilation_session
        replay.close()
