# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Explicit benchmark-only runtime experiments; never installed by the runner.

Contexts temporarily wrap a model or helper and require exclusive ownership in
the current process. They preserve the selected kernels and tensor shapes.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import weakref
from unittest.mock import patch

import torch

from mol.model.cuda_graph_replay import ReplayCache, copy_inputs, owned_buffer, tensor_leaves, tree_key, tree_map


def tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no counter: always refresh them conservatively.
        return None


def sources_snapshot(value):
    return [(weakref.ref(t), tensor_version(t)) for t in tensor_leaves(value)]


def unchanged(snapshot, value):
    leaves = tensor_leaves(value)
    return snapshot is not None and len(snapshot) == len(leaves) and all(
        ref() is leaf and version is not None and tensor_version(leaf) == version
        for (ref, version), leaf in zip(snapshot, leaves, strict=True)
    )


class AggregationMetadata:
    """Prediction-scoped cache of stable reduction order and segment lengths."""
    def __init__(self):
        self.entries = {}
        self.builds = self.hits = 0

    def __call__(self, x_atom, atom_to_token_idx, n_token, reduce):
        index = atom_to_token_idx
        key = (id(index), n_token)
        version = tensor_version(index)
        cached = self.entries.get(key)
        if cached is None or cached[0]() is not index or version is None or cached[1] != version:
            order = torch.argsort(index, stable=True)
            lengths = torch.bincount(index, minlength=n_token or 0)
            if n_token is not None and lengths.numel() != n_token:
                raise RuntimeError("atom_to_token_idx exceeds n_token")
            cached = (weakref.ref(index), version, order, lengths)
            self.entries[key] = cached
            self.builds += 1
        else:
            self.hits += 1
        order, lengths = cached[2:]
        out = torch.segment_reduce(x_atom.index_select(-2, order), "sum",
                                   lengths=lengths.expand(*x_atom.shape[:-2], -1),
                                   axis=x_atom.ndim - 2)
        if reduce == "mean":
            out = out / lengths.clamp_min(1).to(out.dtype).unsqueeze(-1)
        return out


@contextmanager
def cache_aggregation_metadata():
    cache = AggregationMetadata()
    try:
        with patch("mol.model.utils._aggregate_atom_to_token_grouped", cache):
            yield cache
    finally:
        cache.entries.clear()


@contextmanager
def cache_sampler_branches():
    from mol.model import generator
    with patch("mol.model.mol.sample_diffusion", partial(
        generator.sample_diffusion, precompute_schedule_branches=True
    )):
        yield


@dataclass
class ReplaySession:
    enabled: bool = True
    replay_enabled: bool = True
    graph_captures: int = 0
    graph_replays: int = 0
    graph_fallbacks: int = 0
    graph_cache_entries: int = 0
    graph_evictions: int = 0
    copied_tensors: int = 0
    copied_bytes: int = 0
    skipped_tensors: int = 0
    skipped_bytes: int = 0
    cache_bytes: int = 0
    peak_cache_bytes: int = 0


@dataclass
class Entry:
    graph: object
    static: dict
    output: object
    snapshots: dict
    device: torch.device
    allocated_bytes: int


def physical_bytes(tensor):
    count = 1
    for n, stride in zip(tensor.shape, tensor.stride()):
        count *= 1 if stride == 0 and n else n
    return count * tensor.element_size()


class ExperimentalReplayCache(ReplayCache):
    """Eager capture with optional stable-input refresh and LRU eviction.

    Stable inputs must not be mutated through .data or untracked external CUDA
    writes. Ordinary in-place PyTorch edits and tensor replacement are tracked;
    tensors without version counters are always copied. max_cache_bytes bounds
    accounted live allocations retained by captures, not transient capture peak
    or allocator reservation. The benchmark also caps the process allocator.
    """
    def __init__(self, original, module, session, max_graphs=2, *,
                 stable_inputs=(), evict=False, max_cache_bytes=None):
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        if max_cache_bytes is not None and max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        super().__init__(original, module, session, max_graphs)
        self.entries = OrderedDict()
        self.stable_inputs = frozenset(stable_inputs)
        self.evict = evict
        self.max_cache_bytes = max_cache_bytes
        self.rejected = set()

    def _refresh(self, entry, arguments, *, force=False):
        for name, source in arguments.items():
            leaves = tensor_leaves(source)
            size = sum(physical_bytes(t) for t in leaves)
            if not force and name in self.stable_inputs and unchanged(entry.snapshots.get(name), source):
                self.session.skipped_tensors += len(leaves)
                self.session.skipped_bytes += size
                continue
            copy_inputs(tensor_leaves(entry.static[name]), leaves)
            self.session.copied_tensors += len(leaves)
            self.session.copied_bytes += size
            if name in self.stable_inputs:
                entry.snapshots[name] = sources_snapshot(source)

    def _evict_one(self):
        _, entry = self.entries.popitem(last=False)
        torch.cuda.synchronize(entry.device)
        entry.graph.reset()
        self.session.cache_bytes -= entry.allocated_bytes
        self.session.graph_cache_entries = len(self.entries)
        self.session.graph_evictions += 1

    def __call__(self, *args, **kwargs):
        session = self.session
        if not session.enabled or not session.replay_enabled or self.module.training or torch.is_grad_enabled():
            return self.original(*args, **kwargs)
        from mol.model.context_parallel import active_context
        if active_context() is not None:
            raise ValueError("Context parallelism is outside this experiment")
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        if any(arguments.get(name) is not None for name in ("n_queries", "n_keys", "atom_block_caches")):
            raise ValueError("Replay supports token diffusion only")
        leaves = tensor_leaves(arguments)
        device = leaves[0].device
        if device.type != "cuda" or any(t.device != device for t in leaves):
            raise ValueError("Graph inputs must all be on one CUDA device")
        key = (tree_key(arguments), torch.is_autocast_enabled("cuda"),
               torch.get_autocast_dtype("cuda"), torch.is_inference_mode_enabled(),
               torch.backends.cuda.matmul.allow_tf32)
        entry = self.entries.get(key)
        if entry is None:
            if key in self.rejected or (len(self.entries) >= self.max_graphs and not self.evict):
                session.graph_fallbacks += 1
                return self.original(*args, **kwargs)
            if len(self.entries) >= self.max_graphs:
                self._evict_one()
            before = torch.cuda.memory_allocated(device)
            static = tree_map(owned_buffer, arguments)
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    copy_inputs(tensor_leaves(static), leaves)
                    self.original(**static)
            stream.synchronize()
            copy_inputs(tensor_leaves(static), leaves)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(device), torch.cuda.graph(graph, stream=stream):
                output = self.original(**static)
            allocated = max(0, torch.cuda.memory_allocated(device) - before)
            entry = Entry(graph, static, output, {}, device, allocated)
            session.graph_captures += 1
            if self.max_cache_bytes is not None:
                while self.entries and session.cache_bytes + allocated > self.max_cache_bytes and self.evict:
                    self._evict_one()
                if session.cache_bytes + allocated > self.max_cache_bytes:
                    torch.cuda.synchronize(device)
                    graph.reset()
                    self.rejected.add(key)
                    session.graph_fallbacks += 1
                    del entry, graph, static, output
                    return self.original(*args, **kwargs)
            self.entries[key] = entry
            session.cache_bytes += allocated
            session.peak_cache_bytes = max(session.peak_cache_bytes, session.cache_bytes)
            session.graph_cache_entries = len(self.entries)
        elif self.evict:
            self.entries.move_to_end(key)
        # Always refresh after capture; a captured callable may modify buffers.
        self._refresh(entry, arguments)
        entry.graph.replay()
        session.graph_replays += 1
        return tree_map(lambda t: t.clone(), entry.output)

    def close(self):
        for device in {entry.device for entry in self.entries.values()}:
            torch.cuda.synchronize(device)
        for entry in self.entries.values():
            entry.graph.reset()
        self.entries.clear()
        self.rejected.clear()
        self.session.cache_bytes = 0
        self.session.graph_cache_entries = 0


@contextmanager
def replay_eager_token_stack(transformer, **options):
    if transformer.training or getattr(transformer, "_compilation_session", None) is not None:
        raise ValueError("Need an exclusively owned eval transformer")
    session = ReplaySession()
    missing = object()
    previous = transformer.__dict__.get("forward", missing)
    replay = ExperimentalReplayCache(transformer.forward, transformer, session, **options)
    transformer._compilation_session = session
    transformer.forward = replay
    try:
        yield replay
    finally:
        if previous is missing:
            del transformer.forward
        else:
            transformer.forward = previous
        del transformer._compilation_session
        replay.close()
