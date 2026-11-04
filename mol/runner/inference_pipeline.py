# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU preparation and writing with explicit per-job ownership."""

import copy
import multiprocessing
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch

from mol.utils.seed import seed_everything

_worker_dataset = None
_worker_deterministic = False


def _initialize_features(dataset, deterministic):
    # Spawned workers never own model/GPU state. Keep their RNGs independent
    # from inference and avoid CPU thread oversubscription per GPU worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    global _worker_dataset, _worker_deterministic
    _worker_dataset, _worker_deterministic = dataset, deterministic


def _prepare_features(task):
    index, seed = task
    seed_everything(seed, deterministic=_worker_deterministic)
    result = _worker_dataset[index]
    # Restore the serial loader's post-featurization CPU RNG state in the
    # parent too. This preserves consumers that still use a global RNG.
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    return result, state


@contextmanager
def prepared_jobs(dataset, tasks, *, deterministic, prefetch):
    """Yield (seed, data, atoms, error) in order, with at most prefetch futures."""
    if not prefetch:

        def serial():
            for index, seed in tasks:
                seed_everything(seed, deterministic=deterministic)
                yield (seed, *dataset[index])

        yield serial()
        return

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_features,
        initargs=(dataset, deterministic),
    ) as executor:
        tasks = iter(tasks)
        pending = deque()

        def enqueue():
            task = next(tasks, None)
            if task is not None:
                pending.append((task, executor.submit(_prepare_features, task)))
                return True
            return False

        for _ in range(prefetch):
            if not enqueue():
                break

        def consume():
            while pending:
                (_, seed), future = pending.popleft()
                result, state = future.result()  # Worker crashes propagate to the CLI.
                enqueue()
                seed_everything(seed, deterministic=deterministic)
                random.setstate(state[0])
                np.random.set_state(state[1])
                torch.random.set_rng_state(state[2])
                yield (seed, *result)

        try:
            yield consume()
        finally:
            for _, future in pending:
                future.cancel()


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class PredictionWriter:
    """One CPU-only write at a time; bounded ownership and errors drained on exit."""

    def __init__(self, dumper, report, *, background=False):
        self.dumper, self.report = dumper, report
        self.executor = ThreadPoolExecutor(max_workers=1) if background else None
        self.pending = None

    def __enter__(self):
        return self

    def _finish(self):
        if self.pending is not None:
            future, name, seed = self.pending
            self.pending = None
            try:
                future.result()
            except Exception as exc:
                self.report(
                    {"sample_name": name}, seed, f"Output writing failed: {exc}"
                )

    def submit(self, **kwargs):
        if self.executor is None:
            self.dumper.dump(**kwargs)
            return
        self._finish()
        # Do not retain large unused confidence logits or any CUDA tensors in
        # the writer. Copy before submitting so prediction buffers may be freed.
        kwargs["pred_dict"] = _cpu_copy(
            {
                key: kwargs["pred_dict"][key]
                for key in ("coordinate", "summary_confidence", "full_data")
            }
        )
        kwargs["atom_array"] = kwargs["atom_array"].copy()
        future = self.executor.submit(self.dumper.dump, **kwargs)
        self.pending = future, kwargs["pdb_id"], kwargs["seed"]

    def __exit__(self, *exc):
        try:
            self._finish()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
