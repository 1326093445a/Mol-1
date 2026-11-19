# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Bounded exact-length sessions and per-job failure ownership for slot refill."""

import copy
import itertools
import time
import traceback

import torch

from mol.utils.logger import get_logger

logger = get_logger(__name__)


def run_refill_sessions(runner, configs, prepared, submit, report):
    from mol.runner.inference import update_inference_configs

    def valid_jobs():
        for seed, data, atoms, error in prepared:
            if error:
                report(data, seed, error)
            else:
                yield data, atoms, seed

    # Only a contiguous exact-length run shares slots. One-item lookahead and
    # one preparation group keep CPU/GPU queues bounded even for huge manifests.
    for n_token, jobs in itertools.groupby(valid_jobs(), key=lambda item: int(item[0]["N_token"].item())):
        pending = {}
        counter = itertools.count()

        def batches():
            while items := list(itertools.islice(jobs, configs.batch_size)):
                batch = []
                for data, atoms, seed in items:
                    key = next(counter)
                    pending[key] = data, atoms, seed
                    batch.append((key, data, seed))
                yield batch

        def completed(key, pred):
            data, atoms, seed = pending.pop(key)  # Duplicate/unknown output is an error.
            try:
                submit(data, atoms, seed, pred)
            except Exception:
                report(data, seed, traceback.format_exc())

        started = time.monotonic()
        try:
            runner.update_model_configs(update_inference_configs(copy.deepcopy(configs), n_token))
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            stats = runner.predict_refill(batches(), completed)
            if pending:
                raise ValueError("Refill returned without emitting every admitted job")
            remaining = next(jobs, None)
            if remaining is not None:
                pending[next(counter)] = remaining
                raise ValueError("Refill returned before consuming every job in the session")
            logger.info("Fixed-slot refill: %s, %.2fs including output submission", stats, time.monotonic() - started)
            if runner.device.type == "cuda":
                logger.info("Peak allocated GPU memory: %.3f GiB", torch.cuda.max_memory_allocated(runner.device) / 2**30)
        except Exception as exc:
            detail = traceback.format_exc()
            exc.__traceback__ = None
            # A failed session never retries with smaller B or overwrites jobs
            # already emitted. Later lengths can still start independent sessions.
            for data, _, seed in pending.values():
                report(data, seed, detail)
            for data, _, seed in jobs:
                report(data, seed, "Refill session aborted: " + detail)
            if not pending:
                # An exception after the last emission must still fail the CLI.
                logger.error("Refill session failed after output: %s", detail)
                raise
            pending.clear()
            if runner.device.type == "cuda":
                torch.cuda.empty_cache()
