# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-slot inference, refilled only between complete samples."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from mol.config.inference import validate_input_batching
from mol.model import batching
from mol.model.early_stopping import make_first_sample_gate
from mol.utils.torch_utils import autocasting_disable_decorator


@dataclass
class Slot:
    key: object
    seed: int
    state: tuple
    schedule: torch.Tensor
    gate: object
    generator: torch.Generator
    numpy_rng: np.random.Generator
    coordinates: list = field(default_factory=list)


def forward_refill(model, batches, on_prediction):
    """Consume bounded batches of (opaque key, features, seed), emit on completion.

    All inputs in a session must have exactly the same token count. Trunks and
    diffusion always run at configured B, including the final partial group.
    Padding slots repeat real conditioning but own their RNGs and emit nothing.
    At most B active and B-1 queued real jobs are retained between preparations.
    The callback must consume/copy a prediction before returning.
    """
    from mol.model.mol import update_input_feature_dict

    cfg = model.configs
    validate_input_batching(cfg)
    if not cfg.infer_setting.get("batch_refill", False) or model.training:
        raise ValueError("forward_refill requires batch_refill=true and eval mode")
    if model._device().type not in ("cpu", "cuda"):
        raise ValueError("Input refill supports CPU and CUDA")
    width = cfg.batch_size
    batches = iter(batches)
    ready = deque()
    token_count, chunk = None, None
    exhausted = False
    stats = dict(completed=0, stopped=0, trunk_batches=0, sample_rounds=0,
                 real_samples=0, placeholder_samples=0, refills=0, max_ready=0)
    sample_precision = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion)

    def prepare_next():
        nonlocal token_count, chunk, exhausted
        items = next(batches, None)
        if items is None:
            exhausted = True
            return []
        if not 1 <= len(items) <= width:
            raise ValueError("Refill preparation needs 1..batch_size inputs")
        if any(key is None for key, _, _ in items):
            raise ValueError("None is reserved for padding slots")
        lengths = [feat["token_index"].numel() for _, feat, _ in items]
        if token_count is None:
            token_count = lengths[0]
            chunk = (model._get_dynamic_chunk_size(token_count)
                     if cfg.infer_setting.dynamic_chunk_size else cfg.infer_setting.chunk_size)
            budget_chunk = max(1, 450_000_000 // (width * token_count * token_count))
            chunk = min(chunk or budget_chunk, budget_chunk)
        if any(n != token_count for n in lengths):
            raise ValueError("Refill requires exactly equal token counts; start a new session for a different length")
        budget = cfg.infer_setting.get("batch_max_pair_tokens", 0)
        if budget and width * token_count**2 > budget:
            raise ValueError("Fixed refill batch exceeds batch_max_pair_tokens; choose a smaller batch_size")
        padded = list(items) + [(None, items[-1][1], 0)] * (width - len(items))
        features = [update_input_feature_dict(
            model.relative_position_encoding.generate_relp(dict(feat))
        ) for _, feat, _ in padded]
        trunks = batching._trunk_batch(model, features, chunk)
        states, schedules = [], []
        for feat, (s_input, s, z) in zip(features, trunks, strict=True):
            states.append((feat, s_input, s, z, model.prepare_inference_cache(feat, z)))
            schedules.append(model.inference_noise_scheduler(
                N_step=cfg.sample_diffusion.N_step, device=s_input.device, dtype=s_input.dtype
            ))
        # Same order and precision as ordinary batched inference. All per-input
        # caches are owned by that input, including atom encoder/decoder caches.
        sample_states, sample_schedules = sample_precision(batching.prepare_sample_states)(
            model, states, schedules
        )
        slots = []
        for (key, _, seed), feat, trunk, state, schedule in zip(
            padded, features, trunks, sample_states, sample_schedules, strict=True
        ):
            gate = (make_first_sample_gate(model, [feat], [trunk], chunk, inplace_safe=True)
                    if key is not None else None)
            slots.append(Slot(key, seed, state, schedule, gate,
                              torch.Generator(device=state[1].device).manual_seed(seed),
                              np.random.default_rng(seed)))
        stats["trunk_batches"] += 1
        return slots

    slots = prepare_next()
    if not slots:
        return stats
    dirty = True
    projections = static_pair = attention_bias = None
    while any(slot.key is not None for slot in slots):
        states = [slot.state for slot in slots]
        if dirty:
            projections, static_pair, attention_bias = sample_precision(batching.prepare_batch_conditioning)(model, states)
            dirty = False

        @sample_precision
        def sample():
            return batching.sample_diffusion_batch(
                states,
                lambda x, t: batching._denoise_batch(
                    model.diffusion_module, states, x, t, chunk_size=chunk,
                    fusion=model.enable_efficient_fusion, projections=projections,
                    static_pair=static_pair, attention_bias=attention_bias,
                ),
                schedules=[slot.schedule for slot in slots], seeds=[slot.seed for slot in slots],
                generators=[slot.generator for slot in slots],
                numpy_rngs=[slot.numpy_rng for slot in slots], n_sample=1, chunk_size=1,
                **{key: cfg.sample_diffusion[key] for key in
                   ("gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta")},
            )

        coordinates = sample()
        stats["sample_rounds"] += 1
        stats["real_samples"] += sum(slot.key is not None for slot in slots)
        stats["placeholder_samples"] += sum(slot.key is None for slot in slots)
        for slot, coordinate in zip(slots, coordinates, strict=True):
            if slot.key is None:
                continue
            slot.coordinates.append(coordinate)
            stopped = slot.gate([coordinate]) if len(slot.coordinates) == 1 else False
            if stopped or len(slot.coordinates) == cfg.sample_diffusion.N_sample:
                pred = slot.gate.finish_heads(0, torch.cat(slot.coordinates, dim=0))
                model.summarize_prediction(pred, slot.state[0], model.N_cycle)
                slot.gate.annotate(pred, 0, batch_policy="preserve_shape_refill", batch_size=width)
                on_prediction(slot.key, pred)
                del pred
                stats["completed"] += 1
                stats["stopped"] += int(stopped)
                slot.key, slot.gate = None, None
                slot.coordinates.clear()
        del coordinates, coordinate, states
        # Retired slots remain placeholders until a compatible job is ready.
        # Incoming trunks are computed at B, never in a singleton shortcut.
        for i, slot in enumerate(slots):
            if slot.key is not None:
                continue
            if not ready and not exhausted:
                # Release old batch-wide projections before preparing more jobs.
                projections = static_pair = attention_bias = None
                dirty = True
                ready.extend(s for s in prepare_next() if s.key is not None)
            if ready:
                slots[i] = ready.popleft()
                stats["refills"] += 1
                dirty = True
            stats["max_ready"] = max(stats["max_ready"], len(ready))
    return stats
