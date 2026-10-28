# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Modifications Copyright 2026 Molence Bio
# Adapted from OpenDDE's opendde/model/batching.py (Apache-2.0).
"""Batch token networks while retaining each complex's original atom topology.

The independent-input axis B and diffusion-sample axis S remain separate.
Only explicit token axes are padded; atom processing and confidence stay local.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mol.config.inference import validate_input_batching
from mol.model.utils import centre_random_augmentation
from mol.utils.torch_utils import autocasting_disable_decorator


def pad_token_tensors(
    values: list[torch.Tensor], *, pair: bool = False
) -> torch.Tensor:
    """Stack [*, T, C] or [*, T, T, C], padding only their token axes."""
    if not values:
        raise ValueError("Cannot pad an empty inference batch.")
    spatial_dims = 3 if pair else 2
    prefix = values[0].shape[:-spatial_dims]
    width = values[0].shape[-1]
    for value in values:
        if value.shape[:-spatial_dims] != prefix or value.shape[-1] != width:
            raise ValueError("Batch members must share sample and channel dimensions.")
        if value.shape[-2] < 1 or (pair and value.shape[-3] != value.shape[-2]):
            raise ValueError(
                "Token dimensions must be nonempty; pair tensors must be square."
            )
    size = max(value.shape[-2] for value in values)
    return torch.stack(
        [
            F.pad(
                value,
                (0, 0, 0, size - value.shape[-2])
                + ((0, size - value.shape[-3]) if pair else ()),
            )
            for value in values
        ]
    )


def token_masks(
    lengths: list[int],
    reference: torch.Tensor,
    biases: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair validity and finite attention bias, including per-job structural bias."""
    size = max(lengths)
    valid = (
        torch.arange(size, device=reference.device)[None, :]
        < torch.tensor(lengths, device=reference.device)[:, None]
    )
    pair = valid[:, :, None] & valid[:, None, :]
    bias = reference.new_zeros(len(lengths), size, size)
    if biases is not None:
        for i, (length, item) in enumerate(zip(lengths, biases, strict=True)):
            if item is not None:
                if item.shape != (length, length):
                    raise ValueError(
                        "Structural attention bias has the wrong token axes."
                    )
                bias[i, :length, :length] = item
    # Finite masks also keep completely padded query rows finite. Those rows
    # are discarded; valid queries cannot attend to padded keys.
    bias.masked_fill_(~pair, -torch.finfo(bias.dtype).max)
    return pair.to(reference.dtype), bias


def pairformer_batch(stack, singles, pairs, *, chunk_size=None, **kwargs):
    """Run one masked trunk call, then restore each input's token extent."""
    lengths = [s.shape[-2] for s in singles]
    s = pad_token_tensors(singles)
    z = pad_token_tensors(pairs, pair=True)
    mask, bias = token_masks(lengths, z)
    s, z = stack(
        s=s,
        z=z,
        pair_mask=mask,
        extra_attn_bias=bias,
        triangle_multiplicative="torch",
        triangle_attention="torch",
        inplace_safe=True,
        chunk_size=chunk_size,
        **kwargs,
    )
    # Own storage before the next per-input in-place recycle; cropped views
    # must not retain the entire padded batch.
    return (
        [s[i, :n].clone() for i, n in enumerate(lengths)],
        [z[i, :n, :n].clone() for i, n in enumerate(lengths)],
    )


def _trunk_batch(model, features, chunk_size):
    inputs, initial_s, initial_z = [], [], []
    kernels = dict(triangle_multiplicative="torch", triangle_attention="torch")
    for feat in features:
        s_input = model.input_embedder(feat, inplace_safe=False, chunk_size=chunk_size)
        s = model.linear_no_bias_sinit(s_input)
        z = (
            model.linear_no_bias_zinit1(s)[:, None, :]
            + model.linear_no_bias_zinit2(s)[None, :, :]
            + model.relative_position_encoding(feat["relp"])
            + model.linear_no_bias_token_bond(feat["token_bonds"].unsqueeze(-1))
        )
        inputs.append(s_input)
        initial_s.append(s)
        initial_z.append(z)
    singles = [torch.zeros_like(s) for s in initial_s]
    pairs = [torch.zeros_like(z) for z in initial_z]
    for _ in range(model.N_cycle):
        for i, feat in enumerate(features):
            z = initial_z[i] + model.linear_no_bias_z_cycle(
                model.layernorm_z_cycle(pairs[i])
            )
            # Preserve the existing query-only MSA and empty-template semantics.
            if model.template_embedder.n_blocks > 0:
                z = z + model.template_embedder(
                    feat, z, inplace_safe=True, chunk_size=chunk_size, **kernels
                )
            pairs[i] = model.msa_module(
                feat,
                z,
                inputs[i],
                pair_mask=None,
                inplace_safe=True,
                chunk_size=chunk_size,
                pair_transition_chunk_size=model._pair_transition_chunk_size(
                    "msa_pair_transition_chunk_size", chunk_size
                ),
                **kernels,
            )
            singles[i] = initial_s[i] + model.linear_no_bias_s(
                model.layernorm_s(singles[i])
            )
        singles, pairs = pairformer_batch(
            model.pairformer_stack,
            singles,
            pairs,
            chunk_size=chunk_size,
            pairmixer_transition_chunk_size=model._pair_transition_chunk_size(
                "trunk_pairmixer_transition_chunk_size", chunk_size
            ),
            pairformer_transition_chunk_size=model._pair_transition_chunk_size(
                "trunk_pairformer_transition_chunk_size", chunk_size
            ),
        )
    return list(zip(inputs, singles, pairs, strict=True))


def _denoise_batch(module, states, noisy, levels, *, chunk_size, fusion, projections):
    """Per-input atom networks surrounding one B x S token transformer."""
    atoms, conditions, pair_conditions, skips = [], [], [], []
    for (feat, s_input, s, z, cache), x, level in zip(
        states, noisy, levels, strict=True
    ):
        single, pair = module.diffusion_conditioning(
            level,
            feat["relp"],
            s_inputs=s_input,
            s_trunk=s,
            z_trunk=z,
            pair_z=cache["pair_z"],
            inplace_safe=True,
            use_conditioning=True,
        )
        a, q, c, p = module.atom_attention_encoder(
            feat["atom_to_token_idx"],
            feat["ref_pos"],
            feat["ref_charge"],
            feat["ref_mask"],
            feat["ref_atom_name_chars"],
            feat["ref_element"],
            feat["d_lm"],
            feat["v_lm"],
            feat["pad_info"],
            r_l=x / torch.sqrt(module.sigma_data**2 + level**2)[..., None, None],
            s=s.unsqueeze(-3),
            z=pair.unsqueeze(-4),
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        atoms.append(a.float() + module.linear_no_bias_s(module.layernorm_s(single)))
        conditions.append(single.float())
        pair_conditions.append(pair.float())
        skips.append((q, c, p))
    lengths = [a.shape[-2] for a in atoms]
    z_batch = pad_token_tensors(pair_conditions, pair=True).unsqueeze(-4)
    _, bias = token_masks(lengths, z_batch)
    if projections is None:
        z_batch = module._prepare_diffusion_pair(z_batch, fusion)
    a_batch = module.diffusion_transformer(
        a=pad_token_tensors(atoms),
        s=pad_token_tensors(conditions),
        z=z_batch,
        inplace_safe=True,
        chunk_size=chunk_size,
        enable_efficient_fusion=fusion,
        projected_pair_biases=projections,
        extra_attn_bias=bias,
    )
    del z_batch, atoms, conditions, pair_conditions
    result = []
    for i, ((feat, *_), x, level, skip, n) in enumerate(
        zip(states, noisy, levels, skips, lengths, strict=True)
    ):
        update = module.atom_attention_decoder(
            atom_to_token_idx=feat["atom_to_token_idx"],
            a=module.layernorm_a(a_batch[i, :, :n]),
            q_skip=skip[0],
            c_skip=skip[1],
            p_skip=skip[2],
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        ratio = (level / module.sigma_data)[..., None, None].to(update.dtype)
        result.append(
            (
                1 / (1 + ratio**2) * x
                + level[..., None, None] / torch.sqrt(1 + ratio**2) * update
            ).to(update.dtype)
        )
    return result


def sample_diffusion_batch(
    states,
    denoise,
    *,
    schedules,
    seeds,
    n_sample,
    chunk_size,
    gamma0,
    gamma_min,
    noise_scale_lambda,
    step_scale_eta,
):
    """Same unguided Euler sampler as B=1, with independent unpadded RNG streams."""
    generators = [
        torch.Generator(device=state[1].device).manual_seed(seed)
        for state, seed in zip(states, seeds, strict=True)
    ]
    numpy_rngs = [np.random.default_rng(seed) for seed in seeds]
    outputs: list[list[torch.Tensor]] = [[] for _ in states]
    chunk_size = chunk_size or n_sample
    for start in range(0, n_sample, chunk_size):
        count = min(chunk_size, n_sample - start)
        coords = [
            schedule[0]
            * torch.randn(
                count,
                state[0]["atom_to_token_idx"].numel(),
                3,
                device=state[1].device,
                dtype=state[1].dtype,
                generator=generator,
            )
            for state, schedule, generator in zip(
                states, schedules, generators, strict=True
            )
        ]
        for step in range(len(schedules[0]) - 1):
            noisy, levels = [], []
            for i, (x, state, schedule, generator, rng) in enumerate(
                zip(coords, states, schedules, generators, numpy_rngs, strict=True)
            ):
                dtype = state[1].dtype
                x = (
                    centre_random_augmentation(
                        x, N_sample=1, torch_generator=generator, numpy_rng=rng
                    )
                    .squeeze(-3)
                    .to(dtype)
                )
                last, current = schedule[step], schedule[step + 1]
                gamma = float(gamma0) if current > gamma_min else 0
                level = last * (gamma + 1)
                noisy.append(
                    x
                    + noise_scale_lambda
                    * torch.sqrt(level**2 - last**2)
                    * torch.randn(
                        x.shape, device=x.device, dtype=dtype, generator=generator
                    )
                )
                levels.append(level.reshape(1).expand(count).to(dtype))
            denoised = denoise(noisy, levels)
            if len(denoised) != len(states):
                raise ValueError("Denoiser changed the number of jobs.")
            coords = [
                x
                + step_scale_eta
                * (schedule[step + 1] - level)[..., None, None]
                * ((x - predicted) / level[..., None, None])
                for x, predicted, level, schedule in zip(
                    noisy, denoised, levels, schedules, strict=True
                )
            ]
        for output, x in zip(outputs, coords, strict=True):
            output.append(x)
    return [torch.cat(output, dim=0) for output in outputs]


def forward_batch(model, features: list[dict[str, Any]], seeds: list[int]):
    """Return one complete prediction dictionary per independent complex."""
    from mol.model.mol import update_input_feature_dict

    if not features or len(features) != len(seeds):
        raise ValueError("Need at least one input and one seed per input")
    # Validate even a partial batch of one: it uses the same batched code path.
    validate_input_batching(model.configs, batch_size=max(2, len(features)))
    if model._device().type not in ("cpu", "cuda"):
        raise ValueError("Input batching supports CPU and CUDA")
    cfg = model.configs
    features = [
        update_input_feature_dict(
            model.relative_position_encoding.generate_relp(dict(feat))
        )
        for feat in features
    ]
    n_max = max(feat["token_index"].numel() for feat in features)
    chunk = (
        model._get_dynamic_chunk_size(n_max)
        if cfg.infer_setting.dynamic_chunk_size
        else cfg.infer_setting.chunk_size
    )
    # Bound triangle attention score work across inputs as well as query rows.
    budget_chunk = max(1, 450_000_000 // (len(features) * n_max * n_max))
    chunk = min(chunk or budget_chunk, budget_chunk)
    trunks = _trunk_batch(model, features, chunk)
    states, schedules = [], []
    for feat, (s_input, s, z) in zip(features, trunks, strict=True):
        cache = model.prepare_inference_cache(feat, z)
        states.append((feat, s_input, s, z, cache))
        schedules.append(
            model.inference_noise_scheduler(
                N_step=cfg.sample_diffusion.N_step,
                device=s_input.device,
                dtype=s_input.dtype,
            )
        )

    def sample(states, schedules):
        sample_states, sample_schedules = states, schedules
        if cfg.skip_amp.sample_diffusion:
            sample_states = []
            for feat, s_input, s, z, cache in states:
                local_cache = {
                    "pair_z": None
                    if cache["pair_z"] is None
                    else cache["pair_z"].float(),
                    "p_lm/c_l": [
                        None if t is None else t.float() for t in cache["p_lm/c_l"]
                    ],
                }
                sample_states.append(
                    (feat, s_input.float(), s.float(), z.float(), local_cache)
                )
            sample_schedules = [t.float() for t in schedules]
        projections = None
        if model._inference_diffusion_cache_mode() == "full":
            projections = model._prepare_projected_pair_biases(
                pad_token_tensors(
                    [state[4]["pair_z"] for state in sample_states], pair=True
                )
            )
        return sample_diffusion_batch(
            sample_states,
            lambda x, t: _denoise_batch(
                model.diffusion_module,
                sample_states,
                x,
                t,
                chunk_size=chunk,
                fusion=model.enable_efficient_fusion,
                projections=projections,
            ),
            schedules=sample_schedules,
            seeds=seeds,
            n_sample=cfg.sample_diffusion.N_sample,
            chunk_size=cfg.infer_setting.sample_diffusion_chunk_size,
            **{
                key: cfg.sample_diffusion[key]
                for key in (
                    "gamma0",
                    "gamma_min",
                    "noise_scale_lambda",
                    "step_scale_eta",
                )
            },
        )

    coordinates = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion)(sample)(
        states, schedules
    )
    del states, schedules, cache
    predictions = []
    for feat, (s_input, s, z), coordinate in zip(
        features, trunks, coordinates, strict=True
    ):
        pred = model.run_inference_heads(
            feat,
            s_input,
            s,
            z,
            coordinate,
            chunk,
            kernels=dict(triangle_multiplicative="torch", triangle_attention="torch"),
            inplace_safe=True,
        )
        predictions.append(model.summarize_prediction(pred, feat, model.N_cycle))
    return predictions
