# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""All ranks process the same input; only rank zero featurizes and writes files."""

import copy
import time

import torch
import torch.distributed as dist

from mol.data.inference.infer_dataloader import InferenceDataset
from mol.runner.batched_prediction import _job_seeds, validate_prediction
from mol.utils.logger import get_logger
from mol.utils.seed import seed_everything


logger = get_logger(__name__)


def rank_zero_action(action):
    """Broadcast a trusted local result or an error before peers move on."""
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (True, action())
        except Exception as exc:
            payload[0] = (False, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(payload, src=0)
    success, result = payload[0]
    if not success:
        raise RuntimeError(f"Context-parallel rank-zero operation failed: {result}")
    return result


def infer_predict_context_parallel(runner, configs):
    from mol.runner.inference import (
        fix_cterminal_carboxyl_oxygens,
        update_inference_configs,
    )

    owned = {}

    def prepare():
        dataset = InferenceDataset(configs)
        if not isinstance(dataset.inputs, list) or not dataset.inputs:
            raise ValueError("Input JSON must be a non-empty job list")
        names = [job.get("name") for job in dataset.inputs]
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("Each input needs a unique, non-empty name")
        owned["dataset"] = dataset
        return [(job["name"], _job_seeds(job, configs)) for job in dataset.inputs]

    jobs = rank_zero_action(prepare)
    for index, (name, seeds) in enumerate(jobs):
        for seed in seeds:
            seed_everything(seed, deterministic=configs.deterministic)

            def load():
                data, atoms, error = owned["dataset"][index]
                if error:
                    raise ValueError(error)
                owned["atoms"] = atoms
                data["input_feature_dict"]["inference_seed"] = torch.tensor(seed)
                return data

            data = rank_zero_action(load)
            n_token = int(data["N_token"].item())
            runner.update_model_configs(
                update_inference_configs(copy.deepcopy(configs), n_token)
            )
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            started = time.monotonic()
            # Distributed failures are fatal: never let one rank continue to a
            # new job while another is still in the failed job's collectives.
            prediction = runner.predict(data)

            def save(prediction=prediction, data=data):
                atoms = owned["atoms"]
                prediction["coordinate"] = fix_cterminal_carboxyl_oxygens(
                    prediction["coordinate"], atoms
                )
                validate_prediction(
                    prediction,
                    n_atom=len(atoms),
                    n_token=n_token,
                    n_sample=configs.sample_diffusion.N_sample,
                )
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atoms,
                    ipsae_interface=data.get("ipsae_interface"),
                    entity_poly_type={
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )

            rank_zero_action(save)
            del save
            peak = (
                torch.cuda.max_memory_allocated(runner.device) / 2**30
                if runner.device.type == "cuda"
                else 0
            )
            logger.info(
                "[Rank %d] Shared prediction %s seed %d: %.2fs, peak %.3f GiB, partitions %s",
                dist.get_rank(),
                name,
                seed,
                time.monotonic() - started,
                peak,
                runner.last_context_parallel_stats,
            )
            del prediction, data
