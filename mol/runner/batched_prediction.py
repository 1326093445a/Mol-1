# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Schedule independent no-MSA complexes and retain per-job output ownership."""

import copy
import os
import time
import traceback

import torch

from mol.config.inference import validate_input_batching
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.utils.distributed import DIST_WRAPPER
from mol.utils.logger import get_logger
from mol.utils.seed import seed_everything

logger = get_logger(__name__)


def validate_prediction(prediction, *, n_atom, n_token, n_sample):
    """Check axes explicitly before associating a result with an input's CIF."""
    shapes = {
        "coordinate": (n_sample, n_atom, 3),
        "contact_probs": (n_token, n_token),
    }
    for key, shape in shapes.items():
        if tuple(prediction[key].shape) != shape:
            raise ValueError(
                f"{key} has shape {tuple(prediction[key].shape)}, expected {shape}"
            )
    for key, prefix in (
        ("plddt", (n_sample, n_atom)),
        ("resolved", (n_sample, n_atom)),
        ("pae", (n_sample, n_token, n_token)),
        ("pde", (n_sample, n_token, n_token)),
    ):
        value = prediction[key]
        if value.ndim != len(prefix) + 1 or tuple(value.shape[:-1]) != prefix:
            raise ValueError(f"{key} has incorrect per-input/sample axes")
    for key in ("coordinate", "plddt", "pae", "pde", "resolved", "contact_probs"):
        if not torch.isfinite(prediction[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("summary_confidence", "full_data"):
        if len(prediction[key]) != n_sample:
            raise ValueError(f"{key} must contain one entry per diffusion sample")
    for full in prediction["full_data"]:
        for key, shape in (
            ("atom_plddt", (n_atom,)),
            ("atom_coordinate", (n_atom, 3)),
            ("token_pair_pae", (n_token, n_token)),
            ("token_pair_pde", (n_token, n_token)),
            ("token_has_frame", (n_token,)),
            ("token_asym_id", (n_token,)),
            ("atom_to_token_idx", (n_atom,)),
            ("atom_is_polymer", (n_atom,)),
        ):
            if key in full and tuple(full[key].shape) != shape:
                raise ValueError(f"full_data.{key} has incorrect per-input axes")


def _precision_bucket(n_token):
    # Same boundaries as update_inference_configs, so a larger neighbour
    # cannot silently change an input's precision policy.
    return 0 if n_token <= 2560 else 1 if n_token <= 3840 else 2


def _job_seeds(job, configs):
    seeds = job.get("modelSeeds") if configs.use_seeds_in_json else None
    seeds = list(configs.seeds) if seeds is None else seeds
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError(f"{job['name']}: expected a non-empty seed list")
    seeds = [int(seed) for seed in seeds]
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError(f"{job['name']}: seeds must be in [0, 2**32)")
    return list(dict.fromkeys(seeds))


def infer_predict_batched(runner, configs):
    from mol.runner.inference import (
        fix_cterminal_carboxyl_oxygens,
        update_inference_configs,
    )

    validate_input_batching(configs)
    dataset = InferenceDataset(configs)
    jobs = dataset.inputs
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Input JSON must be a non-empty job list")
    names = [job.get("name") for job in jobs]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Each batched input needs a unique, non-empty name")
    schedules = [_job_seeds(job, configs) for job in jobs]
    os.makedirs(runner.error_dir, exist_ok=True)
    failures = []

    def report(data, seed, error):
        name = data["sample_name"]
        message = f"{name} [seed:{seed}]: {error}"
        failures.append(message)
        logger.error(message)
        with open(
            os.path.join(runner.error_dir, f"{name}.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(message + "\n")

    def cleanup():
        if runner.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(items):
        started = time.monotonic()
        predictions = None
        retry = False
        try:
            local_configs = update_inference_configs(
                copy.deepcopy(configs),
                max(int(data["N_token"].item()) for data, _, _ in items),
            )
            runner.update_model_configs(local_configs)
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            predictions = runner.predict_batch(
                [data for data, _, _ in items], [seed for _, _, seed in items]
            )
            if len(predictions) != len(items):
                raise ValueError(
                    "Model returned an incorrect number of input predictions"
                )
        except torch.cuda.OutOfMemoryError as exc:
            exc.__traceback__ = (
                None  # release failed-forward intermediates before retrying
            )
            if len(items) > 1:
                retry = True
                logger.warning(
                    "Batch of %d exceeded memory; retrying smaller batches", len(items)
                )
            else:
                report(items[0][0], items[0][2], exc)
        except Exception as exc:
            predictions = None
            detail = traceback.format_exc()
            for data, _, seed in items:
                report(data, seed, detail)
            exc.__traceback__ = None
        if retry:
            cleanup()
            mid = len(items) // 2
            predict(items[:mid])
            predict(items[mid:])
            return
        if predictions is not None:
            completed = 0
            for (data, atom_array, seed), pred in zip(items, predictions, strict=True):
                try:
                    pred["coordinate"] = fix_cterminal_carboxyl_oxygens(
                        pred["coordinate"], atom_array
                    )
                    validate_prediction(
                        pred,
                        n_atom=len(atom_array),
                        n_token=int(data["N_token"].item()),
                        n_sample=configs.sample_diffusion.N_sample,
                    )
                    runner.dumper.dump(
                        dataset_name="",
                        pdb_id=data["sample_name"],
                        seed=seed,
                        pred_dict=pred,
                        atom_array=atom_array,
                        entity_poly_type={
                            k: v
                            for k, v in data["entity_poly_type"].items()
                            if v != "non-polymer"
                        },
                    )
                    completed += 1
                except Exception:
                    report(data, seed, traceback.format_exc())
            del pred
            logger.info(
                "[Rank %d] %d/%d inputs, %d samples/input, %.2fs including output",
                DIST_WRAPPER.rank,
                completed,
                len(items),
                configs.sample_diffusion.N_sample,
                time.monotonic() - started,
            )
            if runner.device.type == "cuda":
                logger.info(
                    "Peak allocated GPU memory: %.3f GiB",
                    torch.cuda.max_memory_allocated(runner.device) / 2**30,
                )
        del predictions
        cleanup()

    def estimated_size(index):
        return sum(
            len(next(iter(entity.values())).get("sequence", ""))
            * next(iter(entity.values())).get("count", 1)
            for entity in jobs[index]["sequences"]
        )

    # Shard without padding/duplication. Each rank owns whole jobs, including
    # all of a job's seeds. Sorting reduces wasted quadratic pair padding.
    indices = sorted(range(len(jobs)), key=estimated_size)
    owned = indices[DIST_WRAPPER.rank :: DIST_WRAPPER.world_size]
    pending, bucket = [], None
    for seed in dict.fromkeys(seed for seeds in schedules for seed in seeds):
        for index in owned:
            if seed not in schedules[index]:
                continue
            seed_everything(seed, deterministic=configs.deterministic)
            data, atom_array, error = dataset[index]
            if error:
                report(data, seed, error)
                continue
            current = _precision_bucket(int(data["N_token"].item()))
            if pending and current != bucket:
                predict(pending)
                pending = []
            bucket = current
            pending.append((data, atom_array, seed))
            if len(pending) == configs.batch_size:
                predict(pending)
                pending = []
    if pending:
        predict(pending)
    if failures:
        raise RuntimeError(
            f"{len(failures)} input/seed prediction(s) failed. First: {failures[0]}"
        )
