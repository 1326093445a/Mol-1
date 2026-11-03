# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: configs/configs_inference.py

# pylint: disable=C0114
import math
import os
from pathlib import Path

from mol.config.base import model_configs as base_model_configs
from mol.config.types import ListValue, RequiredValue, ValueMaybeNone

MOL_ROOT_DIR = os.environ.get("MOL_ROOT_DIR", str(Path.home()))
inference_configs = {
    "model_name": "mol-1-base",  # inference model selection
    # auto prefers CUDA, then Apple MPS, then CPU. An explicit device that is
    # unavailable is an error, not a silent downgrade to CPU.
    "device": "auto",  # auto | cuda | cuda:N | mps | cpu
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": os.path.join(MOL_ROOT_DIR, "checkpoint"),
    "num_workers": 0,
    "batch_size": 1,  # independent complexes per forward, separate from N_sample
    "context_parallel_size": 1,  # GPUs cooperating on the same complex
    "use_msa": True,
    # FP32 matrix multiplications stay FP32; true runs them on TF32 tensor
    # cores, 4-8% faster with slightly different results.
    "enable_tf32": False,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "diffusion_cache_mode": "full",  # full | shared | off
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    "use_seeds_in_json": False,
    # Where prediction differs from evaluation during training: no Monte Carlo
    # dropout in the recycles, and one diffusion sample at a time.
    "mc_dropout_apply_rate": 0.0,
    "mc_dropout_rate": 0.0,
    "infer_setting": {
        **base_model_configs["infer_setting"],
        "sample_diffusion_chunk_size": ValueMaybeNone(1),
        "confidence_sample_chunk_size": ValueMaybeNone(1),
        # Opt-in until full-model throughput is measured without GPU contention.
        "atom_block_cache": False,
        # Zero disables either scheduling limit. These never alter N_sample.
        "batch_max_padding_ratio": 0.0,
        "batch_max_pair_tokens": 0,
    },
}


def validate_input_batching(configs, *, batch_size=None):
    """Check the supported no-MSA inference mode before loading model assets."""
    size = configs.get("batch_size", 1) if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch_size must be a positive integer")
    if size == 1:
        return
    ratio = configs.infer_setting.get("batch_max_padding_ratio", 0.0)
    if not math.isfinite(ratio) or (ratio != 0 and ratio < 1):
        raise ValueError("batch_max_padding_ratio must be zero (disabled) or >= 1")
    budget = configs.infer_setting.get("batch_max_pair_tokens", 0)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("batch_max_pair_tokens must be a non-negative integer")
    unsupported = []
    for key in ("use_msa", "use_template", "use_rna_msa"):
        if configs.get(key, False):
            unsupported.append(f"{key}=true")
    for key, supported in (
        (
            "triangle_multiplicative",
            ("auto", "torch", "cuequivariance", "flashpairformer"),
        ),
        ("triangle_attention", ("auto", "torch", "cuequivariance")),
    ):
        if configs[key] not in supported:
            unsupported.append(f"{key}={configs[key]} (choose {', '.join(supported)})")
    if configs.model.N_model_seed != 1:
        unsupported.append("model.N_model_seed != 1")
    if configs.get("mc_dropout_apply_rate", 0) != 0:
        unsupported.append("mc_dropout_apply_rate != 0")
    if configs.get("num_workers", 0) != 0:
        unsupported.append("num_workers != 0 (per-job featurization seeding)")
    if configs.get("device", "auto").startswith("mps"):
        unsupported.append("MPS (batching supports CPU and CUDA)")
    if unsupported:
        raise ValueError("batch_size > 1 does not support: " + ", ".join(unsupported))
    for name, value in (
        ("model.N_cycle", configs.model.N_cycle),
        ("sample_diffusion.N_step", configs.sample_diffusion.N_step),
        ("sample_diffusion.N_sample", configs.sample_diffusion.N_sample),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")


def validate_context_parallel(configs):
    """Validate the no-MSA, one-complex torchrun mode before loading assets."""
    size = configs.get("context_parallel_size", 1)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("context_parallel_size must be a positive integer")
    if size == 1:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != size:
        raise ValueError(
            "context_parallel_size must equal torchrun WORLD_SIZE; "
            "launch with --nproc_per_node matching this value"
        )
    if configs.get("batch_size", 1) != 1:
        raise ValueError(
            "context parallelism requires batch_size=1; "
            "all GPUs work on the same complex"
        )
    # Feature restrictions are shared; distributed stacks still require torch.
    validate_input_batching(configs, batch_size=2)
    for key in ("triangle_multiplicative", "triangle_attention"):
        if configs[key] not in ("auto", "torch"):
            raise ValueError(f"context parallelism requires {key}=auto or torch")
    if configs.dtype not in ("fp32", "bf16"):
        raise ValueError("context parallelism supports fp32 and bf16")
    if configs.get("device", "auto") not in ("auto", "cuda", "cpu"):
        raise ValueError(
            "Use device=auto or cuda and CUDA_VISIBLE_DEVICES with torchrun, "
            "not an explicit device index"
        )
