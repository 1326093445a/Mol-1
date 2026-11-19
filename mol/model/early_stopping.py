# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""First-generated-sample screening with a fixed input-batch shape."""

import random
from contextlib import contextmanager

import numpy as np
import torch

from mol.config.inference import validate_early_stopping
from mol.metrics.ipsae import ipsae_summary
from mol.model import sample_confidence
from mol.utils.torch_utils import autocasting_disable_decorator


@contextmanager
def preserve_rng(device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def make_first_sample_gate(model, features, trunks, chunk, *, inplace_safe):
    threshold = model.configs.infer_setting.get("early_stop_ipsae_threshold")
    if threshold is None or model.training or model.configs.sample_diffusion.N_sample <= 1:
        return None
    validate_early_stopping(model.configs)
    if any("ipsae_early_stop" not in feat for feat in features):
        raise ValueError("ipSAE early stopping requires CA-token groups and interface metadata from InferenceDataset")
    return FirstSampleGate(model, features, trunks, chunk, threshold, inplace_safe)


class FirstSampleGate:
    def __init__(self, model, features, trunks, chunk, threshold, inplace_safe):
        self.model, self.features, self.trunks = model, features, trunks
        self.chunk, self.threshold, self.inplace_safe = chunk, float(threshold), inplace_safe
        self.requested = model.configs.sample_diffusion.N_sample
        self.first = [None] * len(features)
        self.scores = [None] * len(features)
        self.stopped = False
        self.device = trunks[0][0].device
        # The callback runs inside the sampler's precision context. Confidence
        # and distogram must instead use the original outer prediction context.
        self.amp_enabled = torch.is_autocast_enabled(self.device.type)
        self.amp_dtype = torch.get_autocast_dtype(self.device.type)

    def heads(self, i, coordinate):
        return self.model.run_inference_heads(
            self.features[i], *self.trunks[i], coordinate, self.chunk,
            inplace_safe=self.inplace_safe,
        )

    def __call__(self, coordinates):
        with preserve_rng(self.device), torch.autocast(
            self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            for i, (feat, coordinate) in enumerate(zip(self.features, coordinates, strict=True)):
                metadata = feat["ipsae_early_stop"]
                groups, interface = metadata["groups"], metadata.get("interface")
                if interface is None and len(groups) != 2:
                    continue  # Undefined ipSAE is never interpreted as zero.
                pred = self.heads(i, coordinate)
                self.first[i] = pred
                pae = autocasting_disable_decorator(True)(sample_confidence.logits_to_score)(
                    pred["pae"].to(pred["plddt"].device),
                    **sample_confidence.get_bin_params(self.model.configs.loss.pae),
                )[0]
                self.scores[i] = ipsae_summary(pae, groups, interface=interface)["ipsae_min_10"]
        self.stopped = all(
            score is not None and (score == 0 or score < self.threshold)
            for score in self.scores
        )
        return self.stopped

    def finish_heads(self, i, coordinate):
        """Reuse scored heads; later samples keep their original chunk size."""
        first = self.first[i]
        if first is None:
            return self.heads(i, coordinate)
        if coordinate.shape[0] == 1:
            return first
        rest = self.heads(i, coordinate[1:])
        result = {"coordinate": coordinate, "contact_probs": first["contact_probs"]}
        for key in ("plddt", "pae", "pde", "resolved"):
            result[key] = torch.cat((first[key], rest[key]), dim=0)
        return result

    def annotate(self, prediction, i, *, batch_policy="preserve_shape_all_inputs", batch_size=None):
        completed = prediction["coordinate"].shape[0]
        score = self.scores[i]
        reason = (
            "first_sample_below_threshold" if self.stopped else
            "ipsae_unavailable" if score is None else
            "batch_not_all_below_threshold" if len(self.features) > 1 else
            "first_sample_passed"
        )
        info = {
            "metric": "ipsae_min_10", "threshold": self.threshold,
            "first_sample_score": score, "first_sample_generation_index": 0,
            "batch_policy": batch_policy, "batch_size": batch_size or len(self.features),
            "requested_samples": self.requested, "completed_samples": completed,
            "skipped_samples": self.requested - completed,
            "stopped": self.stopped, "reason": reason,
        }
        prediction["sampling_early_stop"] = info
        for index, summary in enumerate(prediction["summary_confidence"]):
            summary["sample_generation_index"] = index
            summary["sampling_early_stop"] = dict(info)
        return prediction
