# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Fast initialization must preserve complete checkpoint and seeded predictions."""

import copy
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch

from mol.model.mol import Mol
from mol.model.triangular import layers
from mol.runner.inference import InferenceRunner
from mol.utils.seed import seed_everything
from tests.test_input_batching import activate_residuals, tiny_config, tiny_features


def test_skip_context_is_nested_thread_local_and_restored_after_exception():
    def initialize():
        weights = torch.full((3, 3), 123.0)
        layers.trunc_normal_init_(weights)
        return weights

    with patch.object(layers.truncnorm, "rvs", wraps=layers.truncnorm.rvs) as draws:
        with pytest.raises(RuntimeError, match="construction failed"):
            with layers.skip_trunc_normal_init():
                with layers.skip_trunc_normal_init():
                    assert torch.all(initialize() == 123)
                assert torch.all(initialize() == 123)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    assert not torch.all(executor.submit(initialize).result() == 123)
                assert draws.call_count == 1
                raise RuntimeError("construction failed")
        assert not torch.all(initialize() == 123)
        assert draws.call_count == 2


@pytest.fixture
def checkpoint_config(tmp_path):
    config = tiny_config()
    config.device = "cpu"
    config.dump_dir = str(tmp_path)
    config.load_checkpoint_path = str(tmp_path / "tiny.pt")
    config.load_strict = True
    seed_everything(19, deterministic=False)
    model = Mol(config)
    activate_residuals(model)
    torch.save({"model": model.state_dict()}, config.load_checkpoint_path)
    return config


@pytest.mark.parametrize("ddp_prefix", [False, True])
def test_checkpoint_and_seeded_predictions_match_exactly(checkpoint_config, ddp_prefix):
    config = checkpoint_config
    saved = torch.load(config.load_checkpoint_path, weights_only=False)["model"]
    if ddp_prefix:
        torch.save(
            {"model": {"module." + key: value for key, value in saved.items()}},
            config.load_checkpoint_path,
        )
    predictions = []
    for fast_init in (False, True):
        config.infer_setting.fast_init = fast_init
        seed_everything(23, deterministic=False)
        with patch.object(layers.truncnorm, "rvs", wraps=layers.truncnorm.rvs) as draws:
            runner = InferenceRunner(copy.deepcopy(config))
        assert (draws.call_count == 0) == fast_init
        assert runner.model_init_seconds >= 0
        assert runner.checkpoint_load_seconds >= 0
        assert runner.checkpoint_loaded
        assert runner.checkpoint_load_report == {
            "path": config.load_checkpoint_path,
            "strict": True,
            "complete": True,
            "loaded_tensors": len(saved),
            "model_tensors": len(saved),
            "missing_keys": [],
            "unexpected_keys": [],
        }
        torch.testing.assert_close(runner.model.state_dict(), saved, rtol=0, atol=0)
        seed_everything(101, deterministic=False)
        predictions.append(runner.predict({"input_feature_dict": tiny_features(5, 101)}))
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)


def test_fast_init_rejects_partial_loading_before_model_construction(checkpoint_config):
    config = checkpoint_config
    config.infer_setting.fast_init = True
    config.load_strict = False
    with patch("mol.runner.inference.Mol") as build:
        with pytest.raises(ValueError, match="requires load_strict=true"):
            InferenceRunner(config)
    build.assert_not_called()


def test_missing_checkpoint_weight_still_fails_and_restores_initialization(checkpoint_config):
    config = checkpoint_config
    saved = torch.load(config.load_checkpoint_path, weights_only=False)
    saved["model"].pop(next(iter(saved["model"])))
    torch.save(saved, config.load_checkpoint_path)
    config.infer_setting.fast_init = True
    with pytest.raises(RuntimeError, match="Missing key"):
        InferenceRunner(config)
    with patch.object(layers.truncnorm, "rvs", wraps=layers.truncnorm.rvs) as draws:
        Mol(config)  # Ordinary model/training construction retains initialization.
    assert draws.call_count > 0


def test_changing_config_cannot_enable_partial_load_after_fast_construction(checkpoint_config):
    config = checkpoint_config
    config.infer_setting.fast_init = True
    runner = InferenceRunner(config)
    config.infer_setting.fast_init = False
    config.load_strict = False
    with pytest.raises(ValueError, match="requires load_strict=true"):
        runner.load_checkpoint()
    assert not runner.checkpoint_loaded


def assert_prediction_blocked(runner):
    assert not runner.checkpoint_loaded
    with pytest.raises(RuntimeError, match="not ready for prediction"):
        runner.predict({})
    with pytest.raises(RuntimeError, match="not ready for prediction"):
        runner.predict_batch([], [])


def test_rebuilding_or_replacing_model_requires_loading_checkpoint(checkpoint_config):
    config = checkpoint_config
    config.infer_setting.fast_init = True
    runner = InferenceRunner(config)
    runner.init_model()
    assert_prediction_blocked(runner)
    runner.load_checkpoint()
    assert runner.checkpoint_loaded
    runner.model = Mol(config)
    assert_prediction_blocked(runner)


@pytest.mark.parametrize("problem, message", [
    ("missing", "Missing key"),
    ("unexpected", "Unexpected key"),
    ("shape", "Shape mismatch"),
    ("empty", "nonempty 'model'"),
    ("malformed", "nonempty 'model'"),
    ("non_tensor", "values must be tensors"),
    ("mixed_prefix", "mixes prefixed"),
])
def test_invalid_checkpoint_is_rejected_before_copy_and_can_be_recovered(
    checkpoint_config, problem, message
):
    config = checkpoint_config
    config.infer_setting.fast_init = True
    runner = InferenceRunner(config)
    original = copy.deepcopy(runner.model.state_dict())
    broken = copy.deepcopy(original)
    first, last = next(iter(broken)), next(reversed(broken))
    broken[first] = broken[first] + 7
    if problem == "missing":
        del broken[last]
    elif problem == "unexpected":
        broken["extra_weight"] = torch.ones(1)
    elif problem == "shape":
        broken[last] = torch.ones(broken[last].numel() + 1)
    elif problem == "empty":
        broken = {}
    elif problem == "malformed":
        broken = []
    elif problem == "non_tensor":
        broken[last] = "not a tensor"
    elif problem == "mixed_prefix":
        broken["module." + last] = broken.pop(last)
    torch.save({"model": broken}, config.load_checkpoint_path)
    with pytest.raises((RuntimeError, ValueError), match=message):
        runner.load_checkpoint()
    torch.testing.assert_close(runner.model.state_dict(), original, rtol=0, atol=0)
    assert runner.checkpoint_load_report is None
    assert_prediction_blocked(runner)

    torch.save({"model": original}, config.load_checkpoint_path)
    runner.load_checkpoint()
    assert runner.checkpoint_loaded and runner.checkpoint_load_report["complete"]


def test_copy_failure_and_missing_file_invalidate_previous_success(checkpoint_config):
    config = checkpoint_config
    config.infer_setting.fast_init = True
    runner = InferenceRunner(config)

    def fail_copy(**kwargs):
        with torch.no_grad():
            next(runner.model.parameters()).add_(100)
        raise RuntimeError("copy failed")

    with patch.object(runner.model, "load_state_dict", side_effect=fail_copy):
        with pytest.raises(RuntimeError, match="copy failed"):
            runner.load_checkpoint()
    assert_prediction_blocked(runner)
    runner.load_checkpoint()
    assert runner.checkpoint_loaded

    config.load_checkpoint_path += ".missing"
    with pytest.raises(FileNotFoundError):
        runner.load_checkpoint()
    assert_prediction_blocked(runner)


def test_explicit_non_strict_load_reports_partial_coverage(checkpoint_config, caplog):
    config = checkpoint_config
    config.infer_setting.fast_init = False
    config.load_strict = False
    saved = torch.load(config.load_checkpoint_path, weights_only=False)
    missing = next(iter(saved["model"]))
    del saved["model"][missing]
    torch.save(saved, config.load_checkpoint_path)
    runner = InferenceRunner(config)
    assert runner.checkpoint_loaded
    assert not runner.checkpoint_load_report["complete"]
    assert runner.checkpoint_load_report["missing_keys"] == [missing]
    assert runner.checkpoint_load_report["loaded_tensors"] == len(saved["model"])
    assert "Partial checkpoint accepted" in caplog.text
