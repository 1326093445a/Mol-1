# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Runner-owned inference acceleration has bounded, explicit lifecycle rules."""

import os
import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mol.config.inference import validate_inference_acceleration
from mol.runner.inference import InferenceRunner
from tests.test_fast_init import checkpoint_config
from tests.test_input_batching import tiny_features


@pytest.fixture
def acceleration_runner(checkpoint_config):
    checkpoint_config.infer_setting.compile_diffusion = True
    checkpoint_config.infer_setting.cuda_graphs = True
    runner = InferenceRunner(checkpoint_config)
    runner.device = torch.device("cuda")  # Exercise lifecycle without allocating CUDA tensors.
    calls, closed = [], []

    @contextmanager
    def context(transformer, **options):
        session = SimpleNamespace(fused_layer_norm_modules=0, graph_cache_entries=0)
        transformer._compilation_session = session
        calls.append(options)
        try:
            yield session
        finally:
            del transformer._compilation_session
            closed.append(session)

    with patch("mol.model.compiled_diffusion.compiled_diffusion_blocks", side_effect=context):
        yield runner, calls, closed
        runner.close()


def test_default_cuda_runner_replays_eager_without_compilation(checkpoint_config):
    runner = InferenceRunner(checkpoint_config)
    runner.device = torch.device("cuda")
    transformer = runner.model.diffusion_module.diffusion_transformer
    calls, closed = [], []

    @contextmanager
    def replay_context(module, **options):
        session = SimpleNamespace(fused_layer_norm_modules=0)
        module._compilation_session = session
        calls.append(options)
        try:
            yield session
        finally:
            del module._compilation_session
            closed.append(session)

    try:
        with patch("torch.compile", side_effect=AssertionError("Unexpected compilation")), \
             patch("mol.model.compiled_diffusion.compiled_diffusion_blocks") as compile_blocks, \
             patch("mol.model.cuda_graph_replay.eager_diffusion_replay", side_effect=replay_context):
            assert not runner.configs.infer_setting.compile_diffusion
            assert runner.configs.infer_setting.cuda_graphs
            runner._ensure_diffusion_acceleration()
            session = runner.diffusion_compilation_session
            runner._ensure_diffusion_acceleration()
            assert calls == [{"max_graphs": 2}] and not closed
            assert runner.diffusion_compilation_session is session
            runner.configs.infer_setting.cuda_graphs = False
            runner._ensure_diffusion_acceleration()
            assert closed == [session]
        compile_blocks.assert_not_called()
        assert runner.diffusion_compilation_session is None
        assert not hasattr(transformer, "_compilation_session")
    finally:
        runner.close()


def test_switching_compilation_preserves_independent_replay_control(acceleration_runner):
    runner, calls, closed = acceleration_runner
    eager_calls, eager_closed = [], []

    @contextmanager
    def eager_context(module, **options):
        session = SimpleNamespace(fused_layer_norm_modules=0)
        module._compilation_session = session
        eager_calls.append(options)
        try:
            yield session
        finally:
            del module._compilation_session
            eager_closed.append(session)

    with patch("mol.model.cuda_graph_replay.eager_diffusion_replay", side_effect=eager_context):
        runner._ensure_diffusion_acceleration()
        runner.configs.infer_setting.compile_diffusion = False
        runner._ensure_diffusion_acceleration()
        assert len(closed) == 1 and eager_calls == [{"max_graphs": 2}]
        runner.configs.infer_setting.compile_diffusion = True
        runner._ensure_diffusion_acceleration()
        assert len(eager_closed) == 1 and len(calls) == 2


def test_compile_only_does_not_implicitly_enable_replay(acceleration_runner):
    runner, calls, _ = acceleration_runner
    del runner.configs.infer_setting["cuda_graphs"]
    runner._ensure_diffusion_acceleration()
    assert calls[0]["cuda_graphs"] is False


def test_runner_reuses_session_and_can_close_or_disable_it(acceleration_runner):
    runner, calls, closed = acceleration_runner
    runner._ensure_diffusion_acceleration()
    session = runner.diffusion_compilation_session
    runner._ensure_diffusion_acceleration()
    assert len(calls) == 1 and runner.diffusion_compilation_session is session
    assert calls[0] == {"cuda_graphs": True, "max_graphs": 2, "dynamic_shapes": True, "compile_threads": 1}
    runner.configs.infer_setting.cuda_graphs = False
    runner._ensure_diffusion_acceleration()
    assert closed == [session] and len(calls) == 2
    runner.configs.infer_setting.compile_diffusion = False
    runner._ensure_diffusion_acceleration()
    assert len(closed) == 2 and runner.diffusion_compilation_session is None
    runner.close()
    assert len(closed) == 2


def test_reload_and_prediction_failure_release_sessions(acceleration_runner):
    runner, calls, closed = acceleration_runner
    runner._ensure_diffusion_acceleration()
    runner.device = torch.device("cpu")
    runner.load_checkpoint()
    assert len(closed) == 1 and runner.checkpoint_loaded
    runner.device = torch.device("cuda")
    with pytest.raises(RuntimeError, match="prediction failure"):
        with runner._accelerated_prediction():
            raise RuntimeError("prediction failure")
    assert len(closed) == 2 and runner.diffusion_compilation_session is None


def test_input_transfer_failure_releases_retained_graphs(acceleration_runner):
    runner, _, closed = acceleration_runner
    runner._ensure_diffusion_acceleration()
    with patch("mol.runner.inference.to_device", side_effect=torch.cuda.OutOfMemoryError("input transfer")):
        with pytest.raises(torch.cuda.OutOfMemoryError, match="input transfer"):
            runner.predict({})
    assert len(closed) == 1 and runner.diffusion_compilation_session is None
    assert runner.checkpoint_loaded


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_cpu_and_context_parallel_do_not_install_cuda_acceleration(acceleration_runner, compile_enabled):
    runner, calls, _ = acceleration_runner
    runner.configs.infer_setting.compile_diffusion = compile_enabled
    runner.device = torch.device("cpu")
    runner._ensure_diffusion_acceleration()
    runner.device = torch.device("mps")
    runner._ensure_diffusion_acceleration()
    runner.device = torch.device("cuda")
    runner.configs.context_parallel_size = 2
    runner._ensure_diffusion_acceleration()
    assert not calls


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_explicit_external_context_keeps_ownership(acceleration_runner, compile_enabled):
    runner, calls, _ = acceleration_runner
    runner.configs.infer_setting.compile_diffusion = compile_enabled
    transformer = runner.model.diffusion_module.diffusion_transformer
    external = object()
    transformer._compilation_session = external
    runner._ensure_diffusion_acceleration()
    runner.close()
    assert not calls and transformer._compilation_session is external
    del transformer._compilation_session


def test_discarding_runner_restores_model_methods(checkpoint_config):
    checkpoint_config.infer_setting.compile_diffusion = True
    runner = InferenceRunner(checkpoint_config)
    runner.device = torch.device("cuda")
    transformer = runner.model.diffusion_module.diffusion_transformer
    closed = []

    @contextmanager
    def context(module, **options):
        session = SimpleNamespace(fused_layer_norm_modules=0)
        module._compilation_session = session
        try:
            yield session
        finally:
            del module._compilation_session
            closed.append(True)

    with patch("mol.model.compiled_diffusion.compiled_diffusion_blocks", side_effect=context):
        runner._ensure_diffusion_acceleration()
    reference = weakref.ref(runner)
    del runner
    gc.collect()
    assert reference() is None and closed == [True]
    assert not hasattr(transformer, "_compilation_session")


@pytest.mark.parametrize("name,value", [("compile_threads", 0), ("cuda_graph_max_cache_size", True), ("cpu_thread_limit", -1)])
def test_direct_api_invalid_limits_fail_before_model_loading(name, value):
    with pytest.raises(ValueError, match=name):
        validate_inference_acceleration({"infer_setting": {name: value}})


@pytest.mark.skipif(os.environ.get("MOL_RUN_CUDA_GRAPH_TESTS") != "1", reason="Explicit CUDA runner integration test")
def test_cuda_opt_in_replay_across_requests_and_reload(checkpoint_config):
    config = checkpoint_config
    config.infer_setting.compile_diffusion = True
    config.infer_setting.cuda_graphs = True
    config.device = "cuda"
    config.model.N_cycle = 1
    config.sample_diffusion.N_step = 2
    config.sample_diffusion.N_sample = 1
    runner = InferenceRunner(config)
    reference = None
    try:
        # More than Dynamo's default eight static specializations. Graph capacity
        # stays two; other lengths must continue through compiled execution.
        for n in (*range(3, 15), 3):
            result = runner.predict_batch([{"input_feature_dict": tiny_features(n, 101)}], [101])[0]
            if n == 3:
                if reference is None:
                    reference = result["coordinate"].clone()
                else:
                    assert torch.equal(reference, result["coordinate"])
        session = runner.diffusion_compilation_session
        assert session.graph_captures == 2 and session.graph_cache_entries == 2
        assert session.graph_replays > 0 and session.graph_fallbacks > 0
        runner.load_checkpoint()
        assert session.graph_cache_entries == 0
        assert runner.diffusion_compilation_session is None
        replayed = runner.predict_batch([{"input_feature_dict": tiny_features(3, 101)}], [101])[0]
        assert torch.equal(reference, replayed["coordinate"])
        assert runner.diffusion_compilation_session.graph_captures == 1
    finally:
        runner.close()
    assert all("forward" not in module.__dict__ for module in runner.model.modules())


@pytest.mark.skipif(os.environ.get("MOL_RUN_CUDA_GRAPH_TESTS") != "1", reason="Explicit CUDA runner integration test")
def test_cuda_default_eager_replay_matches_control_and_releases_graphs(checkpoint_config):
    config = checkpoint_config
    config.device = "cuda"
    config.model.N_cycle = 1
    config.sample_diffusion.N_step = 2
    config.sample_diffusion.N_sample = 2
    runner = InferenceRunner(config)
    keys = ("coordinate", "pae", "pde", "plddt", "resolved")
    references = {}
    try:
        with patch("torch.compile", side_effect=AssertionError("Unexpected compilation")):
            config.infer_setting.cuda_graphs = False
            for n in (3, 4, 5):
                inputs = [{"input_feature_dict": tiny_features(n, seed)} for seed in (101, 103)]
                references[n] = runner.predict_batch(inputs, [101, 103])
            config.infer_setting.cuda_graphs = True
            for n in (3, 4, 5, 3):
                inputs = [{"input_feature_dict": tiny_features(n, seed)} for seed in (101, 103)]
                outputs = runner.predict_batch(inputs, [101, 103])
                for got, want in zip(outputs, references[n], strict=True):
                    for key in keys:
                        assert torch.equal(got[key], want[key])
            session = runner.diffusion_compilation_session
            assert session.compiled_calls == 0
            assert session.graph_captures == 2 and session.graph_cache_entries == 2
            assert session.graph_replays > 0 and session.graph_fallbacks > 0
            runner.load_checkpoint()
            assert session.graph_cache_entries == 0 and runner.diffusion_compilation_session is None
            inputs = [{"input_feature_dict": tiny_features(3, seed)} for seed in (101, 103)]
            outputs = runner.predict_batch(inputs, [101, 103])
            for got, want in zip(outputs, references[3], strict=True):
                for key in keys:
                    assert torch.equal(got[key], want[key])
            with patch("mol.runner.inference.to_device", side_effect=RuntimeError("test transfer failure")):
                with pytest.raises(RuntimeError, match="test transfer failure"):
                    runner.predict({})
            assert runner.diffusion_compilation_session is None
    finally:
        runner.close()
    assert all("forward" not in module.__dict__ for module in runner.model.modules())
