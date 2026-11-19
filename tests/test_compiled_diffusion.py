"""Compilation must retain sampling, cached conditioning and module ownership."""

import copy
import os
from unittest.mock import patch

import pytest
import torch

from mol.model.compiled_diffusion import compiled_diffusion_blocks
from mol.model.modules.transformer import DiffusionTransformer
from mol.model.modules.transformer import ProjectedPairBiases
from mol.model.mol import Mol
from tests.test_input_batching import activate_residuals, tiny_config, tiny_features


def token_transformer():
    torch.manual_seed(17)
    model = DiffusionTransformer(c_a=16, c_s=12, c_z=8, n_blocks=2, n_heads=4).eval()
    activate_residuals(model)
    return model


def inputs(tokens=5):
    return (torch.randn(2, tokens, 16), torch.randn(2, tokens, 12), torch.randn(1, tokens, tokens, 8))


@pytest.mark.parametrize("cache", ["full", "shared", "off"])
def test_complete_seeded_batch_matches_eager_and_restores_forwards(cache):
    torch.manual_seed(71)
    cfg = tiny_config()
    cfg.diffusion_cache_mode = cache
    cfg.infer_setting.atom_block_cache = cache != "off"
    model = Mol(cfg).eval()
    activate_residuals(model)
    transformer = model.diffusion_module.diffusion_transformer
    state = {k: v.clone() for k, v in model.state_dict().items()}
    parameter_ids = [id(p) for p in model.parameters()]
    features = [tiny_features(3, 101), tiny_features(5, 103)]
    atom_forward = model.diffusion_module.atom_attention_encoder.atom_transformer.forward
    with torch.no_grad():
        expected = model.forward_batch(copy.deepcopy(features), [101, 103])
        with compiled_diffusion_blocks(transformer, backend="eager") as session:
            actual = model.forward_batch(copy.deepcopy(features), [101, 103])
            assert session.compiled_calls > 0
            assert session.eager_calls == 0
            assert [id(p) for p in model.parameters()] == parameter_ids
            assert model.state_dict().keys() == state.keys()
            assert model.diffusion_module.atom_attention_encoder.atom_transformer.forward == atom_forward
            session.enabled = False
            reverted = model.forward_batch(copy.deepcopy(features), [101, 103])
            assert session.eager_calls > 0
        for wanted, got, restored in zip(expected, actual, reverted, strict=True):
            for key in ("coordinate", "pae", "pde", "plddt", "resolved"):
                assert torch.equal(got[key], wanted[key]), key
                assert torch.equal(restored[key], wanted[key]), key
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[key]), key
    assert all("forward" not in b.__dict__ for b in transformer.blocks)


def test_training_and_gradient_enabled_calls_bypass_compilation():
    model = token_transformer()
    a, s, z = inputs()
    with compiled_diffusion_blocks(model, backend="eager") as session:
        model(a.requires_grad_(), s, z).sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        model.train()
        with torch.no_grad():
            model(a, s, z)
        assert session.compiled_calls == 0
        assert session.eager_calls == 4


def test_compiled_graph_uses_fresh_inputs_and_fixed_shape_reuses_graph():
    model = token_transformer()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    with torch.no_grad(), compiled_diffusion_blocks(model, backend=backend) as session:
        for n in (5, 5, 7, 5):
            data = inputs(n)
            session.enabled = False
            expected = model(*data)
            session.enabled = True
            actual = model(*data)
            assert actual.shape == data[0].shape
            assert torch.equal(actual, expected)
            if n == 7:
                graphs_after_new_shape = len(graphs)
        assert len(graphs) == graphs_after_new_shape
        assert graphs


def test_setup_failure_and_prediction_failure_restore_instances():
    model = token_transformer()
    original = model.blocks[0].forward
    with patch.object(torch, "compile", side_effect=[lambda *a, **k: a, RuntimeError("setup failed")]):
        with pytest.raises(RuntimeError, match="setup failed"):
            with compiled_diffusion_blocks(model):
                pass
    assert model.blocks[0].forward == original
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert not hasattr(model, "_compilation_session")
    with pytest.raises(RuntimeError, match="prediction failed"):
        with compiled_diffusion_blocks(model, backend="eager"):
            raise RuntimeError("prediction failed")
    assert model.blocks[0].forward == original


def test_fused_norm_forward_is_scoped_and_restored_on_setup_failure():
    from mol.model.layer_norm.layer_norm import FusedLayerNorm

    model = token_transformer()
    # Exercise installation independently of the process-wide LayerNorm choice.
    with patch("torch.cuda.is_available", return_value=False):
        norm = FusedLayerNorm(16, create_scale=False, create_offset=False).eval()
    model.blocks[0].attention_pair_bias.layernorm_a.layernorm_a = norm
    original = norm.forward
    with patch.object(torch, "compile", side_effect=RuntimeError("setup failed")):
        with pytest.raises(RuntimeError, match="setup failed"):
            with compiled_diffusion_blocks(model):
                pass
    assert norm.forward == original
    assert "forward" not in norm.__dict__
    assert not hasattr(model, "_compilation_session")
    with pytest.raises(RuntimeError, match="prediction failed"):
        with compiled_diffusion_blocks(model, backend="eager"):
            raise RuntimeError("prediction failed")
    assert norm.forward == original
    assert all("forward" not in module.__dict__ for module in model.modules())


def test_nested_sessions_and_training_entry_rejected():
    model = token_transformer()
    with compiled_diffusion_blocks(model, backend="eager"):
        with pytest.raises(RuntimeError, match="already"):
            with compiled_diffusion_blocks(model, backend="eager"):
                pass
    model.train()
    with pytest.raises(ValueError, match="eval"):
        with compiled_diffusion_blocks(model):
            pass


def test_twenty_four_blocks_reuse_graphs_without_cache_thrashing():
    torch.manual_seed(19)
    model = DiffusionTransformer(c_a=16, c_s=12, c_z=8, n_blocks=24, n_heads=4).eval()
    activate_residuals(model)
    data = inputs()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    with torch.no_grad():
        expected = model(*data)
        with compiled_diffusion_blocks(model, backend=backend) as session:
            first = model(*data)
            count = len(graphs)
            second = model(*data)
            assert len(graphs) == count > 0
            assert session.compiled_calls == 48
        assert torch.equal(first, expected) and torch.equal(second, expected)


def test_benchmark_refuses_an_occupied_gpu_before_allocating(monkeypatch):
    import os
    import subprocess
    from scripts.benchmark_diffusion_compile import require_unoccupied_gpu

    gpu = "GPU-cce25f68-61f4-0137-5b73-95af751fbf8d"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    results = [
        subprocess.CompletedProcess([], 0, stdout=gpu + "\n"),
        subprocess.CompletedProcess([], 0, stdout=f"{gpu}, {os.getpid() + 100}\n"),
    ]
    with patch.object(subprocess, "run", side_effect=results):
        with pytest.raises(RuntimeError, match="other compute processes"):
            require_unoccupied_gpu()
    with patch.object(subprocess, "run", side_effect=results):
        assert require_unoccupied_gpu(allow_busy=True) == [f"{gpu}, {os.getpid() + 100}"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="UUID"):
        require_unoccupied_gpu()


def test_graph_configuration_is_explicit_and_requires_cuda():
    model = token_transformer()
    with pytest.raises(ValueError, match="CUDA model"):
        with compiled_diffusion_blocks(model, cuda_graphs=True):
            pass
    with pytest.raises(ValueError, match="positive integer"):
        with compiled_diffusion_blocks(model, max_graphs=0):
            pass


def test_static_buffers_preserve_strides_and_refresh_dataclass_tensors():
    from mol.model.cuda_graph_replay import tree_key, tree_map, tensor_leaves, owned_buffer, copy_inputs

    original = {
        "transposed": torch.randn(3, 5).T,
        "expanded": torch.randn(1, 3).expand(4, 3),
        "projections": ProjectedPairBiases((torch.randn(2, 3),)),
        "enabled": True,
    }
    static = tree_map(owned_buffer, original)
    assert tree_key(static) == tree_key(original)
    destinations = tensor_leaves(static)
    copy_inputs(destinations, tensor_leaves(original))
    for got, want in zip(destinations, tensor_leaves(original)):
        assert torch.equal(got, want)
        assert got.stride() == want.stride()
    original["projections"].blocks[0].add_(7)
    original["expanded"] = torch.randn(1, 3).expand(4, 3)
    copy_inputs(destinations, tensor_leaves(original))
    assert torch.equal(static["projections"].blocks[0], original["projections"].blocks[0])
    assert torch.equal(static["expanded"], original["expanded"])


@pytest.mark.skipif(os.environ.get("MOL_RUN_CUDA_GRAPH_TESTS") != "1", reason="Explicitly enabled CUDA graph test")
def test_cuda_replay_refreshes_conditioning_masks_and_preserves_retained_outputs():
    model = token_transformer().cuda()
    rng = torch.cuda.get_rng_state().clone()
    saved = []
    with torch.no_grad(), compiled_diffusion_blocks(model, cuda_graphs=True, max_graphs=1) as session:
        for index, n in enumerate((5, 5, 7, 5)):
            a, s, z = (t.cuda() for t in inputs(n))
            bias = torch.zeros(n, n, device="cuda")
            bias[:, -1] = -1e8 if index % 2 else 0
            projections = model.prepare_pair_biases(z)
            kwargs = {"projected_pair_biases": projections, "extra_attn_bias": bias}
            session.replay_enabled = False
            expected = model(a, s, z, **kwargs)
            session.replay_enabled = True
            actual = model(a, s, z, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"request {index}, tokens {n}")
            saved.append((actual, actual.clone()))
            # In-place edits to the same cache tensors must also reach replay.
            projections.blocks[0][..., 0].add_(0.75)
            session.replay_enabled = False
            expected2 = model(a + 0.3, s - 0.2, z, **kwargs)
            session.replay_enabled = True
            actual2 = model(a + 0.3, s - 0.2, z, **kwargs)
            torch.testing.assert_close(actual2, expected2, rtol=0, atol=0)
        assert session.graph_captures == 1
        assert session.graph_replays == 6
        assert session.graph_fallbacks == 2
        for actual, snapshot in saved:
            assert torch.equal(actual, snapshot)
        assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert "forward" not in model.__dict__
    assert all("forward" not in b.__dict__ for b in model.blocks)
    assert all("forward" not in module.__dict__ for module in model.modules())
    for actual, snapshot in saved:
        assert torch.equal(actual, snapshot)


@pytest.mark.skipif(os.environ.get("MOL_RUN_CUDA_GRAPH_TESTS") != "1", reason="Explicitly enabled CUDA fused LayerNorm test")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [(False, False), (False, True), (True, False), (True, True)])
def test_fused_layer_norm_custom_op_keeps_original_kernel(dtype, affine):
    from mol.model.compiled_layer_norm import compile_compatible_forward, fused_layer_norm
    from mol.model.layer_norm.layer_norm import FusedLayerNorm

    # Each parameterized case deliberately compiles several distinct layouts;
    # do not accumulate these standalone probe graphs across test cases.
    torch.compiler.reset()
    model = FusedLayerNorm(16, create_scale=affine[0], create_offset=affine[1]).cuda().eval()
    original = model.forward
    forward = compile_compatible_forward(model, original)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        # Include transposed storage: output metadata must describe the kernel's
        # contiguous output, not the input strides.
        for x in (torch.randn(2, 3, 16, device="cuda", dtype=dtype),
                  torch.randn(3, 2, 16, device="cuda", dtype=dtype).transpose(0, 1)):
            expected = original(x)
            opaque = fused_layer_norm(x, model.weight, model.bias, [16], model.eps)
            actual = compiled(x)
            assert torch.equal(opaque, expected)
            assert torch.equal(actual, expected)
            assert actual.stride() == expected.stride()
        assert graphs
        assert all(any("mol_inference.fused_layer_norm" in str(node.target)
                       for node in graph.graph.nodes) for graph in graphs)
        torch.library.opcheck(
            fused_layer_norm,
            (x, model.weight, model.bias, [16], model.eps),
            test_utils=("test_schema", "test_faketensor"),
        )
        static = x.contiguous()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            original(static)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = original(static)
        for _ in range(2):
            static.copy_(torch.randn_like(static))
            graph.replay()
            assert torch.equal(captured, original(static))
        torch.cuda.synchronize()
        graph.reset()
