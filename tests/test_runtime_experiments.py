"""Optional experiments must retain exact arithmetic, inputs and ownership."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mol.model import utils
from mol.model.generator import sample_diffusion
from mol.model.modules.transformer import DiffusionTransformer
from mol.model.runtime_experiments import (
    AggregationMetadata, cache_aggregation_metadata, replay_eager_token_stack,
    sources_snapshot, unchanged,
)


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_aggregation_reuse_and_in_place_map_edits(reduce):
    index = torch.tensor([2, 0, 2, 0, 3])
    x = torch.randn(2, 5, 8)
    cache = AggregationMetadata()
    for step in range(4):
        if step == 2:
            index[0] = 1
        actual = cache(x, index, 5, reduce)
        expected = utils._aggregate_atom_to_token_grouped(x, index, 5, reduce)
        assert torch.equal(actual, expected)
    assert cache.builds == 2 and cache.hits == 2
    with pytest.raises(RuntimeError, match='exceeds n_token'):
        cache(x, index, 2, reduce)


def test_aggregation_context_restores_and_discards_entries_on_failure():
    original = utils._aggregate_atom_to_token_grouped
    with pytest.raises(RuntimeError, match='test failure'):
        with cache_aggregation_metadata() as cache:
            utils._aggregate_atom_to_token_grouped(torch.randn(3, 2), torch.tensor([0, 1, 1]), 2, 'mean')
            assert len(cache.entries) == 1
            raise RuntimeError('test failure')
    assert utils._aggregate_atom_to_token_grouped is original
    assert not cache.entries


def test_versions_and_replacements_are_detected_and_inference_tensors_refresh():
    tensor = torch.randn(3, 4)
    snapshot = sources_snapshot({'value': tensor})
    assert unchanged(snapshot, {'value': tensor})
    assert not unchanged(snapshot, {'value': tensor.clone()})
    tensor[:, 0].add_(1)
    assert not unchanged(snapshot, {'value': tensor})
    with torch.inference_mode():
        tensor = torch.ones(3)
        assert not unchanged(sources_snapshot(tensor), tensor)
        index = torch.tensor([0, 1, 1])
        cache = AggregationMetadata()
        cache(torch.randn(3, 2), index, 2, 'mean')
        cache(torch.randn(3, 2), index, 2, 'mean')
        assert cache.builds == 2 and cache.hits == 0


@pytest.mark.parametrize('chunk', [1, 2, None])
@pytest.mark.parametrize('gamma_min', [1.0, 0.99999999])
def test_sampler_branches_preserve_outputs_and_global_rng(chunk, gamma_min):
    kwargs = dict(
        denoise_net=lambda **kw: kw['x_noisy'] * 0.25,
        input_feature_dict={'atom_to_token_idx': torch.zeros(8, dtype=torch.long)},
        s_inputs=torch.zeros(2, 1), s_trunk=None, z_trunk=None, pair_z=None, p_lm=None, c_l=None,
        noise_schedule=torch.tensor([4., 2., 1., 0.]), N_sample=3,
        diffusion_chunk_size=chunk, gamma_min=gamma_min,
    )
    outputs, rngs = [], []
    for enabled in (False, True):
        torch.manual_seed(101); np.random.seed(101)
        outputs.append(sample_diffusion(**kwargs, precompute_schedule_branches=enabled))
        rngs.append((torch.get_rng_state(), np.random.get_state()))
    assert torch.equal(*outputs)
    assert torch.equal(rngs[0][0], rngs[1][0])
    assert np.array_equal(rngs[0][1][1], rngs[1][1][1])
    assert rngs[0][1][2:] == rngs[1][1][2:]


cuda = pytest.mark.skipif(os.environ.get('MOL_RUN_CUDA_GRAPH_TESTS') != '1', reason='Explicit CUDA experiment tests')


def token_model():
    torch.manual_seed(17)
    model = DiffusionTransformer(c_a=32, c_s=16, c_z=8, n_blocks=2, n_heads=4).eval().cuda()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .05)
    return model


def tensors(n):
    return (torch.randn(2, n, 32, device='cuda'), torch.randn(2, n, 16, device='cuda'),
            torch.randn(1, n, n, 8, device='cuda'))


@cuda
@pytest.mark.parametrize('stable', [False, True])
def test_cuda_eager_replay_exact_with_mutations_and_retained_outputs(stable):
    model = token_model()
    retained = []
    names = ('z', 'projected_pair_biases', 'extra_attn_bias') if stable else ()
    with torch.no_grad(), patch('torch.compile', side_effect=AssertionError('Unexpected compilation')):
        with replay_eager_token_stack(model, stable_inputs=names) as replay:
            for n in (17, 23):
                a, s, z = tensors(n)
                proj = model.prepare_pair_biases(z)
                mask = torch.zeros(n, n, device='cuda')
                for step in range(4):
                    if step == 2:
                        proj.blocks[0][..., 0].add_(0.5)
                        mask[:, -1] = -1e4
                    kwargs = dict(projected_pair_biases=proj, extra_attn_bias=mask)
                    expected = replay.original(a, s, z, **kwargs)
                    rng = torch.cuda.get_rng_state().clone()
                    actual = model(a, s, z, **kwargs)
                    assert torch.equal(expected, actual)
                    assert torch.equal(rng, torch.cuda.get_rng_state())
                    retained.append((actual, actual.clone()))
                if stable:
                    assert replay.session.skipped_bytes > 0
            assert replay.session.graph_captures == 2
            assert replay.session.graph_replays == 8
        assert replay.session.graph_cache_entries == 0
    assert 'forward' not in model.__dict__ and not hasattr(model, '_compilation_session')
    assert all(torch.equal(a, b) for a, b in retained)


@cuda
@pytest.mark.parametrize('evict', [False, True])
def test_cuda_shape_churn_and_eviction_preserve_outputs(evict):
    model = token_model()
    with torch.no_grad(), replay_eager_token_stack(model, max_graphs=2, evict=evict) as replay:
        for n in (17, 17, 23, 23, 31, 31, 17, 17):
            a, s, z = tensors(n)
            expected = replay.original(a, s, z)
            assert torch.equal(model(a, s, z), expected)
            assert replay.session.graph_cache_entries <= 2
        assert replay.session.graph_evictions == (2 if evict else 0)
        assert replay.session.graph_replays == (8 if evict else 6)
        assert replay.session.graph_fallbacks == (0 if evict else 2)


@cuda
def test_cuda_memory_budget_rejects_once_and_keeps_eager_results():
    model = token_model()
    with torch.no_grad(), replay_eager_token_stack(model, max_cache_bytes=1) as replay:
        a, s, z = tensors(17)
        expected = replay.original(a, s, z)
        assert torch.equal(model(a, s, z), expected)
        assert torch.equal(model(a, s, z), expected)
        assert replay.session.graph_captures == 1
        assert replay.session.graph_fallbacks == 2
        assert replay.session.cache_bytes == 0


@cuda
def test_cuda_inference_tensors_are_recopied_after_mutation():
    model = token_model()
    with torch.inference_mode(), replay_eager_token_stack(model, stable_inputs=('z',)) as replay:
        a, s, z = tensors(17)
        for _ in range(2):
            z.add_(.1)
            assert torch.equal(model(a, s, z), replay.original(a, s, z))
        assert replay.session.skipped_tensors == 0
