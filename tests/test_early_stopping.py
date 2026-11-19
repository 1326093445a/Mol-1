"""First-sample screening preserves surviving samples and fixed batch shapes."""
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mol.config.inference import validate_early_stopping
from mol.model import batching, early_stopping
from mol.model.generator import sample_diffusion
from mol.model.mol import Mol
from mol.runner.batched_prediction import validate_prediction
from mol.runner.dumper import DataDumper
from mol.utils.seed import seed_everything
from tests.test_input_batching import activate_residuals, tiny_config, tiny_features


def config():
    cfg = tiny_config()
    cfg.model.N_cycle = 1
    cfg.sample_diffusion.N_step = 3
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    cfg.infer_setting.confidence_sample_chunk_size = 1
    return cfg


def features(seed, *, global_rng=False, three_chains=False):
    feat = tiny_features(6, seed)
    if global_rng:
        del feat['inference_seed']
    groups = {'A': np.arange(3), 'B': np.arange(3, 6)}
    if three_chains:
        groups = {'A': np.arange(2), 'B': np.arange(2, 4), 'C': np.arange(4, 6)}
    feat['ipsae_early_stop'] = {'groups': groups, 'interface': None}
    return feat


def assert_outputs(got, want, count):
    for key in ('coordinate', 'pae', 'pde', 'plddt', 'resolved'):
        assert torch.equal(got[key], want[key][:count]), key
    assert torch.equal(got['contact_probs'], want['contact_probs'])
    for a, b in zip(got['full_data'], want['full_data'][:count], strict=True):
        assert torch.equal(a['token_pair_pae'], b['token_pair_pae'])


@pytest.mark.parametrize('score,stopped', [(0., True), (.049, True), (.05, False), (.2, False)])
@pytest.mark.parametrize('global_rng', [False, True])
def test_single_first_sample_reuses_heads_and_preserves_rng(score, stopped, global_rng):
    cfg = config()
    seed_everything(71, deterministic=False)
    model = Mol(cfg).eval()
    activate_residuals(model)
    feat = features(101, global_rng=global_rng)
    with torch.no_grad():
        seed_everything(101, deterministic=False)
        control = model(input_feature_dict=copy.deepcopy(feat), label_full_dict=None, label_dict=None, mode='inference')[0]
        rng = torch.get_rng_state(), np.random.get_state()
        cfg.infer_setting.early_stop_ipsae_threshold = .05
        seed_everything(101, deterministic=False)
        with patch.object(early_stopping, 'ipsae_summary', return_value={'ipsae_min_10': score}), \
             patch.object(model, 'run_inference_heads', wraps=model.run_inference_heads) as heads:
            actual = model(input_feature_dict=copy.deepcopy(feat), label_full_dict=None, label_dict=None, mode='inference')[0]
        assert_outputs(actual, control, 1 if stopped else 3)
        assert torch.equal(torch.get_rng_state(), rng[0])
        assert np.array_equal(np.random.get_state()[1], rng[1][1])
        assert np.random.get_state()[2:] == rng[1][2:]
        assert [call.args[4].shape[0] for call in heads.call_args_list] == ([1] if stopped else [1, 2])
        assert actual['sampling_early_stop']['stopped'] == stopped
        assert actual['summary_confidence'][0]['sampling_early_stop']['skipped_samples'] == (2 if stopped else 0)
        validate_prediction(actual, n_atom=24, n_token=6, n_sample=3, early_stop_threshold=.05)
        if stopped:
            with pytest.raises(ValueError, match='shortfall'):
                validate_prediction(actual, n_atom=24, n_token=6, n_sample=3)


@pytest.mark.parametrize('scores', [(0., 0.), (0., .2), (.05, .049)])
def test_batch_stops_only_if_all_first_scores_fail_and_never_shrinks(scores):
    cfg = config()
    seed_everything(71, deterministic=False)
    model = Mol(cfg).eval()
    activate_residuals(model)
    feats = [features(101), features(103)]
    with torch.no_grad():
        control = model.forward_batch(copy.deepcopy(feats), [101, 103])
        cfg.infer_setting.early_stop_ipsae_threshold = .05
        sizes = []
        original = batching._denoise_batch

        def denoise(module, states, *args, **kwargs):
            sizes.append(len(states))
            return original(module, states, *args, **kwargs)

        with patch.object(early_stopping, 'ipsae_summary', side_effect=[{'ipsae_min_10': s} for s in scores]), \
             patch.object(batching, '_denoise_batch', side_effect=denoise):
            actual = model.forward_batch(copy.deepcopy(feats), [101, 103])
    stopped = all(s < .05 for s in scores)
    assert sizes == [2] * (3 if stopped else 9)
    for got, want in zip(actual, control, strict=True):
        assert_outputs(got, want, 1 if stopped else 3)
        assert got['sampling_early_stop']['stopped'] == stopped


def test_undefined_score_keeps_entire_batch_and_explicit_interface_can_stop():
    cfg = config()
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    model = Mol(cfg).eval()
    feats = [features(101, three_chains=True), features(103)]
    with torch.no_grad():
        with patch.object(early_stopping, 'ipsae_summary', return_value={'ipsae_min_10': 0}) as score:
            result = model.forward_batch(copy.deepcopy(feats), [101, 103])
        assert score.call_count == 1
        assert all(p['coordinate'].shape[0] == 3 for p in result)
        assert result[0]['sampling_early_stop']['reason'] == 'ipsae_unavailable'
        feats[0]['ipsae_early_stop']['interface'] = [['A'], ['B', 'C']]
        with patch.object(early_stopping, 'ipsae_summary', return_value={'ipsae_min_10': 0}) as score:
            result = model.forward_batch(copy.deepcopy(feats), [101, 103])
        assert score.call_count == 2
        assert score.call_args_list[0].kwargs['interface'] == [['A'], ['B', 'C']]
        assert all(p['coordinate'].shape[0] == 1 for p in result)


def test_real_ipsae_score_matches_emitted_pae():
    from mol.metrics.ipsae import ipsae_summary

    cfg = config()
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    feat = features(101)
    model = Mol(cfg).eval()
    with torch.no_grad():
        result = model(input_feature_dict=feat, label_full_dict=None, label_dict=None, mode='inference')[0]
    score = ipsae_summary(result['full_data'][0]['token_pair_pae'], feat['ipsae_early_stop']['groups'])['ipsae_min_10']
    assert result['sampling_early_stop']['first_sample_score'] == score
    assert result['sampling_early_stop']['stopped'] == (score < .05)


@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan'), float('inf'), True])
def test_bad_thresholds_fail_before_prediction(threshold):
    cfg = config()
    cfg.infer_setting.early_stop_ipsae_threshold = threshold
    with pytest.raises(ValueError, match='threshold'):
        validate_early_stopping(cfg)


@pytest.mark.parametrize('key', ['sample_diffusion_chunk_size', 'confidence_sample_chunk_size'])
def test_sample_chunk_changes_are_never_silent(key):
    cfg = config()
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    cfg.infer_setting[key] = 2
    with pytest.raises(ValueError, match=key):
        validate_early_stopping(cfg)


def test_single_requested_sample_does_not_invoke_gate():
    cfg = config()
    cfg.sample_diffusion.N_sample = 1
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    model = Mol(cfg).eval()
    with torch.no_grad(), patch.object(early_stopping, 'ipsae_summary', side_effect=AssertionError('Unexpected scoring')):
        result = model(input_feature_dict=tiny_features(6, 101), label_full_dict=None, label_dict=None, mode='inference')[0]
    assert result['coordinate'].shape[0] == 1 and 'sampling_early_stop' not in result


def test_stopping_reduces_denoiser_calls_with_global_rng_preserved():
    kwargs = dict(input_feature_dict={'atom_to_token_idx': torch.zeros(4, dtype=torch.long)},
                  s_inputs=torch.zeros(2, 1), s_trunk=None, z_trunk=None, pair_z=None,
                  p_lm=None, c_l=None, noise_schedule=torch.tensor([4., 2., 1., 0.]),
                  N_sample=5, diffusion_chunk_size=1)
    calls, results, rngs = [], [], []
    for stop in (False, True):
        seed_everything(101, deterministic=False)
        count = []

        def denoise(**kw):
            count.append(1)
            return kw['x_noisy'] * .5

        results.append(sample_diffusion(denoise_net=denoise, **kwargs,
                                       stop_after_first_sample=(lambda x: True) if stop else None))
        calls.append(len(count)); rngs.append((torch.get_rng_state(), np.random.get_state()))
    assert calls == [15, 3]
    assert torch.equal(results[0][:1], results[1])
    assert torch.equal(rngs[0][0], rngs[1][0])
    assert np.array_equal(rngs[0][1][1], rngs[1][1][1])
    assert rngs[0][1][2:] == rngs[1][1][2:]


def status_prediction():
    return {'coordinate':torch.zeros(1,4,3), 'sampling_early_stop':{
        'threshold':.05, 'stopped':True, 'requested_samples':5,
        'completed_samples':1, 'skipped_samples':4, 'first_sample_score':0.,
    }}


def test_sampling_status_is_complete_only_after_successful_output(tmp_path):
    dumper = DataDumper(str(tmp_path))
    kwargs = dict(dataset_name='',pdb_id='job',seed=101,pred_dict=status_prediction(),
                  atom_array=None,entity_poly_type={})
    path = tmp_path/'job/seed_101/sampling_status.json'
    with patch.object(dumper,'dump_predictions',side_effect=OSError('writer failed')):
        with pytest.raises(OSError,match='writer failed'):
            dumper.dump(**kwargs)
    assert json.loads(path.read_text())['status'] == 'writing'
    with patch.object(dumper,'dump_predictions'):
        dumper.dump(**kwargs)
    report = json.loads(path.read_text())
    assert report['status'] == 'complete' and report['skipped_samples'] == 4
    assert not path.with_suffix('.json.tmp').exists()


def test_stale_samples_are_rejected_without_deleting_them(tmp_path):
    directory = tmp_path/'job/seed_101/predictions'
    directory.mkdir(parents=True)
    old = directory/'job_summary_confidence_sample_4.json'
    old.write_text('old output')
    with pytest.raises(ValueError,match='stale samples'):
        DataDumper(str(tmp_path)).dump(dataset_name='',pdb_id='job',seed=101,
                                     pred_dict=status_prediction(),atom_array=None,entity_poly_type={})
    assert old.read_text() == 'old output'


def test_multi_rank_early_stop_is_rejected_before_collectives():
    cfg = config()
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    cfg.context_parallel_size = 2
    with pytest.raises(ValueError,match='context parallelism'):
        validate_early_stopping(cfg)
