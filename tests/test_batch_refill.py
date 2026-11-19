"""Refill keeps shape, caches, sample streams and output ownership independent."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mol.config.inference import validate_input_batching
from mol.model import batching, early_stopping
from mol.model.mol import Mol
from mol.model.refill import forward_refill
from mol.runner.batched_prediction import validate_prediction
from mol.runner.inference_pipeline import PredictionWriter
from mol.runner.refill_prediction import run_refill_sessions
from tests.test_early_stopping import assert_outputs, config, features
from tests.test_input_batching import activate_residuals


def refill_config():
    cfg = config()
    cfg.batch_size = 2
    cfg.infer_setting.batch_refill = True
    cfg.infer_setting.early_stop_ipsae_threshold = .05
    return cfg


@pytest.mark.parametrize("cache", ["full", "shared", "off"])
def test_mixed_refill_is_exact_with_survivors_new_contents_and_partial_tail(cache):
    torch.manual_seed(71)
    cfg = refill_config()
    cfg.diffusion_cache_mode = cache
    model = Mol(cfg).eval()
    activate_residuals(model)
    seeds = [101, 103, 107, 109, 113]
    feats = [features(seed) for seed in seeds]
    # Equal token count does not imply equal atom topology. Refill this slot
    # with three rather than four atoms per token, retaining valid frames.
    keep = feats[2]['atom_to_tokatom_idx'] < 3
    for name, value in list(feats[2].items()):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 24:
            feats[2][name] = value[keep]
    feats[2]['frame_atom_index'] = torch.arange(6)[:, None] * 3 + torch.arange(3)[None, :]
    expected, actual, shapes, trunks = {}, {}, [], []
    with torch.no_grad():
        cfg.infer_setting.batch_refill = False
        cfg.infer_setting.early_stop_ipsae_threshold = None
        for start in range(0, 5, 2):
            group = copy.deepcopy(feats[start:start+2])
            group_seeds = seeds[start:start+2]
            if len(group) == 1:
                group.append(copy.deepcopy(group[-1])); group_seeds.append(0)
            outputs = model.forward_batch(group, group_seeds)
            expected.update({start+i: p for i, p in enumerate(outputs) if start+i < 5})
        cfg.infer_setting.batch_refill = True
        cfg.infer_setting.early_stop_ipsae_threshold = .05
        score_by_seed = {seed: (.0 if i % 2 == 0 else .2) for i, seed in enumerate(seeds)}
        native = early_stopping.FirstSampleGate.__call__

        def score(gate, coordinate):
            with patch.object(early_stopping, "ipsae_summary", return_value={
                "ipsae_min_10": score_by_seed[int(gate.features[0]["inference_seed"])]
            }):
                return native(gate, coordinate)

        denoiser = batching._denoise_batch

        def record(module, states, *args, **kwargs):
            shapes.append(tuple((state[1].shape[-2], x.shape[0]) for state, x in zip(states, args[0])))
            return denoiser(module, states, *args, **kwargs)

        hook = model.pairformer_stack.register_forward_pre_hook(
            lambda m, a, kw: trunks.append(tuple(kw["s"].shape[:2])), with_kwargs=True)
        batches = [[(i, copy.deepcopy(feats[i]), seeds[i]) for i in range(start, min(start+2, 5))]
                   for start in range(0, 5, 2)]
        try:
            with patch.object(early_stopping.FirstSampleGate, "__call__", score), \
                 patch.object(batching, "_denoise_batch", record):
                stats = forward_refill(model, batches, lambda key, pred: actual.__setitem__(key, pred))
        finally:
            hook.remove()
    assert set(actual) == set(expected)
    assert set(trunks) == {(2, 6)} and len(trunks) == 3
    assert set(shapes) == {((6, 1), (6, 1))}
    assert stats == dict(completed=5, stopped=3, trunk_batches=3, sample_rounds=5,
                         real_samples=9, placeholder_samples=1, refills=3, max_ready=1)
    for key, pred in actual.items():
        assert_outputs(pred, expected[key], 1 if key % 2 == 0 else 3)
        validate_prediction(pred, n_atom=18 if key == 2 else 24, n_token=6, n_sample=3, early_stop_threshold=.05)
        assert pred['sampling_early_stop']['batch_policy'] == 'preserve_shape_refill'
        assert pred['sampling_early_stop']['batch_size'] == 2


def test_undefined_score_continues_and_finished_slots_are_only_placeholders():
    cfg = refill_config()
    model = Mol(cfg).eval()
    feats = [features(101, three_chains=True), features(103)]
    actual = {}
    with torch.no_grad(), patch.object(early_stopping, "ipsae_summary", return_value={"ipsae_min_10": 0}):
        stats = forward_refill(model, [[(i, feat, 101+2*i) for i, feat in enumerate(feats)]],
                               lambda key, pred: actual.__setitem__(key, pred))
    assert actual[0]['coordinate'].shape[0] == 3
    assert actual[0]['sampling_early_stop']['reason'] == 'ipsae_unavailable'
    assert actual[1]['coordinate'].shape[0] == 1
    assert stats['sample_rounds'] == 3 and stats['placeholder_samples'] == 2


@pytest.mark.parametrize('setting,value,match', [
    ('batch_size', 1, 'batch_size'), ('sample_diffusion.N_sample', 1, 'N_sample'),
    ('infer_setting.early_stop_ipsae_threshold', None, 'threshold'),
    ('infer_setting.sample_diffusion_chunk_size', 2, 'sample_diffusion_chunk_size'),
    ('context_parallel_size', 2, 'context parallelism'),
])
def test_refill_rejects_incompatible_settings(setting, value, match):
    cfg = refill_config()
    node = cfg
    parts = setting.split('.')
    for part in parts[:-1]: node = node[part]
    node[parts[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_input_batching(cfg)


def test_refill_rejects_unequal_tokens_and_fixed_batch_budget():
    from tests.test_input_batching import tiny_features

    cfg = refill_config()
    model = Mol(cfg).eval()
    with pytest.raises(ValueError, match='exactly equal'):
        forward_refill(model, [[(0, features(101), 101), (1, tiny_features(7, 103), 103)]], lambda *a: None)
    cfg.infer_setting.batch_max_pair_tokens = 71
    with pytest.raises(ValueError, match='batch_max_pair_tokens'):
        forward_refill(model, [[(0, features(101), 101)]], lambda *a: None)


@pytest.mark.parametrize('fail', [None, 'model', 'writer'])
def test_session_scheduler_streams_equal_lengths_and_owns_failures(fail):
    cfg = refill_config()
    prepared = [(101+i, {'sample_name': str(i), 'N_token': torch.tensor(n)}, None, '')
                for i, n in enumerate([6, 6, 6, 7, 7])]
    sessions, written, errors = [], [], []

    def predict(batches, completed):
        seen = []
        sessions.append(seen)
        for batch in batches:
            seen.extend(int(data['N_token']) for _, data, _ in batch)
            for key, data, seed in batch:
                if fail == 'model' and data['sample_name'] == '1':
                    raise torch.cuda.OutOfMemoryError('test OOM')
                completed(key, {'seed': seed})
        return {}

    def submit(data, atoms, seed, pred):
        assert pred['seed'] == seed
        if fail == 'writer' and data['sample_name'] == '1':
            raise OSError('writer failed')
        written.append(data['sample_name'])

    runner = SimpleNamespace(device=torch.device('cpu'), update_model_configs=lambda c: None, predict_refill=predict)
    run_refill_sessions(runner, cfg, iter(prepared), submit,
                        lambda data, seed, error: errors.append(data['sample_name']))
    assert sessions[-1] == [7, 7]
    assert all(len(set(session)) == 1 for session in sessions)
    assert sorted(written + errors) == ['0', '1', '2', '3', '4']
    assert errors == ([] if fail is None else ['1', '2'] if fail == 'model' else ['1'])


def test_background_writer_retains_stop_completion_metadata():
    from tests.test_early_stopping import status_prediction

    writes = []
    dumper = SimpleNamespace(dump=lambda **kw: writes.append(kw))
    pred = status_prediction() | {'summary_confidence': [{}], 'full_data': [{}]}
    with PredictionWriter(dumper, lambda *a: pytest.fail(str(a)), background=True) as writer:
        writer.submit(pred_dict=pred, atom_array=torch.zeros(1).numpy(), pdb_id='job', seed=101)
    assert writes[0]['pred_dict']['sampling_early_stop'] == pred['sampling_early_stop']


def test_scheduler_detects_silent_unconsumed_jobs():
    cfg = refill_config()
    runner = SimpleNamespace(device=torch.device('cpu'), update_model_configs=lambda c: None,
                             predict_refill=lambda batches, completed: {})
    prepared = [(101+i, {'sample_name': str(i), 'N_token': torch.tensor(6)}, None, '') for i in range(3)]
    errors = []
    run_refill_sessions(runner, cfg, iter(prepared), lambda *a: pytest.fail('unexpected output'),
                        lambda data, seed, error: errors.append(data['sample_name']))
    assert errors == ['0', '1', '2']


@pytest.mark.parametrize('background', [False, True])
def test_inference_routes_refill_and_writes_each_job_seed_once(tmp_path, monkeypatch, background):
    import numpy as np
    from mol.runner import batched_prediction, inference

    cfg = refill_config()
    cfg.infer_setting.background_write = background
    cfg.use_seeds_in_json = True
    cfg.input_json_path = 'fixture.json'
    model = Mol(cfg).eval()
    writes = []

    class Dataset:
        def __init__(self, configs):
            self.inputs = [dict(name=f'job{i}', modelSeeds=[101, 103],
                                sequences=[{'proteinChain': {'sequence': 'AAAAAA'}}]) for i in range(3)]

        def __getitem__(self, i):
            seed = torch.initial_seed()
            return dict(sample_name=f'job{i}', N_token=torch.tensor(6),
                        input_feature_dict=features(seed), entity_poly_type={'1': 'polypeptide(L)'}), np.zeros(24), ''

    def predict(batches, completed):
        items = ([(key, data['input_feature_dict'], seed) for key, data, seed in batch] for batch in batches)
        with torch.no_grad():
            return forward_refill(model, items, completed)

    runner = SimpleNamespace(
        device=torch.device('cpu'), error_dir=str(tmp_path/'ERR'),
        update_model_configs=lambda c: setattr(model, 'configs', c), predict_refill=predict,
        dumper=SimpleNamespace(dump=lambda **kw: writes.append(kw)),
    )
    monkeypatch.setattr(batched_prediction, 'InferenceDataset', Dataset)
    monkeypatch.setattr(inference, 'fix_cterminal_carboxyl_oxygens', lambda x, atoms: x)
    with patch.object(early_stopping, 'ipsae_summary', return_value={'ipsae_min_10': 0}):
        inference.infer_predict(runner, cfg)
    assert {(w['pdb_id'], w['seed']) for w in writes} == {(f'job{i}', s) for i in range(3) for s in (101, 103)}
    assert len(writes) == 6
    assert all(w['pred_dict']['sampling_early_stop']['completed_samples'] == 1 for w in writes)
