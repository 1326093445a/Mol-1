"""Validate fixed-slot refill on the unchanged shipped 9fm7 example and real weights."""
import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark_diffusion_compile import differences, file_digest, require_unoccupied_gpu, summarize, tensor_digest
from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.model import batching
from mol.model.early_stopping import FirstSampleGate
from mol.model.layer_norm.layer_norm import FusedLayerNorm
from mol.model.triangular.layers import OpenFoldLayerNorm
from mol.runner.batch_inference import get_default_runner
from mol.runner.batched_prediction import validate_prediction
from mol.runner.dumper import DataDumper
from mol.runner.inference import fix_cterminal_carboxyl_oxygens, update_inference_configs
from mol.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--allow-busy-gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path('examples/examples_with_template/example_9fm7.json')
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in (root/'mol').rglob('*') if p.suffix in {'.py', '.cu', '.cpp', '.h'}) + [Path(__file__).resolve()]
    report = {'status': 'running', 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_hashes': {str(p.relative_to(root)): file_digest(p) for p in sources},
              'input': str(source), 'input_sha256': file_digest(source), 'checkpoint_sha256': file_digest(args.checkpoint),
              'passes': [], 'note': 'Shared GPU; timings are provisional and include trunks, confidence, comparison and writing. Mixed arms use a test-only continue override for seeds 103 and 105, with cutoff 1.0 for other jobs. This exercises refill and is not a natural pass rate.'}

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    with patch.dict(inference_configs, {'dump_dir': str(args.output/'runner')}):
        runner = get_default_runner(
            device='cuda', dtype='bf16', use_msa=False, use_template=False,
            seeds=[101], batch_size=2, n_cycle=args.cycles, n_step=args.steps,
            n_sample=args.samples, load_checkpoint_path=str(args.checkpoint),
            memory_settings={'infer_setting.early_stop_ipsae_threshold': .05,
                             'infer_setting.sample_diffusion_chunk_size': 1,
                             'infer_setting.confidence_sample_chunk_size': 1,
                             'infer_setting.batch_max_padding_ratio': 1.0},
        )
    runner.configs.input_json_path = str(source.resolve())
    dataset = InferenceDataset(runner.configs)
    seeds = [101+2*i for i in range(args.jobs)]
    inputs = {}
    for key, seed in enumerate(seeds):
        seed_everything(seed, deterministic=False)
        data, atoms, error = dataset[0]
        if error: raise RuntimeError(error)
        data['input_feature_dict']['inference_seed'] = torch.tensor(seed)
        inputs[key] = data, atoms
    data, atoms = inputs[0]
    assert len({int(item[0]['N_token']) for item in inputs.values()}) == 1
    report['reference_position_sha256'] = {key: tensor_digest(item[0]['input_feature_dict']['ref_pos']) for key, item in inputs.items()}
    assert len(set(report['reference_position_sha256'].values())) == args.jobs, 'Expected seed-specific reference augmentation'
    report['tokens'] = int(data['N_token'])
    report['seeds'] = seeds
    report['layernorm'] = {'fused': sum(isinstance(m, FusedLayerNorm) for m in runner.model.modules()),
                           'torch': sum(isinstance(m, OpenFoldLayerNorm) for m in runner.model.modules())}
    assert report['layernorm']['fused'] and not report['layernorm']['torch']
    original_gate, original_denoise = FirstSampleGate.__call__, batching._denoise_batch
    references, reference_cifs, reference_contacts = {}, {}, {}
    try:
        for mode in ('baseline', 'fixed_screen', 'refill_screen', 'fixed_mixed', 'refill_mixed'):
            mixed, refill = mode.endswith('mixed'), mode.startswith('refill')
            threshold = None if mode == 'baseline' else 1.0 if mixed else .05
            runner.configs.infer_setting.batch_refill = refill
            runner.configs.infer_setting.early_stop_ipsae_threshold = threshold
            runner.update_model_configs(update_inference_configs(copy.deepcopy(runner.configs), int(data['N_token'])))
            comparisons, counts, decisions, shapes, completion_order = {}, {}, {}, set(), []
            denoise_calls = 0
            output_seconds = 0.
            competitors = require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)

            def gate(self, coordinates):
                stopped = original_gate(self, coordinates)
                if mixed and any(int(f['inference_seed']) in (103, 105) for f in self.features):
                    self.stopped = False
                    return False
                return stopped

            def denoise(module, states, x, t, **kwargs):
                nonlocal denoise_calls
                denoise_calls += 1
                shapes.add((len(states), states[0][1].shape[-2], x[0].shape[0]))
                return original_denoise(module, states, x, t, **kwargs)

            def batches():
                for start in range(0, args.jobs, 2):
                    items = []
                    for key in range(start, min(start+2, args.jobs)):
                        job = copy.deepcopy(inputs[key][0])
                        items.append((key, job, seeds[key]))
                    yield items

            def completed(key, pred):
                nonlocal output_seconds
                torch.cuda.synchronize(); started = time.perf_counter()
                data, atoms = inputs[key]
                groups = data['input_feature_dict']['ipsae_early_stop']['groups']
                interface = data.get('ipsae_interface')
                count = pred['coordinate'].shape[0]
                validate_prediction(pred, n_atom=len(atoms), n_token=int(data['N_token']),
                                    n_sample=args.samples, early_stop_threshold=threshold)
                summary = summarize([pred], [groups], [interface])[0]
                if mode == 'baseline':
                    references[key] = summary
                    reference_contacts[key] = pred['contact_probs'].detach().cpu().clone()
                expected = {k: v[:count] for k, v in references[key].items()}
                comparisons[key] = differences([summary], [expected])[0]
                comparisons[key]['contact_probs_exact'] = torch.equal(pred['contact_probs'].detach().cpu(), reference_contacts[key])
                decision = pred.get('sampling_early_stop')
                if decision:
                    assert decision['first_sample_score'] == references[key]['ipsae'][0]['ipsae_min_10']
                counts[key], decisions[key] = count, decision
                completion_order.append(key)
                pred['coordinate'] = fix_cterminal_carboxyl_oxygens(pred['coordinate'], atoms)
                DataDumper(str(args.output/mode), sorted_by_ranking_score=False).dump(
                    dataset_name='', pdb_id=data['sample_name'], seed=seeds[key], pred_dict=pred,
                    atom_array=atoms, entity_poly_type={k: v for k, v in data['entity_poly_type'].items() if v != 'non-polymer'},
                    ipsae_interface=interface,
                )
                torch.cuda.synchronize(); output_seconds += time.perf_counter()-started
                print(json.dumps({'event': 'output_checked', 'mode': mode, 'key': key, 'samples': count}), flush=True)

            seed_everything(101, deterministic=False)
            with patch.object(FirstSampleGate, '__call__', gate), patch.object(batching, '_denoise_batch', denoise):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                stats = None
                if refill:
                    stats = runner.predict_refill(batches(), completed)
                else:
                    for items in batches():
                        padded = items + [(None, items[-1][1], 0)] * (2-len(items))
                        predictions = runner.predict_batch([item[1] for item in padded], [item[2] for item in padded])
                        for (key, _, _), pred in zip(padded, predictions, strict=True):
                            if key is not None: completed(key, pred)
                        del predictions, pred
                torch.cuda.synchronize(); elapsed = time.perf_counter()-started
                peak = torch.cuda.max_memory_allocated()
            assert set(counts) == set(range(args.jobs))
            cifs = {str(p.relative_to(args.output/mode)): file_digest(p) for p in (args.output/mode).rglob('*.cif')}
            if mode == 'baseline': reference_cifs = cifs
            cifs_exact = all(reference_cifs.get(k) == v for k, v in cifs.items())
            exact = all(v['exact'] if isinstance(v, dict) else v for row in comparisons.values() for v in row.values())
            session = runner.diffusion_compilation_session
            record = dict(mode=mode, seconds=elapsed, output_comparison_seconds=output_seconds,
                          inference_seconds=elapsed-output_seconds, peak_allocated_bytes=peak,
                          outputs_exact=exact, cifs_exact=cifs_exact, comparisons=comparisons,
                          samples=counts, decisions=decisions, completion_order=completion_order,
                          denoise_calls=denoise_calls, shapes=sorted(shapes), stats=stats,
                          competing_processes=competitors, cif_sha256=cifs,
                          graph_stats={k: getattr(session, k) for k in ('compiled_calls', 'graph_captures', 'graph_replays')})
            report['passes'].append(record); save()
            print(json.dumps({k: record[k] for k in ('mode', 'seconds', 'inference_seconds', 'outputs_exact', 'cifs_exact', 'samples', 'stats')}), flush=True)
            assert exact and cifs_exact, mode
            assert shapes == {(2, int(data['N_token']), 1)}
            if mode == 'refill_mixed':
                assert all(counts[i] == (args.samples if seeds[i] in (103, 105) else 1) for i in counts)
                assert stats['refills'] == args.jobs-2
        report['baseline_ipsae_min_10'] = {key: [s['ipsae_min_10'] for s in ref['ipsae']] for key, ref in references.items()}
        report['source_hashes_match'] = all(file_digest(root/p) == h for p, h in report['source_hashes'].items())
        report['status'] = 'completed'; save()
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = repr(exc); save(); raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
