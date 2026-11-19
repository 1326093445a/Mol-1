# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-example comparisons for the five opt-in runtime experiments."""
import argparse
import copy
import hashlib
import json
import os
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark_diffusion_compile import differences, file_digest, require_unoccupied_gpu, summarize, tensor_digest
from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
from mol.metrics.ipsae import protein_token_groups
from mol.model.cuda_graph_replay import EagerReplaySession
from mol.model.layer_norm.layer_norm import FusedLayerNorm
from mol.model.runtime_experiments import (
    ExperimentalReplayCache, ReplaySession, cache_aggregation_metadata, cache_sampler_branches,
)
from mol.model.triangular.layers import OpenFoldLayerNorm
from mol.runner.batch_inference import get_default_runner
from mol.runner.dumper import DataDumper
from mol.runner.inference import fix_cterminal_carboxyl_oxygens, update_inference_configs
from mol.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input-json', type=Path, required=True)
    p.add_argument('--indices', default='0')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--path', choices=('single', 'batch'), default='single')
    p.add_argument('--input-batch-size', type=int, default=1,
                   help='For the batch path, repeat each selected example with independent consecutive seeds')
    p.add_argument('--cycles', type=int, default=20)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--sample-chunk', type=int, default=1)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--modes', default='baseline,aggregation,sampler,graph,stable,lru,combined')
    p.add_argument('--allow-busy-gpu', action='store_true')
    p.add_argument('--gpu-memory-fraction', type=float, default=.4)
    args = p.parse_args()
    modes = args.modes.split(',')
    if modes[0] != 'baseline' or set(modes) - {'baseline', 'aggregation', 'sampler', 'graph', 'stable', 'lru', 'combined', 'runner'}:
        p.error('Begin with baseline; use only supported experiment modes')
    if min(args.cycles, args.steps, args.samples, args.sample_chunk, args.input_batch_size) < 1 or args.repeats < 1:
        p.error('Counts must be positive')
    if args.path == 'single' and args.input_batch_size != 1:
        p.error('Multiple inputs require --path batch')
    args.output.mkdir(parents=True, exist_ok=False)
    require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jobs = json.loads(args.input_json.read_text())
    indices = [int(x) for x in args.indices.split(',')]
    if any(i < 0 or i >= len(jobs) for i in indices):
        p.error('Input index out of range')
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in (root / 'mol').rglob('*') if p.suffix in {'.py', '.cu', '.cpp', '.h'})
    sources += [Path(__file__).resolve(), root / 'scripts/benchmark_diffusion_compile.py']
    report = {
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'input_sha256': file_digest(args.input_json), 'checkpoint_sha256': file_digest(args.checkpoint),
        'source_sha256': {str(p.relative_to(root)): file_digest(p) for p in sources},
        'status': 'running', 'passes': [],
        'timing_scope': 'Prediction including transfers and confidence, excluding preparation/comparison/writing. Each mode has its own first sequence then repeats; graphs are released between modes. Shared-GPU timings are provisional, not a production speedup claim.',
        'graph_policy': 'Manual graph arms: max one exact signature and accounted cache allocation budget 12 GiB. Runner arm: configured graph capacity. No added padding or compilation.',
        'runner_defaults': {key: inference_configs['infer_setting'][key]
                            for key in ('compile_diffusion', 'cuda_graphs', 'cuda_graph_max_cache_size')},
    }

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    with patch.dict(inference_configs, {'dump_dir': str(args.output / 'runner')}):
        runner = get_default_runner(
            seeds=[args.seed], device='cuda', dtype='bf16', use_msa=False,
            batch_size=1 if args.path == 'single' else max(2, args.input_batch_size),
            n_cycle=args.cycles, n_step=args.steps, n_sample=args.samples,
            load_checkpoint_path=str(args.checkpoint.resolve()),
            trimul_kernel='auto', triatt_kernel='auto',
            memory_settings={'infer_setting.compile_diffusion': False, 'infer_setting.cuda_graphs': False,
                             'infer_setting.sample_diffusion_chunk_size': args.sample_chunk,
                             'infer_setting.confidence_sample_chunk_size': 1,
                             'infer_setting.batch_max_padding_ratio': 1.0},
        )
    report['checkpoint_load'] = runner.checkpoint_load_report
    report['effective_config'] = json.loads(json.dumps(runner.configs.to_dict(), default=str))
    report['layernorm_modules'] = {
        'fused': sum(isinstance(m, FusedLayerNorm) for m in runner.model.modules()),
        'torch': sum(isinstance(m, OpenFoldLayerNorm) for m in runner.model.modules()),
    }
    if not report['layernorm_modules']['fused'] or report['layernorm_modules']['torch']:
        raise AssertionError('This benchmark requires the default fused LayerNorm')
    runner.configs.input_json_path = str(args.input_json.resolve())
    dataset = InferenceDataset(runner.configs)
    prepared = {}
    for index in dict.fromkeys(indices):
        seed_everything(args.seed, deterministic=False)
        data, atoms, error = dataset[index]
        if error:
            raise RuntimeError(error)
        groups = protein_token_groups(atoms, data['input_feature_dict']['atom_to_token_idx'], int(data['N_token']))
        prepared[index] = (data, atoms, groups, jobs[index].get('ipsae_interface'))
    report['inputs'] = [{'index': i, 'name': jobs[i]['name'], 'tokens': int(prepared[i][0]['N_token']),
                         'atoms': len(prepared[i][1]), 'protein_groups': len(prepared[i][2])} for i in dict.fromkeys(indices)]
    reference, file_reference, rng_reference = {}, {}, {}
    transformer = runner.model.diffusion_module.diffusion_transformer
    replay = None
    try:
        for mode in modes:
            runner.close()
            runner.configs.infer_setting.cuda_graphs = report['runner_defaults']['cuda_graphs'] if mode == 'runner' else False
            if mode == 'runner' and (report['runner_defaults']['compile_diffusion'] or not report['runner_defaults']['cuda_graphs']):
                raise AssertionError('Runner comparison requires eager replay enabled by default')
            replay = None
            if mode in {'graph', 'stable', 'lru', 'combined'}:
                replay = ExperimentalReplayCache(
                    transformer.forward, transformer, ReplaySession(), max_graphs=1,
                    stable_inputs=('z', 'projected_pair_biases', 'extra_attn_bias') if mode in {'stable', 'combined'} else (),
                    evict=mode in {'lru', 'combined'}, max_cache_bytes=12 * 1024**3,
                )
            for repeat in range(args.repeats + 1):
                for request, index in enumerate(indices):
                    original, atoms, groups, interface = prepared[index]
                    request_inputs = [copy.deepcopy(original) for _ in range(args.input_batch_size)]
                    request_seeds = [args.seed + i for i in range(args.input_batch_size)]
                    data = request_inputs[0]
                    runner.update_model_configs(update_inference_configs(copy.deepcopy(runner.configs), int(data['N_token'])))
                    seed_everything(args.seed, deterministic=False)
                    competitors = require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
                    session = replay.session if replay else getattr(runner, 'diffusion_compilation_session', None)
                    if session is None and mode == 'runner':
                        session = EagerReplaySession()
                    before = asdict(session) if session else {}
                    with ExitStack() as stack:
                        aggregation = stack.enter_context(cache_aggregation_metadata()) if mode in {'aggregation', 'combined'} else None
                        if mode in {'sampler', 'combined'}:
                            stack.enter_context(cache_sampler_branches())
                        if replay:
                            stack.enter_context(patch.object(transformer, 'forward', replay))
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        predictions = [runner.predict(data)] if args.path == 'single' else runner.predict_batch(request_inputs, request_seeds)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        peak = torch.cuda.max_memory_allocated()
                    rng = (torch.cuda.get_rng_state().cpu(), torch.get_rng_state(), np.random.get_state())
                    result = summarize(predictions, [groups] * args.input_batch_size, [interface] * args.input_batch_size)
                    if index not in reference:
                        reference[index], rng_reference[index] = result, rng
                    comparison = differences(result, reference[index])
                    exact = all(v['exact'] if isinstance(v, dict) else v for row in comparison for v in row.values())
                    old_rng = rng_reference[index]
                    rng_exact = torch.equal(rng[0], old_rng[0]) and torch.equal(rng[1], old_rng[1]) and np.array_equal(rng[2][1], old_rng[2][1]) and rng[2][2:] == old_rng[2][2:]
                    label = f'{mode}_{repeat}_{request}'
                    directory = args.output / label
                    for pred, seed in zip(predictions, request_seeds, strict=True):
                        pred['coordinate'] = fix_cterminal_carboxyl_oxygens(pred['coordinate'], atoms)
                        DataDumper(str(directory)).dump(
                            dataset_name='', pdb_id=jobs[index]['name'], seed=seed,
                            pred_dict=pred, atom_array=atoms, ipsae_interface=interface,
                            entity_poly_type={k: v for k, v in original['entity_poly_type'].items() if v != 'non-polymer'},
                        )
                    hashes = {str(p.relative_to(directory)): file_digest(p) for p in directory.rglob('*') if p.suffix in {'.cif', '.json'}}
                    if index not in file_reference:
                        file_reference[index] = hashes
                    record = {
                        'label': label, 'mode': mode, 'repeat': repeat, 'input_index': index,
                        'input_seeds': request_seeds,
                        'seconds': seconds, 'peak_allocated_bytes': peak,
                        'all_outputs_exact': exact, 'rng_exact': rng_exact,
                        'files_byte_identical': hashes == file_reference[index],
                        'comparison': comparison, 'file_hashes': hashes,
                        'competing_processes_before': competitors,
                        'competing_processes_after': require_unoccupied_gpu(allow_busy=args.allow_busy_gpu),
                        'aggregation': {'builds': aggregation.builds, 'hits': aggregation.hits} if aggregation else None,
                        'execution': {'compile_diffusion': runner.model.configs.infer_setting.compile_diffusion,
                                      'cuda_graphs': runner.model.configs.infer_setting.cuda_graphs,
                                      'owner': 'runner' if mode == 'runner' else 'benchmark'},
                    }
                    session = replay.session if replay else getattr(runner, 'diffusion_compilation_session', None)
                    record['session'] = asdict(session) if session else None
                    record['session_delta'] = {k: getattr(session, k) - v for k, v in before.items() if isinstance(v, int) and not isinstance(v, bool)} if session else None
                    report['passes'].append(record)
                    save()
                    print(json.dumps({k: record[k] for k in ('label', 'seconds', 'all_outputs_exact', 'rng_exact', 'files_byte_identical', 'aggregation', 'session_delta')}), flush=True)
                    del predictions, pred, result, data, request_inputs
            if replay:
                replay.close()
                replay = None
        report['status'] = 'completed'
        report['all_outputs_exact'] = all(p['all_outputs_exact'] and p['rng_exact'] and p['files_byte_identical'] for p in report['passes'])
        report['source_hashes_match'] = all(file_digest(root / p) == h for p, h in report['source_sha256'].items())
        save()
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = repr(exc); save()
        raise
    finally:
        if replay:
            replay.close()
        runner.close()


if __name__ == '__main__':
    main()
