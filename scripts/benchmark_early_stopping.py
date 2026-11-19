"""Compare optional first-sample screening against the same full sample stream."""
import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark_diffusion_compile import differences, file_digest, require_unoccupied_gpu, summarize
from mol.config.inference import inference_configs
from mol.data.inference.infer_dataloader import InferenceDataset
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
    parser.add_argument('--input-json', type=Path, default=Path('examples/examples_with_template/example_9fm7.json'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--allow-busy-gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in (root/'mol').rglob('*') if p.suffix in {'.py','.cu','.cpp','.h'}) + [Path(__file__).resolve()]
    report = {'status':'running', 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'source_hashes':{str(p.relative_to(root)):file_digest(p) for p in sources},
              'checkpoint_sha256':file_digest(args.checkpoint), 'input_sha256':file_digest(args.input_json),
              'passes':[], 'note':'Shared-GPU correctness check. The continue arm deliberately suppresses the gate decision to check unchanged later samples; it is not a measured natural pass rate.'}

    def save():
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    save()
    with patch.dict(inference_configs, {'dump_dir':str(args.output/'runner')}):
        runner = get_default_runner(
            device='cuda', dtype='bf16', use_msa=False, use_template=False,
            seeds=[101], batch_size=args.batch_size, n_cycle=args.cycles,
            n_step=args.steps, n_sample=args.samples, load_checkpoint_path=str(args.checkpoint),
            memory_settings={'infer_setting.early_stop_ipsae_threshold':.05,
                             'infer_setting.sample_diffusion_chunk_size':1,
                             'infer_setting.confidence_sample_chunk_size':1,
                             'infer_setting.batch_max_padding_ratio':1.0},
        )
    runner.configs.input_json_path = str(args.input_json.resolve())
    dataset = InferenceDataset(runner.configs)
    seed_everything(101, deterministic=False)
    data, atoms, error = dataset[0]
    if error:
        raise RuntimeError(error)
    groups = data['input_feature_dict']['ipsae_early_stop']['groups']
    interface = data.get('ipsae_interface')
    report['input'] = {'name':data['sample_name'],'tokens':int(data['N_token']),'protein_groups':len(groups)}
    report['layernorm'] = {'fused':sum(isinstance(m,FusedLayerNorm) for m in runner.model.modules()),
                           'torch':sum(isinstance(m,OpenFoldLayerNorm) for m in runner.model.modules())}
    assert report['layernorm']['fused'] and not report['layernorm']['torch']
    references = None
    reference_rng = None
    reference_cifs = {}
    original_gate = FirstSampleGate.__call__

    def force_continue(gate, coordinates):
        original_gate(gate, coordinates)
        gate.stopped = False
        return False

    try:
        for mode in ('baseline','screen','stop','continue'):
            threshold = None if mode == 'baseline' else (1.0 if mode == 'stop' else .05)
            runner.configs.infer_setting.early_stop_ipsae_threshold = threshold
            runner.update_model_configs(update_inference_configs(copy.deepcopy(runner.configs),int(data['N_token'])))
            inputs = [copy.deepcopy(data) for _ in range(args.batch_size)]
            seeds = [101+2*i for i in range(args.batch_size)]
            seed_everything(101, deterministic=False)
            competitors = require_unoccupied_gpu(allow_busy=args.allow_busy_gpu)
            gate = force_continue if mode == 'continue' else original_gate
            with patch.object(FirstSampleGate, '__call__', gate):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                predictions = [runner.predict(inputs[0])] if args.batch_size == 1 else runner.predict_batch(inputs,seeds)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                peak = torch.cuda.max_memory_allocated()
            rng = torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state()
            results = summarize(predictions,[groups]*args.batch_size,[interface]*args.batch_size)
            if references is None:
                references,reference_rng = results,rng
            expected=[]
            for pred,ref in zip(predictions,references,strict=True):
                n=pred['coordinate'].shape[0]
                expected.append({key:value[:n] for key,value in ref.items()})
                validate_prediction(pred,n_atom=len(atoms),n_token=int(data['N_token']),n_sample=args.samples,
                                    early_stop_threshold=threshold)
            comparisons=differences(results,expected)
            exact=all(v['exact'] if isinstance(v,dict) else v for row in comparisons for v in row.values())
            rng_exact=torch.equal(rng[0],reference_rng[0]) and torch.equal(rng[1],reference_rng[1]) and np.array_equal(rng[2][1],reference_rng[2][1]) and rng[2][2:]==reference_rng[2][2:]
            decisions=[p.get('sampling_early_stop') for p in predictions]
            for decision,ref in zip(decisions,references,strict=True):
                if decision:
                    assert decision['first_sample_score']==ref['ipsae'][0]['ipsae_min_10']
            for seed,pred in zip(seeds,predictions,strict=True):
                pred['coordinate']=fix_cterminal_carboxyl_oxygens(pred['coordinate'],atoms)
                DataDumper(str(args.output/mode),sorted_by_ranking_score=False).dump(
                    dataset_name='',pdb_id=data['sample_name'],seed=seed,pred_dict=pred,atom_array=atoms,
                    entity_poly_type={k:v for k,v in data['entity_poly_type'].items() if v!='non-polymer'},
                    ipsae_interface=interface)
            cifs={str(p.relative_to(args.output/mode)):file_digest(p) for p in (args.output/mode).rglob('*.cif')}
            if mode=='baseline': reference_cifs=cifs
            cifs_exact=all(reference_cifs.get(k)==v for k,v in cifs.items())
            record={'mode':mode,'seconds':elapsed,'peak_allocated_bytes':peak,'outputs_exact':exact,
                    'rng_exact':rng_exact,'cifs_exact':cifs_exact,'comparisons':comparisons,'decisions':decisions,
                    'samples':[int(p['coordinate'].shape[0]) for p in predictions],
                    'competing_processes':competitors,'cif_sha256':cifs}
            report['passes'].append(record); save()
            print(json.dumps({k:record[k] for k in ('mode','seconds','outputs_exact','rng_exact','cifs_exact','samples','decisions')}),flush=True)
            assert exact and rng_exact and cifs_exact,mode
            if mode=='stop': assert all(n==1 for n in record['samples'])
            if mode=='continue': assert all(n==args.samples for n in record['samples'])
            del predictions,pred,results,inputs
        report['source_hashes_match']=all(file_digest(root/p)==h for p,h in report['source_hashes'].items())
        report['status']='completed';save()
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);save();raise
    finally:
        runner.close()


if __name__=='__main__':
    main()
