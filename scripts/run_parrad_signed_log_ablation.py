"""Sequential paired Stage1 runs; train/validation only, conditional repeats fixed in advance."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer
from continuous_mmwave.radar_text.signed_log_ablation import parameter_sha
from report_parrad_signed_log_ablation import build_report


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=Path(p);temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(p)


def prepare_pair(root,seed,base):
    pair=root if seed==17 else root/f'seed-{seed}'
    pair.mkdir(parents=True,exist_ok=True)
    configs={}
    for arm,mode in [('P0','signed_log_zscore'),('P1','raw_zscore')]:
        value=deepcopy(base);value['model']['normalization']=mode;value['training']['seed']=seed
        configs[arm]=value;dump(pair/f'{arm}-config.json',value)
    stripped=[deepcopy(c) for c in configs.values()]
    for c in stripped:c['model'].pop('normalization')
    assert stripped[0]==stripped[1],'more than normalization differs'
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=RDResNetPretrainer(**configs['P0']['model'])
    initial=dict(state_dict=model.state_dict(),parameter_sha=parameter_sha(model),
                 cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seed=seed)
    # Both constructors must yield identical tensors before any training begins.
    torch.manual_seed(seed);other=RDResNetPretrainer(**configs['P1']['model'])
    assert parameter_sha(other)==initial['parameter_sha']
    path=pair/'shared-initial-parameters.pt';torch.save(initial,path)
    dump(pair/'pair-preflight.json',dict(seed=seed,initial_parameter_sha256=initial['parameter_sha'],
        shared_file_sha256=sha(path),configs_only_differ_in_normalization=True,
        window_protocol=dict(window_seconds=12,stride_seconds=6,min_seconds=8,max_actions=3),
        validation_only=True,prepared_at=datetime.now().astimezone().isoformat()))
    return pair,path


def verify_pair(pair):
    initial=[json.loads((pair/arm/'initialization.json').read_text()) for arm in ['P0','P1']]
    orders=[json.loads((pair/arm/'sampler-sha256.json').read_text()) for arm in ['P0','P1']]
    provenance=[json.loads((pair/arm/'provenance.json').read_text()) for arm in ['P0','P1']]
    assert initial[0]['parameter_sha256']==initial[1]['parameter_sha256']
    assert initial[0]['shared_file_sha256']==initial[1]['shared_file_sha256']
    assert sha(pair/'P0/initial-parameters.pt')==sha(pair/'P1/initial-parameters.pt')
    assert orders[0]==orders[1] and len(orders[0])==provenance[0]['config']['training']['epochs']
    for key in ['subjects','class_weights','label_counts_train_windows','window_ids_sha256','numerics','source_sha256']:
        assert provenance[0][key]==provenance[1][key],key
    for arm in ['P0','P1']:
        assert set(json.loads((pair/arm/'results.json').read_text()))=={'validation'}
        assert json.loads((pair/arm/'status.json').read_text())['test_values_accessed'] is False
    proof=dict(initial_parameters_identical=True,initial_file_sha256=initial[0]['shared_file_sha256'],
        initial_parameter_sha256=initial[0]['parameter_sha256'],all_epoch_sample_orders_identical=True,
        paired_epochs=len(orders[0]),class_weights_identical=True,numerics_identical=True,
        test_values_accessed=False,bitwise_cuda_determinism_claimed=False)
    dump(pair/'fairness.json',proof)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--base-config',required=True,type=Path)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'source-lines.json',root/'source-lines.json')
    base=json.loads(args.base_config.read_text());script=Path(__file__).with_name('train_rd_resnet18_scratch.py')
    assert base['training']['seed']==17 and base['training']['epochs']==20
    expected={'train':['subject_'+s for s in ['01','02','03','04','07','08','09','13','19']],
              'validation':['subject_05','subject_12','subject_16'],'test':['subject_06','subject_17','subject_18']}
    policy=json.loads((Path(base['split_directory'])/'protocol.json').read_text())
    assert policy['subjects']==expected
    assert policy['window_protocol']==dict(window_seconds=12,stride_seconds=6,min_seconds=8,max_actions=3)
    historical=Path('/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/rd_resnet18_scratch_expanded_20261011_v1/config.json')
    if historical.exists():assert json.loads(historical.read_text())==base
    started=datetime.now().astimezone().isoformat()
    dump(root/'experiment-policy.json',dict(started=started,base_config=base,first_seed=17,
        repeat_seeds=[29,43],repeat_trigger_absolute_subject_macro_f1_delta=.01,
        threshold_scope='absolute seed17 validation subject-mean Macro-F1 difference, fixed before training',
        test_values_accessed=False,training_budget='20 epochs per arm, no early stopping',
        action_phrases=json.loads((Path(policy['dataset'])/'action_phrases.json').read_text()),
        deterministic_note='Paired initial tensors/RNG states and private sampling RNG; CUDA bitwise determinism not guaranteed'))
    def status(state,**extra):
        dump(root/'status.json',dict(state=state,pid=os.getpid(),started=started,**extra))
        print(json.dumps(dict(state=state,**extra)),flush=True)
    try:
        seeds=[17]
        for seed in seeds:
            pair,initial=prepare_pair(root,seed,base)
            for arm in ['P0','P1']:
                status('smoke' if args.smoke else 'training',seed=seed,arm=arm)
                command=[sys.executable,str(script),'--config',str(pair/f'{arm}-config.json'),'--output',str(pair/arm),
                         '--paired-initial',str(initial),'--validation-only']
                if args.smoke:command.append('--smoke')
                with (pair/f'{arm}-run.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                build_report(root)
            if args.smoke:
                for arm in ['P0','P1']:
                    assert json.loads((pair/arm/'status.json').read_text())['state']=='smoke_complete'
                a,b=[json.loads((pair/arm/'initialization.json').read_text()) for arm in ['P0','P1']]
                assert a['parameter_sha256']==b['parameter_sha256'] and a['shared_file_sha256']==b['shared_file_sha256']
                status('smoke_complete',test_values_accessed=False);return
            verify_pair(pair)
            if seed==17:
                metrics=[json.loads((pair/arm/'results.json').read_text())['validation']['subject_macro_f1'] for arm in ['P0','P1']]
                delta=metrics[1]-metrics[0];repeat=abs(delta)>=.01
                dump(root/'repeat-decision.json',dict(seed17_subject_macro_f1_P0=metrics[0],seed17_subject_macro_f1_P1=metrics[1],
                     delta_P1_minus_P0=delta,predeclared_absolute_threshold=.01,repeat=repeat))
                if repeat:seeds.extend([29,43])
            build_report(root)
        status('complete',completed_seeds=seeds,test_values_accessed=False)
        build_report(root)
        dump(root/'artifact-manifest.json',{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))
             for p in root.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'})
    except BaseException as error:
        status('failed',error=repr(error));build_report(root);raise


if __name__=='__main__':main()
