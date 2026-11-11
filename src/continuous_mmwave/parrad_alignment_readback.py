"""Verify available evidence in failed runs without certifying completion."""
import json
import math
from pathlib import Path
import torch
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_radar_alignment import build_tasks,training_batches,validate_predictions,development_gates
from continuous_mmwave.parrad_single_event_qa import score_predictions
from continuous_mmwave.parrad_stage4_runner import sha


def check_available(root,settings):
    root=Path(root);read=lambda name:json.loads((root/name).read_text())
    source=read('source.json')
    assert all(sha(p)==h for p,h in source['protected_sha256'].items())
    payload=torch.load(Path(settings['source_output'])/'events.pt',map_location='cpu',weights_only=False)
    assert payload['upstream_digest']==source['upstream_digest']
    events={e['uid']:e for e in payload['events']};tasks=build_tasks(payload['events'],42)
    expected=[[t['uid'] for t in batch] for batch in training_batches(tasks,42)]
    assert read('tasks.json')==tasks and read('batches.json')==expected
    assert sha(root/'tasks.json')==source['task_sha256'] and sha(root/'batches.json')==source['batch_sha256']
    checked=[];arms={}
    def predictions(name):
        result=read(name);leaf=Path(name).name;cohort='fit' if leaf.startswith('fit-') else 'dev'
        mode='text' if leaf in ('fit-text.json','dev-text.json') else 'none' if leaf=='dev-none.json' else 'swapped' if leaf=='dev-swapped.json' else 'raw'
        selected=[t for t in tasks if t['cohort']==cohort and (mode!='swapped' or t['kind']=='presence')]
        assert result['mode']==mode;validate_predictions(result,selected,events)
        assert result['metrics']==score_predictions(result['rows'])
        if mode=='swapped': assert result['evidence_metrics']==score_predictions([dict(r,reference=r['evidence_reference']) for r in result['rows']])
        checked.append(name);return result
    for name in ('fit-text.json','dev-text.json','fit-before.json','dev-before.json','dev-none.json'):
        if (root/name).exists():predictions(name)
    for arm,weight in (('ce',0.),('distill',.5)):
        folder=root/arm
        if not folder.exists():continue
        log_path=folder/'training.jsonl';log=[json.loads(s) for s in log_path.read_text().splitlines()] if log_path.exists() else []
        assert len(log)<=500 and [r['step'] for r in log]==list(range(1,len(log)+1)) and [r['task_uids'] for r in log]==expected[:len(log)]
        assert all(r['weight']==weight and all(math.isfinite(r[k]) for k in ('loss','ce','kl','gradient_norm','loss_scale')) for r in log)
        checks=[];states={}
        for step in (100,250,500):
            path=folder/f'connector-step{step}.pt'
            if not path.exists():continue
            saved=torch.load(path,map_location='cpu',weights_only=False);optimizer=saved['optimizer_state']['state']
            assert step<=len(log) and saved['step']==step and saved['source']==source and saved['arm']==arm
            assert len(optimizer)==6 and {int(v['step']) for v in optimizer.values()}=={step}
            assert all(torch.isfinite(v).all() for v in saved['model_state'].values())
            assert all(torch.isfinite(v[k]).all() for v in optimizer.values() for k in ('exp_avg','exp_avg_sq'))
            EventQAAdapter(**saved['model_settings']).load_state_dict(saved['model_state'],strict=True)
            checks.append(step);states[step]=saved['model_state']
        export=folder/'connector-inference.pt'
        if export.exists():
            value=torch.load(export,map_location='cpu',weights_only=False)
            assert 500 in states and value['step']==500 and value['source']==source and value['arm']==arm
            assert set(value['model_state'])==set(states[500]) and all(torch.equal(v,value['model_state'][k]) for k,v in states[500].items())
        available={}
        for name in ('fit-after.json','dev-after.json','dev-swapped.json'):
            if (folder/name).exists(): available[name]=predictions(arm+'/'+name)
        retrieval=folder/'retrieval-after.json'
        if retrieval.exists(): assert json.loads(retrieval.read_text())==read('retrieval-before.json')
        complete=False
        if (folder/'result.json').exists():
            native=read(arm+'/result.json');assert len(log)==500 and checks==[100,250,500] and export.exists()
            assert set(available)=={'fit-after.json','dev-after.json','dev-swapped.json'} and retrieval.exists()
            gates=development_gates(available['dev-after.json']['metrics'])
            assert native['updates']==500 and native['source']==source and native['weight']==weight
            assert native['gates']==gates and native['development_pass']==all(gates.values())
            assert native['after']=={'fit':available['fit-after.json']['metrics'],'dev':available['dev-after.json']['metrics']}
            assert native['inference_sha256']==sha(export);complete=True
        arms[arm]=dict(logged_updates=len(log),checked_checkpoint_steps=checks,completed_arm_artifacts_checked=complete)
    return dict(source_checks_performed=True,available_prediction_checks=checked,arms=arms,
        complete_checks_performed=False,partial_readback=True)
