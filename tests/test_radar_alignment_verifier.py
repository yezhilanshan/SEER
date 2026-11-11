import json
from pathlib import Path
import subprocess
import sys


def test_early_failure_archives_evidence_without_certifying_training(tmp_path):
    script=Path(__file__).resolve().parents[1]/'.audit-staging/verify-radar-alignment.py'
    assert script.exists(), 'missing independent readback and failure archive'
    root=tmp_path/'stage4_radar_alignment_test';root.mkdir()
    status=dict(status='failed',phase='load',reason='missing source',settings={},code_sha256={},actual_updates={'ce':0,'distill':0})
    (root/'status.json').write_text(json.dumps(status))
    logs=root.with_name(root.name+'.logs');logs.mkdir();(logs/'worker.log').write_text('failure preserved')
    result=subprocess.run([sys.executable,str(script),'--root',str(root)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((root/'independent-readback.json').read_text())
    assert report['status']=='failed' and report['original_failure']=='missing source'
    assert not report['complete_checks_performed']
    assert root.parent.joinpath('stage4-radar-alignment-test-results.tar.gz').exists()


def test_verification_error_still_archives_corrupt_original_evidence(tmp_path):
    script=Path(__file__).resolve().parents[1]/'.audit-staging/verify-radar-alignment.py'
    root=tmp_path/'stage4_radar_alignment_test';root.mkdir()
    status=dict(status='complete',phase='complete',settings={},code_sha256={},actual_updates={'ce':500,'distill':500})
    (root/'status.json').write_text(json.dumps(status))
    (root/'source.json').write_text('broken json for independent readback')
    result=subprocess.run([sys.executable,str(script),'--root',str(root)],capture_output=True,text=True)
    assert result.returncode!=0
    report=json.loads((root/'independent-readback.json').read_text())
    assert report['verification_status']=='failed' and not report['complete_checks_performed']
    assert root.parent.joinpath('stage4-radar-alignment-test-results.tar.gz').exists()


def test_failed_run_with_available_bad_source_does_not_skip_validation(tmp_path):
    script=Path(__file__).resolve().parents[1]/'.audit-staging/verify-radar-alignment.py'
    root=tmp_path/'stage4_radar_alignment_test';root.mkdir()
    status=dict(status='failed',phase='training',reason='nonfinite',settings={},code_sha256={},actual_updates={'ce':500,'distill':1})
    (root/'status.json').write_text(json.dumps(status))
    (root/'source.json').write_text('broken json for independent readback')
    result=subprocess.run([sys.executable,str(script),'--root',str(root)],capture_output=True,text=True)
    assert result.returncode!=0, 'available corrupt source was silently skipped'
    report=json.loads((root/'independent-readback.json').read_text())
    assert report['status']=='failed' and report['verification_status']=='failed'
    assert root.parent.joinpath('stage4-radar-alignment-test-results.tar.gz').exists()


def test_partial_readback_checks_completed_arm_and_rejects_wrong_optimizer_step(tmp_path):
    import torch
    from continuous_mmwave.models.qa_adapter import EventQAAdapter
    from continuous_mmwave.parrad_alignment_readback import check_available
    from continuous_mmwave.parrad_radar_alignment import build_tasks,training_batches,development_gates
    from continuous_mmwave.parrad_single_event_qa import score_predictions
    from continuous_mmwave.parrad_stage4_runner import sha
    root=tmp_path/'run';root.mkdir();cache=tmp_path/'cache';cache.mkdir();arm=root/'ce';arm.mkdir()
    events=[dict(uid=f'{c}:{i}',cohort=c,label=i) for c in ('fit','dev') for i in range(14)]
    torch.save(dict(events=events,upstream_digest='test'),cache/'events.pt')
    tasks=build_tasks(events,42);batches=training_batches(tasks,42)
    write=lambda name,value:root.joinpath(name).write_text(json.dumps(value))
    write('tasks.json',tasks);write('batches.json',[[t['uid'] for t in b] for b in batches])
    source=dict(protected_sha256={str(cache/'events.pt'):sha(cache/'events.pt')},upstream_digest='test',
        task_sha256=sha(root/'tasks.json'),batch_sha256=sha(root/'batches.json'))
    write('source.json',source)
    log=[dict(step=i,task_uids=[t['uid'] for t in b],weight=0.,loss=1.,ce=1.,kl=0.,gradient_norm=1.,loss_scale=256.) for i,b in enumerate(batches,1)]
    (arm/'training.jsonl').write_text('\n'.join(json.dumps(r) for r in log))
    model=EventQAAdapter(dim=4,bottleneck=2);optimizer=torch.optim.AdamW(model.parameters())
    for step in range(1,501):
        optimizer.zero_grad();sum(p.square().sum() for p in model.parameters()).backward();optimizer.step()
        if step in (100,250,500):
            saved=dict(step=step,source=source,arm='ce',model_settings=model.settings,model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
            torch.save(saved,arm/f'connector-step{step}.pt')
    torch.save({k:saved[k] for k in ('step','source','arm','model_settings','model_state')},arm/'connector-inference.pt')
    values={};lookup={t['uid']:t for t in tasks}
    for name,cohort,mode in [('fit-after.json','fit','raw'),('dev-after.json','dev','raw'),('dev-swapped.json','dev','swapped')]:
        selected=[t for t in tasks if t['cohort']==cohort and (mode!='swapped' or t['kind']=='presence')];rows=[]
        for t in selected:
            actual=t['event_uid'] if mode=='raw' else lookup[t['pair_id']+(':No' if t['answer']=='Yes' else ':Yes')]['event_uid']
            r=dict(t,reference=t['answer'],generated=t['answer'],correct=True,reader='text-qa-500',context_tokens=8,actual_event_uid=actual)
            if mode=='swapped':r['evidence_reference']='No' if t['answer']=='Yes' else 'Yes'
            rows.append(r)
        value=dict(mode=mode,rows=rows,metrics=score_predictions(rows))
        if mode=='swapped':value['evidence_metrics']=score_predictions([dict(r,reference=r['evidence_reference']) for r in rows])
        write('ce/'+name,value);values[name]=value
    write('retrieval-before.json',{'test':'same snapshot'});write('ce/retrieval-after.json',{'test':'same snapshot'})
    gates=development_gates(values['dev-after.json']['metrics'])
    write('ce/result.json',dict(updates=500,source=source,weight=0.,gates=gates,development_pass=all(gates.values()),
        after={c:values[c+'-after.json']['metrics'] for c in ('fit','dev')},inference_sha256=sha(arm/'connector-inference.pt')))
    report=check_available(root,dict(source_output=str(cache)))
    assert report['arms']['ce']['completed_arm_artifacts_checked'] and not report['complete_checks_performed']
    bad=torch.load(arm/'connector-step500.pt',weights_only=False)
    next(iter(bad['optimizer_state']['state'].values()))['step']=torch.tensor(499.)
    torch.save(bad,arm/'connector-step500.pt')
    import pytest
    with pytest.raises(AssertionError):check_available(root,dict(source_output=str(cache)))
