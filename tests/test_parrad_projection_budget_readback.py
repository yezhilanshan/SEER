"""Exercise readback on actual files and optimizer states, never production metrics."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def verifier():
    spec=importlib.util.spec_from_file_location('budget_readback',Path('.audit-staging/verify-projection-budget-v1.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def files(tmp_path, *, stopped=False):
    module=verifier()
    before,after=tmp_path/'reference',tmp_path/'extended'
    baseline=dict(step=0,score=.6,retention={'stream':{'event_to_text':.9,'margin':.1}},
                  semantic_complete=True,eligible=True)
    warm=dict(model_state={'teacher.weight':torch.tensor([[.7]]),'projection.weight':torch.tensor([[.7]]),
                           'heads.weight':torch.zeros(1,1)})
    common=dict(status='diagnostics_complete',settings={'held_subject':'subject_04','steps':20},
        provenance=dict(source_manifest_sha256='synthetic',stage2_checkpoint_sha256='synthetic',config_sha256='synthetic',
                        code_sha256={}),validation_test_read=False,held_calibration_numeric_read=False,
        production_model_changed=False,full_pilot_launched=False)
    for root in (before,after):
        (root/'experiment').mkdir(parents=True)
        module.write_json(root/'experiment'/'baseline.json',baseline)
        torch.save(warm,root/'experiment'/'common-warmup.pt')
    old=copy.deepcopy(common); module.write_json(before/'manifest.json',old)
    new=copy.deepcopy(common)
    new['settings'].update(steps=100,budget_limit=100,milestones=[5,10,20,50,100])
    new['conditions']=[]
    for name in ('frozen-control','joint-relation1','joint-relation10'):
        root=after/'experiment'/name; root.mkdir()
        old_root=before/'experiment'/name; old_root.mkdir()
        parameter=torch.nn.Parameter(torch.tensor([[.7]]))
        optimizer=torch.optim.AdamW([parameter],lr=1e-5,weight_decay=0.)
        history=[copy.deepcopy(baseline)]; old_history=[copy.deepcopy(baseline)]
        torch.save({'projection_state':{'weight':parameter.detach().clone()},'step':0},root/'step-00.pt')
        steps=0 if name=='frozen-control' else 100
        for step in range(1,steps+1):
            optimizer.zero_grad(); parameter.square().sum().backward(); optimizer.step()
            failed=stopped and step==100
            fit=dict(converged=not failed,heads=[{'converged':False}] if failed else [{'converged':True}]*3)
            module.write_json(root/f'head-fit-{step:02d}.json',fit)
            if step in (5,10,20,50,100) and not(stopped and step==100):
                row=copy.deepcopy(baseline); row.update(step=step,score=.6+step*.001)
                history.append(row)
                checkpoint=dict(projection_state={'weight':parameter.detach().clone()},step=step)
                torch.save(checkpoint,root/f'step-{step:02d}.pt')
                if step<=20:
                    old_history.append(copy.deepcopy(row)); torch.save(checkpoint,old_root/f'step-{step:02d}.pt')
        module.write_json(old_root/'history.json',old_history)
        is_stopped=stopped and name!='frozen-control'
        selected=0 if is_stopped else steps
        state=copy.deepcopy(warm['model_state']); state['projection.weight']=parameter.detach().clone()
        signature={'synthetic':True}
        saved=dict(step=steps,phase='stopped' if is_stopped else 'ready',model_state=state,
                   optimizer_state=optimizer.state_dict(),history=history,signature=signature,
                   head_report={'converged':False} if is_stopped else {'converged':True})
        torch.save(saved,root/'last.pt'); module.write_json(root/'history.json',history)
        new['conditions'].append(dict(name=name,completed_projection_steps=steps,signature=signature,
             status='head_fit_failed' if is_stopped else 'locked',selected_step=selected,
             selected_sha256=module.sha256(root/f'step-{selected:02d}.pt')))
    module.write_json(after/'manifest.json',new)
    return module,before,after


def test_successful_hundred_step_readback_checks_prefix_and_optimizer(tmp_path):
    module,before,after=files(tmp_path)
    result=module.verify(before,after)
    assert [r['completed_steps'] for r in result['conditions']]==[0,100,100]
    assert [r['prefix20_exact'] for r in result['conditions']]==[None,True,True]


@pytest.mark.parametrize('missing',['checkpoint','history'])
def test_reached_twenty_step_evidence_is_required_even_when_selected_hundred(tmp_path,missing):
    module,before,after=files(tmp_path)
    root=after/'experiment'/'joint-relation1'
    if missing=='checkpoint':
        (root/'step-20.pt').unlink()
    else:
        saved=torch.load(root/'last.pt',weights_only=False)
        saved['history']=[r for r in saved['history'] if r['step']!=20]
        torch.save(saved,root/'last.pt'); module.write_json(root/'history.json',saved['history'])
    with pytest.raises(AssertionError):
        module.verify(before,after)
    assert not (after/'result-readback-checks-v1.json').exists()


def test_failed_refit_at_hundred_is_reported_without_rejecting_earlier_successful_fits(tmp_path):
    module,before,after=files(tmp_path,stopped=True)
    result=module.verify(before,after)
    for condition in result['conditions'][1:]:
        assert condition['status']=='head_fit_failed' and condition['completed_steps']==100
        assert condition['prefix20_exact'] is True
        assert [r['step'] for r in condition['milestones']]==[0,5,10,20,50]
