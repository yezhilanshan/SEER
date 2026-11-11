import numpy as np
import torch
import pytest
import copy
import json


def records_fixture():
    rows=[]
    for k,(subject,role) in enumerate([('subject1','train'),('subject1','train'),('subject7','validation')]):
        x=np.zeros((40,4),np.float32); x[:20,0]=1; x[20:,1]=1
        labels=np.array([0]*15+[-100]*5+[1]*15+[-100]*5)
        target=np.exp(-.5*((np.arange(40)[:,None]-np.array([15,20,35]))/3)**2).max(1).astype(np.float32)
        rows.append(dict(file=f'{subject}_{k}',subject=subject,role=role,frames=40,x=x,labels=labels,
                         boundary_target=target,boundary_mask=np.ones(40,bool),known_frames=labels>=0,
                         boundaries=[15,20,35],instances=[dict(start=0,stop=15,label=0,instance_id=0),
                         dict(start=20,stop=35,label=1,instance_id=1)]))
    return rows


def test_mvp_run_shares_local_weights_calibrates_train_only_and_replays(tmp_path):
    from continuous_mmwave.event_gating.mvp.runner import run_experiment,load_checkpoint
    torch.set_num_threads(1)
    settings=dict(mvp=dict(feature_dim=4,hidden_dim=8,num_classes=3),
                  training=dict(action_steps=2,evidence_steps=2,seed=3),
                  calibration=dict(thresholds=[.5,1.],taus=[1.]))
    out=tmp_path/'run'
    result=run_experiment(settings,records_fixture(),out,synthetic_only=True)
    assert result['synthetic_only'] and not result['real_harrad_training']
    assert set(result['arms'])=={'B0','B1','B2','B3'}
    assert result['arms']['B0']['evidence_sha256']==result['arms']['B1']['evidence_sha256']==result['arms']['B2']['evidence_sha256']
    assert result['fit_files']==['subject1_0'] and result['calibration_files']==['subject1_1']
    assert result['validation_files']==['subject7_2']
    assert result['checkpoint_replay_passed']
    assert (out/'B3'/'subject7_2-trace.npz').exists()
    assert (out/'report.md').exists()
    assert (out/'go-no-go.json').exists()
    decision=json.loads((out/'go-no-go.json').read_text())
    assert decision['status']=='synthetic_only'
    assert not decision['research_go']
    summary=json.loads((out/'summary.json').read_text())
    assert set(summary['B3']['boundary'])=={'0.2','0.5','1.0'}
    assert set(summary['B3']['events'])=={'0.3','0.5'}
    model=load_checkpoint(out/'model.pt')
    assert model['format']=='saef-mvp-v1'
    from continuous_mmwave.event_gating.mvp.runner import evaluate_checkpoint
    replay=evaluate_checkpoint(out/'model.pt',records_fixture(),tmp_path/'evaluation')
    assert replay['arms']==result['arms']
    changed=copy.deepcopy(records_fixture())
    changed[2]['instances'][0]['stop']=14
    with pytest.raises(ValueError,match='identity'):
        evaluate_checkpoint(out/'model.pt',changed,tmp_path/'changed')


def test_training_skips_unlabelled_recordings_but_preserves_budget():
    from continuous_mmwave.event_gating.mvp.runner import train_head
    from continuous_mmwave.event_gating.mvp.model import ActionHead, MVPConfig
    rows=records_fixture()[:2]
    rows[0]['labels'][:]=-100
    log=train_head(ActionHead(MVPConfig(feature_dim=4,num_classes=3)),rows,3,.001,2,'cpu')
    assert log['successful_updates']==3
    assert log['eligible_files']==['subject1_1']
    with pytest.raises(ValueError,match='semantic supervision'):
        train_head(ActionHead(MVPConfig(feature_dim=4)),rows[:1],1,.001,2,'cpu')


def test_run_rejects_record_time_grid_mismatch(tmp_path):
    from continuous_mmwave.event_gating.mvp.runner import run_experiment
    rows=records_fixture()
    rows[0]['frame_period_s']=.2
    with pytest.raises(ValueError,match='time grid'):
        run_experiment(dict(mvp=dict(feature_dim=4)),rows,tmp_path/'bad-time',synthetic_only=True)


def test_run_rejects_calibration_without_internal_boundaries(tmp_path):
    from continuous_mmwave.event_gating.mvp.runner import run_experiment
    rows=records_fixture()
    rows[1]['instances']=[]
    rows[1]['boundaries']=[]
    rows[1]['labels'][:]=-100
    rows[1]['known_frames'][:]=False
    rows[1]['boundary_mask'][:]=False
    settings=dict(mvp=dict(feature_dim=4,hidden_dim=8,num_classes=3),
                  training=dict(action_steps=1,evidence_steps=1),
                  calibration=dict(thresholds=[.5],taus=[1.]))
    out=tmp_path/'no-calibration-support'
    with pytest.raises(ValueError,match='calibration requires.*boundaries'):
        run_experiment(settings,rows,out,synthetic_only=True)
    assert not (out/'action-audit.json').exists()
    assert not (out/'complete.json').exists()


def test_audit_includes_per_record_timing_mask_and_source_proof(tmp_path):
    from test_saef_mvp_data_metrics import assets
    from continuous_mmwave.event_gating.mvp.data import load_records
    from continuous_mmwave.event_gating.mvp.runner import audit_records
    settings,_=assets(tmp_path)
    report=audit_records(load_records(settings),settings)
    row=report['records'][0]
    assert row['feature_dim']==128
    assert row['duration_s']==pytest.approx(160/15)
    assert row['known_frames']==80 and row['unknown_frames']==80
    assert row['interval_convention']=='[start,stop)'
    assert row['continuous_hidden_sha256']
    assert report['source_hashes']


def test_cli_smoke_exercises_new_entrypoint(tmp_path):
    from continuous_mmwave.event_gating.mvp.runner import main
    assert main(['smoke','--output-dir',str(tmp_path/'smoke')])==0


def test_real_experiment_requires_verified_natural_features(tmp_path):
    from continuous_mmwave.event_gating.mvp.runner import run_experiment
    with pytest.raises(ValueError,match='provenance'):
        run_experiment({},records_fixture(),tmp_path/'unverified')
