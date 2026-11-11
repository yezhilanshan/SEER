from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'.audit-staging'))
import proposal_loss_analysis as m

def fixture(root,dev_offset=0.):
    root.mkdir();spec=dict(encoder_index_sha256='synthetic-index',thresholds=[.3,.6,.9],fixed_peak_slots=[dict(id='local/peak/budget8',path='local',calibration_fp_budget=8,
        config=dict(family='peak',high=.6,lookahead_updates=2,nms_s=.72))])
    (root/'proposal-loss-spec-v1.json').write_text(json.dumps(spec))
    for arm in ('weighted','unweighted'):
        folder=root/arm;folder.mkdir()
        for stem,count,role in [('fit',18,'fit'),('proposal',8,'dev')]:
            arrays={};rows=[]
            for i in range(count):
                t=np.arange(60)*.18;b=np.array([1.8,7.2]);y=np.exp(-.5*((t[:,None]-b)/.36)**2).max(1)
                score=.1+.8*y if stem=='fit' else np.clip(.1+.8*y+dev_offset,0,1)
                if arm=='unweighted':score=score*.7
                prefix=f'r{i:02d}';v=np.ones(len(t),bool);v[0]=False
                for k,value in dict(time=t,valid=v,boundaries=b,target=y,local=score,radar=score,fusion=1-(1-score)**2).items():arrays[prefix+'_'+k]=value
                rows.append(dict(uid=f'{role}-u{i:02d}',role=role,subject_id=f's{i%2}',sensor_id='14xx',array_prefix=prefix))
            p=folder/f'{stem}-step500-arrays.npz';np.savez(p,**arrays)
            (folder/f'{stem}-step500.json').write_text(json.dumps(dict(step=500,records=rows,arrays_sha256=m.sha(p))))
    split_rows=[dict(uid=f'{role}-u{i:02d}',role=role,subject_id=f's{i%2}',sensor_id='14xx') for role,count in [('fit',18),('dev',8)] for i in range(count)]
    (root/'source-splits.json').write_text(json.dumps(dict(encoder_index_sha256='synthetic-index',records=split_rows)))
    return spec

def test_thresholds_lock_before_dev_and_dev_changes_do_not_reselect(tmp_path,monkeypatch):
    original=m.load;root=tmp_path/'a';spec=fixture(root)
    def guarded(folder,stem):
        if stem=='proposal':assert (Path(folder).parent/'loss-analysis-v1/selected.sha256').is_file()
        return original(folder,stem)
    monkeypatch.setattr(m,'load',guarded);m.analyze(root,spec)
    second=tmp_path/'b';other=fixture(second,.3);m.analyze(second,other)
    x=json.loads((root/'loss-analysis-v1/selected.json').read_text())['selected']
    z=json.loads((second/'loss-analysis-v1/selected.json').read_text())['selected']
    assert x==z
    for setting in x:
        assert setting['config']['lookahead_updates']==2 and setting['config']['nms_s']==.72
    assert m.sha(root/'loss-analysis-v1/selected.json')==(root/'loss-analysis-v1/selected.sha256').read_text().strip()

def test_mismatched_dev_count_is_rejected(tmp_path):
    root=tmp_path/'a';spec=fixture(root);p=root/'unweighted/proposal-step500.json';meta=json.loads(p.read_text());meta['records'].pop();p.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):m.analyze(root,spec)

def test_metadata_step_mismatch_is_rejected(tmp_path):
    root=tmp_path/'a';fixture(root);p=root/'weighted/fit-step500.json';meta=json.loads(p.read_text());meta['step']=100;p.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):m.load(root/'weighted','fit')

def test_empty_platform_is_missing_not_zero(tmp_path):
    root=tmp_path/'a';fixture(root);rows=m.load(root/'weighted','fit');result=m.platform(rows,'local',1.)
    assert result['p95_s'] is None

def test_original_dev_metadata_without_role_uses_verified_split_index(tmp_path):
    root=tmp_path/'a';spec=fixture(root)
    for arm in ('weighted','unweighted'):
        p=root/arm/'proposal-step500.json';meta=json.loads(p.read_text())
        for r in meta['records']:r.pop('role')
        p.write_text(json.dumps(meta))
    m.analyze(root,spec)
    assert json.loads((root/'loss-analysis-v1/readback.json').read_text())['passed']

def test_split_index_wrong_uid_is_rejected(tmp_path):
    root=tmp_path/'a';spec=fixture(root);p=root/'source-splits.json';value=json.loads(p.read_text());value['records'][-1]['uid']='unapproved';p.write_text(json.dumps(value))
    with pytest.raises(AssertionError):m.analyze(root,spec)
