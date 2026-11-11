from pathlib import Path
import sys,json,importlib.util
sys.path.insert(0,str(Path(__file__).parents[1]/'.audit-staging'))
import proposal_window_analysis as m
import proposal_loss_analysis as old
spec=importlib.util.spec_from_file_location('old_analysis_fixture',Path(__file__).with_name('test_proposal_loss_analysis.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def fixture(root,offset):
    s=fixtures.fixture(root,offset);(root/'unweighted').rename(root/'window-peak');s['losses']=['weighted','window-peak']
    (root/'proposal-window-spec-v1.json').write_text(json.dumps(s));return s

def test_window_calibration_locks_before_dev_and_misses_partition_exactly(tmp_path,monkeypatch):
    real=old.load
    def checked(folder,stem):
        if stem=='proposal':assert (Path(folder).parent/'window-analysis-v1/selected.sha256').is_file()
        return real(folder,stem)
    monkeypatch.setattr(old,'load',checked)
    first=tmp_path/'a';s=fixture(first,0.);m.analyze(first,s)
    second=tmp_path/'b';t=fixture(second,.3);m.analyze(second,t)
    a=json.loads((first/'window-analysis-v1/selected.json').read_text())['selected'];b=json.loads((second/'window-analysis-v1/selected.json').read_text())['selected']
    assert a==b and {r['arm'] for r in a}=={'weighted','window-peak'}
    assert json.loads((first/'window-analysis-v1/readback.json').read_text())['failure_partition_equals_original_events']
    assert (first/'window-analysis-v1/all-boundary-failures.csv').is_file()

def test_all_18_settings_have_calibrated_widths_for_eight_cells_and_explicit_na(tmp_path):
    root=tmp_path/'all';s=fixture(root,0.)
    s['fixed_peak_slots']=[dict(id=f'{path}/peak/budget{cap}',path=path,calibration_fp_budget=cap,config=dict(family='peak',high=.6,lookahead_updates=2,nms_s=.72)) for path in ('local','radar','fusion') for cap in (3,5,8)]
    for arm in s['losses']:
        for stem in ('fit','proposal'):
            p=root/arm/f'{stem}-step500.json';meta=json.loads(p.read_text())
            for i,r in enumerate(meta['records']):r.update(subject_id=f's{i%4}',sensor_id=f'r{(i//4)%2}')
            p.write_text(json.dumps(meta))
    p=root/'source-splits.json';split=json.loads(p.read_text())
    for role in ('fit','dev'):
        for i,r in enumerate(x for x in split['records'] if x['role']==role):r.update(subject_id=f's{i%4}',sensor_id=f'r{(i//4)%2}')
    p.write_text(json.dumps(split));(root/'proposal-window-spec-v1.json').write_text(json.dumps(s))
    m.analyze(root,s);rows=json.loads((root/'window-analysis-v1/calibrated-platforms.json').read_text())
    assert len(rows)==18
    for setting in rows:
        assert setting['feasible'] and setting['threshold'] is not None
        for split in ('fit','dev'):
            cells={k:v for k,v in setting[split].items() if k.startswith('subject_sensor/')}
            assert len(cells)==8 and all('p95_s' in v for v in cells.values())
    rs=old.load(root/'weighted','proposal')
    na=m.calibrated_platforms(dict(arm='weighted',id='infeasible',path='local',budget=3,feasible=False,config=None),rs,rs)
    assert na['threshold'] is None and all(v is None for split in ('fit','dev') for v in na[split].values())
