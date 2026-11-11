import importlib.util,sys
from pathlib import Path
import numpy as np

BASE=Path(__file__).parents[1]
def core():
    path=BASE/'.audit-staging/radar_gru_pool_core.py';assert path.exists(),'dual-source core absent'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('poolcore',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def event(t):return dict(time_s=t,peak_time_s=t-.18,lookahead_s=.18)

def test_cross_source_merge_is_first_arrival_causal_and_does_not_chain():
    c=core();radar=[event(1),event(1.54)];gru=[event(1.36)]
    result=c.merge(radar,gru)
    assert [x['time_s'] for x in result]==[1,1.54]
    assert result[0]['source']=='both' and result[0]['complete_source_available_s']==1.36
    assert result[1]['source']=='radar'
    short=c.merge(radar[:1],[]);assert short[0]['time_s']==result[0]['time_s']
    assert short[0]['source_at_creation']=='radar'==result[0]['source_at_creation']

def test_same_source_peaks_do_not_collapse_and_tie_is_deterministic():
    c=core();result=c.merge([event(1),event(1.18)],[event(1.18)])
    assert len(result)==2 and result[1]['source']=='both'
    assert result[1]['time_s']==1.18 and result[1]['source_at_creation']=='radar'

def test_lowest_density_selection_does_not_consume_dev_and_missing_explicit():
    c=core();grid=[dict(threshold=.1,recall=.95,density_per_minute=40),dict(threshold=.3,recall=.86,density_per_minute=20),dict(threshold=.4,recall=.84,density_per_minute=18)]
    assert c.select_target(grid,.85)['threshold']==.3
    assert c.select_target(grid,.99) is None

def test_balanced_sample_is_unique_repeatable_and_reports_shortages():
    c=core();rows=[dict(uid=str(cell),time_s=float(i),cell=str(cell)) for cell in range(8) for i in range(20)]
    a,info=c.balanced_sample(rows,50,42);b,_=c.balanced_sample(rows,50,42)
    assert a==b and len(a)==50 and len({(r['uid'],r['time_s']) for r in a})==50
    counts=[sum(r['cell']==str(i) for r in a) for i in range(8)];assert max(counts)-min(counts)<=1
    few,info=c.balanced_sample(rows[:3],50,42);assert len(few)==3 and info['shortage']==47

def test_merge_union_is_remeasured_not_oracle_hit_set_union():
    c=core();from proposal_causal_triggers import measure
    t=np.arange(0,4,.18);v=np.ones(len(t),bool);b=np.array([1.,1.9])
    result=c.merge([event(1.2)],[event(1.56)])
    assert len(result)==1
    m=measure(t,v,b,result);assert m['matched']==1 and m['recall']==.5

def test_supplemental_h5_rd_requires_exact_md_frame_alignment():
    c=core();md=np.arange(60,dtype=np.float32).reshape(20,3);rd=np.arange(20*2*4,dtype=np.float32).reshape(20,2,4)
    actual=c.supplemental_rd(rd,md,md.copy(),4,7)
    np.testing.assert_array_equal(actual,rd[4:7,None]);assert actual.shape==(3,1,2,4)
    import pytest
    with pytest.raises(ValueError,match='alignment'):c.supplemental_rd(rd,md[::-1],md,4,7)

def test_review_csv_preserves_different_case_fields_and_missing_onset(tmp_path):
    c=core();rows=[dict(case='radar',radar_emission=1.,offset=None),dict(case='gru',gru_peak_time=2.,near_any_GT=False,offset=None)]
    path=tmp_path/'review.csv';c.write_review_csv(path,rows)
    import csv
    with path.open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
    assert actual[0]['gru_peak_time']=='' and actual[1]['gru_peak_time']=='2.0'
    assert actual[1]['radar_emission']=='' and all(x['offset']=='' for x in actual)
