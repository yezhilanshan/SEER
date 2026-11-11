import importlib.util
from pathlib import Path
import sys


def module():
    folder=Path(__file__).resolve().parents[1]/'.audit-staging';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('interface_extended',folder/'g0_interface_extended_analysis_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sample():
    rows=[]
    for uid,equal in [('a',True),('b',False)]:
        for step in (0,100,200):
            for control in ('correct','zero','swapped','shuffled'):
                value=(1+step/50) if control=='correct' else step/100
                rows.append(dict(uid=uid,case_uid=uid,panel='dev',kind='order',file=uid,equal_token_length=equal,step=step,control=control,
                    margin={'mean_with_eos':value},true_score={'mean_with_eos':-10+step/100}))
    return rows


def test_explicit_subsets_separate_unequal_from_all():
    m=module();s=m.extended_analysis(sample(),repeats=50)
    tables=[x for x in s['tables'] if x['step']==0 and x['control']=='correct']
    assert {x['subset']:x['n'] for x in tables}=={'all':2,'equal':1,'unequal':1}


def test_paired_intervals_include_middle_checkpoint_and_within_step_controls():
    m=module();s=m.extended_analysis(sample(),repeats=50)
    changes=[x for x in s['step_changes'] if x['subset']=='equal' and x['control']=='correct']
    assert {(x['from_step'],x['to_step']) for x in changes}=={(0,100),(100,200),(0,200)}
    first=next(x for x in changes if x['from_step']==0 and x['to_step']==100)
    assert first['metrics']['mean_with_eos']['margin']['record_macro_mean']==2
    assert first['metrics']['mean_with_eos']['true_score']['record_macro_mean']==1
    within=next(x for x in s['within_step_controls'] if x['subset']=='equal' and x['step']==100 and x['control']=='swapped')
    assert within['metrics']['mean_with_eos']['margin']['record_macro_mean']==2
    did=next(x for x in s['gain_changes'] if x['subset']=='equal' and x['from_step']==0 and x['to_step']==100 and x['control']=='swapped')
    assert did['metrics']['mean_with_eos']['margin']['record_macro_mean']==1
