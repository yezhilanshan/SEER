from continuous_mmwave.v25 import data


def test_questions_are_fixed_by_recording_gt_and_keep_balanced_answers():
    from continuous_mmwave.v25.evaluate import fixed_questions
    intervals=[(0.,1.,0),(1.,2.,1),(2.,3.,-100)]
    first=fixed_questions('record',intervals)
    assert first==fixed_questions('record',intervals)
    assert len({r['uid'] for r in first})==len(first)
    assert len(first)==6
    answers=[r['reference'] for r in first if r['kind']=='presence']
    assert answers.count('Yes')==answers.count('No')==2
    assert all(r['end']<=2 for r in first)


def test_boundary_gate_requires_each_subject_and_paired_full_record_baseline():
    from continuous_mmwave.v25.evaluate import compare_boundary
    def report(a,b):
        return dict(recordings=[dict(record_uid=s+'-set002-sensor',subject_id=s,metrics={'0.36':dict(f1=f,precision=.6)})
                               for s,f in [('subject_02',a),('subject_08',b)]])
    metric=compare_boundary(report(.6,.6),report(.3,.3))
    assert metric['passed'] is True
    regression=compare_boundary(report(.29,.9),report(.3,.3))
    assert regression['passed'] is True  # individual loss .01 <= .02
    regression=compare_boundary(report(.25,.9),report(.3,.3))
    assert regression['passed'] is False
