import pytest


def checker():
    from continuous_mmwave import parrad_qa_readback as r
    assert hasattr(r,'assert_same_evaluation'), 'baseline reuse requires unchanged generation code'
    assert hasattr(r,'assert_same_baseline_source'), 'baseline reuse requires unchanged frozen inputs'
    return r


def test_reuse_allows_solver_change_but_rejects_changed_generation(tmp_path):
    r=checker(); old=tmp_path/'old.py'; new=tmp_path/'new.py'
    old.write_text('def evaluate(x):\n    return x+1\ndef train(x):\n    return x\n')
    new.write_text('def evaluate(x):\n    return x+1\ndef train(x):\n    return x*2\n')
    r.assert_same_evaluation(old,new,names=('evaluate',))
    new.write_text('def evaluate(x):\n    return x+2\n')
    with pytest.raises(ValueError,match='evaluation'): r.assert_same_evaluation(old,new,names=('evaluate',))


def test_reuse_rejects_changed_frozen_reader_or_task_source():
    r=checker()
    old=dict(provenance={'model':'a'},protected_sha256={'cache':'b'},frozen_upstream_digest='c',
        task_sha256='d',batch_sha256='e',base_qa_updates=1000,multi_reader={'frozen_digest':'f'})
    r.assert_same_baseline_source(old,dict(old,numerical_policy={'retry':True}))
    for key in ('multi_reader','task_sha256','protected_sha256'):
        changed=dict(old); changed[key]='changed'
        with pytest.raises(ValueError,match='baseline source'): r.assert_same_baseline_source(old,changed)
