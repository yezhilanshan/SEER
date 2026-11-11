import json

import pytest

from continuous_mmwave import parrad_radar_alignment as alignment
from continuous_mmwave import parrad_radar_alignment_train as train


def metrics():
    return dict(identify_macro_recall=12/14,
                presence_classes={k: dict(accuracy=1.) for k in ('Yes', 'No')})


def test_failed_teacher_remains_failed_when_explicitly_diagnostic():
    decision = alignment.text_preflight_decision(metrics(), 'diagnostic-only')
    assert decision['proceed'] and not decision['passed']
    assert decision['gates'] == dict(identify=False, Yes=True, No=True)
    assert decision['policy'] == 'diagnostic-only'
    assert not alignment.text_preflight_decision(metrics())['proceed']
    with pytest.raises(ValueError, match='policy'):
        alignment.text_preflight_decision(metrics(), 'typo')


@pytest.mark.parametrize('policy,continues', [('required', False), ('diagnostic-only', True)])
def test_training_control_flow_honors_policy_without_falsifying_gate(tmp_path, monkeypatch, policy, continues):
    import torch
    config = tmp_path/'config.yaml'; config.write_text('test: policy')
    settings = dict(seed=42, steps_per_arm=500, learning_rate=.0001, distill_weight=.5,
                    temperature=2., output_dir=str(tmp_path/'output'), qa_reader=str(tmp_path/'reader'),
                    qa_reader_sha256='test', text_preflight_policy=policy)
    events = [dict(uid=f'fit:{label}:{i}', label=label, cohort='fit') for label in range(14) for i in range(14)]
    events += [dict(uid=f'dev:{label}:{i}', label=label, cohort='dev') for label in range(14) for i in range(7)]
    events.append(dict(uid='dev:0:extra', label=0, cohort='dev'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(train, 'load_boundary_sources', lambda *args: (object(), None, None, {'events': events}, {}, {}, 'test'))
    monkeypatch.setattr(train, 'load_multi_reader', lambda *args: (object(), {}))
    # Only adapter access is needed before baseline. Real sampler and connector initialization run.
    from types import SimpleNamespace
    runtime = SimpleNamespace(controller=SimpleNamespace(event_former=SimpleNamespace(adapter=None)))
    monkeypatch.setattr(train, 'load_boundary_sources', lambda *args: (runtime, None, None, {'events': events}, {}, {}, 'test'))
    calls = []
    def evaluate(*args):
        calls.append(args[-1])
        if args[-1] == 'raw':
            raise RuntimeError('baseline reached after diagnostic failure')
        return dict(metrics=metrics(), rows=[])
    monkeypatch.setattr(train, 'evaluate', evaluate)
    if continues:
        with pytest.raises(RuntimeError, match='baseline reached'):
            train.run(settings, config)
    else:
        train.run(settings, config)
    root = tmp_path/'output'
    decision = json.loads((root/'text-gates.json').read_text())
    assert not decision['passed'] and decision['proceed'] == continues
    assert decision['policy'] == policy
    status = json.loads((root/'status.json').read_text())
    assert status['actual_updates'] == {'ce': 0, 'distill': 0}
    assert calls == (['text', 'text', 'raw'] if continues else ['text', 'text'])
    assert status['phase'] == ('baseline' if continues else 'text-preflight')
