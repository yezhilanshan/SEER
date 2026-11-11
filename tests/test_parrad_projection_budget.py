"""Explicit budget opt-in, unchanged optimization prefix and immutable resume evidence."""
import copy
import json
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('steps,milestones,limit', [
    (50, (20, 50), 20), (101, (20, 101), 100), (100, (20, 100), 99),
    (100, (0, 100), 100), (100, (-1, 100), 100),
    (100, (50, 20, 100), 100), (100, (20, 20, 100), 100),
    (100, (20, 101), 100), (100, (20, 50), 100),
    (True, (1,), 20), (50., (20, 50), 100), (100, (20., 100), 100),
])
def test_invalid_budget_refused_before_source_access_or_output_write(tmp_path, monkeypatch, steps, milestones, limit):
    from continuous_mmwave import parrad_projection_primary as module
    def forbidden(*args, **kwargs):
        pytest.fail('invalid budget reached source access')
    monkeypatch.setattr(module, 'load_sequence_source', forbidden)
    with pytest.raises(ValueError, match='budget|milestone|steps'):
        module.run(tmp_path/'source', tmp_path/'parent', tmp_path/'out',
                   steps=steps, milestones=milestones, budget_limit=limit)
    assert not (tmp_path/'out').exists()


def test_extended_real_optimizer_matches_original_twenty_step_prefix(tmp_path):
    from continuous_mmwave import parrad_projection_primary as module
    from test_parrad_projection_primary import fixture
    old, data, caches = fixture()
    original = copy.deepcopy(old.state_dict())
    kwargs = dict(held='subject_04', seed=1701, provenance={'synthetic': True})
    module.train_small_experiment(old, data, caches, tmp_path/'control', steps=20, milestones=(20,), **kwargs)
    result = module.train_small_experiment(old, data, caches, tmp_path/'extended', steps=21,
                                          milestones=(20, 21), budget_limit=100, **kwargs)
    assert [r['completed_projection_steps'] for r in result] == [0, 21, 21]
    for name in ('joint-relation1', 'joint-relation10'):
        control = torch.load(tmp_path/'control'/name/'last.pt', weights_only=False)
        prefix = torch.load(tmp_path/'extended'/name/'step-20.pt', weights_only=False)
        final = torch.load(tmp_path/'extended'/name/'last.pt', weights_only=False)
        for key, value in prefix['projection_state'].items():
            torch.testing.assert_close(value, control['model_state']['projection.'+key], atol=0, rtol=0)
        assert {int(s['step']) for s in final['optimizer_state']['state'].values()} == {21}
        assert final['phase'] == 'ready'
        assert final['signature']['identity']['settings']['steps'] == 21
        assert final['signature']['identity']['settings']['budget_limit'] == 100
        assert [r['step'] for r in final['history']] == [0, 20, 21]
        assert all(torch.equal(value, final['model_state']['teacher.'+key]) for key,value in original.items())
    assert all(torch.equal(value, old.state_dict()[key]) for key,value in original.items())


@pytest.mark.parametrize('steps,milestones,limit', [(100, (20, 50, 100), 100), (20, (10, 20), 20), (20, (5, 10, 20), 100)])
def test_run_resume_budget_mismatch_preserves_manifest_before_numeric_access(tmp_path, monkeypatch, steps, milestones, limit):
    from continuous_mmwave import parrad_projection_primary as module
    config = tmp_path/'config.yaml'; config.write_text('model: {llm_dim: 8, semantic_dim: 128}')
    checkpoint = tmp_path/'stage2.pt'; checkpoint.write_bytes(b'not numeric')
    spec = tmp_path/'spec.md'; spec.write_text('locked budget protocol')
    parent = tmp_path/'parent'; parent.mkdir()
    output = tmp_path/'out'; (output/'experiment').mkdir(parents=True)
    (output/'experiment'/'common-warmup.pt').write_bytes(b'exists')
    checked = {'manifest':{'config':str(config),'stage2_checkpoint':str(checkpoint)},'manifest_sha256':'source'}
    monkeypatch.setattr(module,'load_sequence_source',lambda *args: checked)
    provenance = dict(source_manifest_sha256='source',stage2_checkpoint_sha256=module.sha256(checkpoint))
    (parent/'manifest.json').write_text(json.dumps({'provenance':provenance}))
    provenance.update(parent_manifest_sha256=module.sha256(parent/'manifest.json'),config_sha256=module.sha256(config),
        code_sha256={p.name:module.sha256(p) for p in Path(module.__file__).parent.glob('parrad_projection_*.py')},
        spec_sha256=module.sha256(spec),source_split='train',diagnostic_only=True,cache_parent=str(parent.resolve()))
    evidence=json.dumps({'status':'diagnostics_complete','provenance':provenance,'settings':module.SETTINGS,
                         'conditions':[{'keep':'locked evidence'}]}).encode()
    (output/'manifest.json').write_bytes(evidence)
    def forbidden(*args):
        pytest.fail('budget mismatch reached numeric weights')
    monkeypatch.setattr(module,'_old_projection',forbidden)
    with pytest.raises(ValueError,match='settings'):
        module.run(tmp_path/'source',parent,output,resume=True,spec=spec,
                   steps=steps,milestones=milestones,budget_limit=limit)
    assert (output/'manifest.json').read_bytes() == evidence
