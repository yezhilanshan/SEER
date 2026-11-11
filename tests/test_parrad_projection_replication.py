"""Replication dispatch preserves the locked recipe and identifies seed equivalence."""
import copy
import json
from pathlib import Path

import pytest
import torch


def test_primary_run_exposes_fold_seed_and_spec_without_changing_defaults():
    import inspect
    from continuous_mmwave.parrad_projection_primary import run
    signature = inspect.signature(run)
    assert signature.parameters['held'].default == 'subject_02'
    assert signature.parameters['seed'].default == 1701
    assert 'spec' in signature.parameters
    assert signature.parameters['cache_parent'].default is None


def test_different_training_seeds_are_not_independent_trials_of_deterministic_recipe(tmp_path):
    from continuous_mmwave.parrad_projection_primary import train_small_experiment
    from test_parrad_projection_primary import fixture
    old, data, caches = fixture()
    before = copy.deepcopy(old.state_dict())
    for seed in (1701, 3407):
        train_small_experiment(old, data, caches, tmp_path/str(seed), held='subject_04', seed=seed,
                               steps=1, milestones=(1,), provenance={'synthetic': True})
        warm = torch.load(tmp_path/str(seed)/'common-warmup.pt', weights_only=False)
        assert warm['identity']['settings']['held_subject'] == 'subject_04'
        assert warm['identity']['settings']['seed'] == seed
    for name in ('frozen-control', 'joint-relation1', 'joint-relation10'):
        first = torch.load(tmp_path/'1701'/name/'last.pt', weights_only=False)
        second = torch.load(tmp_path/'3407'/name/'last.pt', weights_only=False)
        for key, value in first['model_state'].items():
            torch.testing.assert_close(value, second['model_state'][key], atol=0, rtol=0)
        for key, state in first['optimizer_state']['state'].items():
            for field, value in state.items():
                torch.testing.assert_close(value, second['optimizer_state']['state'][key][field], atol=0, rtol=0)
        assert first['history'][-1]['score'] == second['history'][-1]['score']
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in before.items())


def test_fold_mismatch_refused_before_source_numeric_access(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_primary as module
    def forbidden(*args, **kwargs):
        pytest.fail('invalid held subject must be rejected before source access')
    monkeypatch.setattr(module, 'load_sequence_source', forbidden)
    with pytest.raises(ValueError, match='held'):
        module.run(tmp_path/'source', tmp_path/'parent', tmp_path/'out', held='subject_05')


@pytest.mark.parametrize('held,seed', [('subject_03', 1701), ('subject_02', 3407)])
def test_resume_parameter_mismatch_preserves_manifest_bytes_before_numeric_access(tmp_path, monkeypatch, held, seed):
    from continuous_mmwave import parrad_projection_primary as module
    config = tmp_path/'config.yaml'; config.write_text('model: {llm_dim: 8, semantic_dim: 128}')
    checkpoint = tmp_path/'stage2.pt'; checkpoint.write_bytes(b'not a numeric checkpoint')
    spec = tmp_path/'spec.md'; spec.write_text('locked settings')
    parent = tmp_path/'parent'; parent.mkdir()
    output = tmp_path/'output'; (output/'experiment').mkdir(parents=True)
    (output/'experiment'/'common-warmup.pt').write_bytes(b'exists')
    checked = {'manifest':{'config':str(config),'stage2_checkpoint':str(checkpoint)},'manifest_sha256':'source'}
    monkeypatch.setattr(module, 'load_sequence_source', lambda *args: checked)
    provenance = dict(source_manifest_sha256='source',stage2_checkpoint_sha256=module.sha256(checkpoint))
    (parent/'manifest.json').write_text(json.dumps({'provenance':provenance}))
    provenance.update(parent_manifest_sha256=module.sha256(parent/'manifest.json'),config_sha256=module.sha256(config),
                      code_sha256={p.name:module.sha256(p) for p in Path(module.__file__).parent.glob('parrad_projection_*.py')},
                      spec_sha256=module.sha256(spec),source_split='train',diagnostic_only=True,cache_parent=str(parent.resolve()))
    evidence=json.dumps({'status':'diagnostics_complete','provenance':provenance,'settings':module.SETTINGS,
                         'conditions':[{'keep':'original evidence'}]}).encode()
    manifest=output/'manifest.json'; manifest.write_bytes(evidence)
    def forbidden(*args):
        pytest.fail('numeric access must not occur on mismatched resume settings')
    monkeypatch.setattr(module,'_old_projection',forbidden)
    with pytest.raises(ValueError,match='settings'):
        module.run(tmp_path/'source',parent,output,resume=True,held=held,seed=seed,spec=spec)
    assert manifest.read_bytes()==evidence
