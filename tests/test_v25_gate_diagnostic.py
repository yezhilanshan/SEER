import copy

import pytest
import torch

from continuous_mmwave.models.event_former import EventStateGate
from continuous_mmwave.v25 import gate_diagnostic as diagnostic


def legacy_gate():
    torch.manual_seed(7)
    gate = EventStateGate(4).float()
    with torch.no_grad():
        gate.network[-1].weight.normal_(0, .3)
        gate.network[0].weight[:, -1].add_(1.)
    return gate


@pytest.mark.parametrize('maturity', [0., 1.])
def test_fixed_condition_fold_and_exact_output_conversion(maturity):
    old = legacy_gate()
    previous, proposal = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    migrated = diagnostic.transferred_gate(old, maturity=maturity, exact=True)
    target, alpha = old(previous, proposal, maturity)
    actual, value = migrated(previous, proposal, 999.)
    torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, alpha, atol=1e-7, rtol=1e-5)
    assert migrated.network[0].in_features == 8
    assert migrated.architecture != 'v25-age-free-tanh'


def test_direct_weight_transfer_does_not_preserve_sigmoid_output_or_modify_teacher():
    old = EventStateGate(4)
    saved = copy.deepcopy(old.state_dict())
    transferred = diagnostic.transferred_gate(old, maturity=1., exact=False)
    previous, proposal = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)
    _, original_alpha = old(previous, proposal, 1.)
    _, raw_alpha = transferred(previous, proposal, 0.)
    assert (original_alpha > 0).all() and (raw_alpha < 0).all()
    assert transferred.architecture == 'v25-age-free-tanh-warmstart-v1'
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)


def test_exact_logit_coordinate_conversion_is_finite_at_extremes():
    logits = torch.tensor([-1000., -30., -2., 0., 30., 1000.])
    coordinate = diagnostic.sigmoid_to_tanh_coordinate(logits)
    assert torch.isfinite(coordinate).all()
    torch.testing.assert_close(coordinate.tanh(), logits.sigmoid(), atol=1e-7, rtol=1e-6)


def test_fixed_maturity_wrapper_ignores_caller_age_and_rejects_invalid_condition():
    old = legacy_gate()
    fixed = diagnostic.fixed_maturity_gate(old, maturity=0.)
    previous, proposal = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(fixed(previous, proposal, 999.)[0], old(previous, proposal, 0.)[0])
    with pytest.raises(ValueError):
        diagnostic.transferred_gate(old, maturity=float('nan'), exact=False)


def test_fit_usability_is_independent_of_dev_and_stops_bad_initialization():
    passing = dict(all_finite=True,negative_alpha_fraction=.1,state_norm_ratio_p99=2.)
    assert diagnostic.warmstart_usable(passing)['passed']
    for bad in (dict(passing,all_finite=False),dict(passing,negative_alpha_fraction=.51),
                dict(passing,state_norm_ratio_p99=4.01)):
        assert not diagnostic.warmstart_usable(bad)['passed']
    with pytest.raises(ValueError):
        diagnostic.warmstart_usable(dict(passing,negative_alpha_fraction=None))


def test_summary_reports_nonfinite_values_without_silently_dropping_them():
    report = diagnostic.tensor_summary(torch.tensor([1., float('nan'), 3.]))
    assert report['total'] == 3 and report['nonfinite'] == 1
    assert report['p99'] is not None


def test_native_fit_probe_separates_common_input_from_student_free_running(tmp_path):
    from types import SimpleNamespace
    from torch import nn
    from continuous_mmwave.v25.checkpoint import file_sha

    class Former(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_gate = EventStateGate(4)
        def initial_state(self, batch):
            return torch.ones(batch,2,4,dtype=torch.float16)
        def build_prompt_cache(self, batch):
            return None
        def stream_update(self, previous, radar, maturity, mask, prompt, *, trace=None):
            proposal = 1.01*previous.float() + .1*radar.mean(1,keepdim=True)
            state,alpha = self.state_gate(previous.to(proposal.dtype),proposal,maturity)
            state = state.to(torch.float16)
            if trace:
                trace(dict(E_prev=previous,E_hat=proposal,alpha=alpha,E=state))
            return SimpleNamespace(event_tokens=state)

    front=nn.Module(); front.event_former=Former()
    from continuous_mmwave.models.semantic import SemanticProjection
    front.semantic_projection=SemanticProjection(4,4)
    path=tmp_path/'record.pt'
    torch.save(dict(radar=torch.ones(65,4,4),radar_mask=torch.ones(65,4,dtype=torch.bool)),path)
    index=dict(records=[dict(uid='fit',role='fit',path=str(path),sha256=file_sha(path))],
               runs=[dict(uid='fit:0',record_uid='fit',role='fit',start=0,end=65)])
    old=front.event_former.state_gate
    raw=diagnostic.transferred_gate(old,maturity=1.,exact=False)
    exact=diagnostic.transferred_gate(old,maturity=1.,exact=True)
    summary=diagnostic.fit_probe(front,index,old,raw,exact,tmp_path)
    import json
    saved=json.loads((tmp_path/'fit-probe.json').read_text())
    assert summary['actual_optimizer_steps']==0
    assert len(saved['trajectory'])==65 and not saved['trajectory'][-1]['within_first64']
    assert summary['negative_alpha_fraction']==1.
    assert not summary['usability']['passed']
    first=saved['trajectory'][0]
    previous=front.event_former.initial_state(1)
    own_state=raw(previous.float(),1.01*previous.float()+.1)[0].to(torch.float16)
    assert abs(first['student_norm']-float(own_state.detach().float().norm()))<1e-6
    assert first['student_norm']!=first['teacher_norm']
    assert len(saved['paired_fixed_exact_trajectory'])==65
    assert saved['paired_fixed_exact_trajectory'][-1]['endpoint_key_max_error'] is not None


def test_diagnostic_config_sha_fields_are_real_sha256_strings():
    from pathlib import Path
    import yaml
    config=yaml.safe_load(Path('configs/parrad-v25-gate-diagnostic-v1.yaml').read_text())
    for name,value in config.items():
        if name.endswith('_sha256'):
            assert len(value)==64 and all(c in '0123456789abcdef' for c in value),name


def test_source_inference_settings_must_match_locked_encoder_index(tmp_path):
    from continuous_mmwave.v25.checkpoint import file_sha
    source=tmp_path/'frontend.yaml'
    source.write_text('max_commit_radar_tokens: 256\n')
    config=dict(frontend_config=str(source),source_checkpoint_sha256='a'*64)
    index=dict(config_sha256=file_sha(source),checkpoint_sha256='a'*64)
    assert diagnostic.validate_inference_source(config,index)==index['config_sha256']
    source.write_text('max_commit_radar_tokens: 128\n')
    with pytest.raises(ValueError,match='frontend inference'):
        diagnostic.validate_inference_source(config,index)


def test_nonfinite_first_update_is_an_unusable_candidate_not_missing_input():
    result=diagnostic.warmstart_usable(dict(all_finite=False,negative_alpha_fraction=1.,
                                          state_norm_ratio_p99=None))
    assert not result['passed']
    assert 'nonfinite-fit-output' in result['reasons']


def test_status_advances_atomically_without_overwriting_immutable_reports(tmp_path):
    import json
    path=tmp_path/'status.json'
    diagnostic.write_status(path,dict(phase='fit',actual_optimizer_steps=0))
    diagnostic.write_status(path,dict(phase='dev',actual_optimizer_steps=0))
    assert json.loads(path.read_text())['phase']=='dev'
    with pytest.raises(ValueError):
        diagnostic.write_status(path,dict(phase='failed',value=float('nan')))
    assert json.loads(path.read_text())['phase']=='dev'
