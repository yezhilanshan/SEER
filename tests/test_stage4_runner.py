import pytest
import torch
from continuous_mmwave.parrad_stage4_runner import chronological_groups, check_pilot_row, label_runs, training_summary, caption_tensors, finalized_tokens, boundary_diagnostic, validate_provenance, upstream_digest


def test_groups_keep_recordings_and_order_and_preserve_level2_sources():
    events = [{'uid': str(i), 'recording': 'a', 'start': float(i), 'label': i % 14} for i in range(18)]
    groups = chronological_groups(events, group_size=4)
    assert len(groups) == 5
    assert len(groups[-1]) == 2
    assert [e['uid'] for group in groups for e in group] == [str(i) for i in range(18)]
    events[-1]['recording'] = 'b'
    with pytest.raises(ValueError, match='recording'):
        chronological_groups(events, group_size=4)


def test_only_approved_train_arrays_selected_and_clean_label_runs():
    check_pilot_row({'split': 'train', 'subject_id': 'subject_02', 'session_id': 'set000', 'sensor_id': 'TI_xWR14xx'})
    for field, value in [('split','test'), ('session_id','set003'), ('subject_id','subject_01')]:
        row = dict(split='train', subject_id='subject_02', session_id='set000', sensor_id='TI_xWR14xx')
        row[field] = value
        with pytest.raises(ValueError, match='pilot'):
            check_pilot_row(row)
    assert label_runs(torch.tensor([[1,1],[1,1],[1,2],[2,2],[-100,-100],[3,3]])) == [(0,2,1),(3,4,2),(5,6,3)]
    assert 'then' in training_summary([{'label': 1}, {'label': 2}])


def test_caption_targets_include_eos_exclude_instruction():
    class Tokenizer:
        eos_token_id = 99
        def encode(self, text, add_special_tokens=False): return [10, 11] if text == 'prompt' else [12, 13]
    ids, attention, targets = caption_tensors(Tokenizer(), 'prompt', 'summary', 'cpu')
    assert ids.tolist() == [[10,11,12,13,99]]
    assert attention.all()
    assert targets.tolist() == [[False,False,True,True,True]]


def test_real_event_former_commit_contract_and_boundary_diagnostic():
    from continuous_mmwave.models.event_former import StreamingSemanticEventFormer, TinyLLMAdapter
    from continuous_mmwave.parrad_config import PARradConfig
    config = PARradConfig(llm_dim=16, llm_heads=4, llm_layers=2, radar_dim=16,
                          cross_attn_dim=16, cross_attn_heads=4, dropout=0)
    former = StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4), config, torch.tensor([1,2])).eval()
    tokens = finalized_tokens(former, torch.randn(1,4,16), torch.ones(1,4,dtype=torch.bool), None)
    assert tokens.shape == (8,16)
    assert boundary_diagnostic([1.,2.], [1.1,3.])['true_positives'] == 1


def test_cached_frontend_mismatch_is_rejected_before_reuse():
    provenance=dict(frontend_config='a', stage1_checkpoint='b',stage2_checkpoint='c',stage3_checkpoint='d',threshold_artifact='e')
    validate_provenance(provenance,provenance)
    with pytest.raises(ValueError,match='provenance'):
        validate_provenance(provenance,dict(provenance,stage2_checkpoint='changed'))
    with pytest.raises(ValueError,match='provenance'):
        validate_provenance({},provenance)


def test_upstream_digest_handles_native_scalar_parameters_and_detects_change():
    from types import SimpleNamespace
    scalar=torch.nn.Module(); scalar.register_parameter('alpha',torch.nn.Parameter(torch.tensor(.1)))
    controller=SimpleNamespace(**{name:scalar for name in ('event_former','semantic_projection','local_change','radar_transition','semantic_difference','verify')})
    runtime=SimpleNamespace(radar_encoder=scalar,controller=controller)
    before=upstream_digest(runtime)
    with torch.no_grad(): scalar.alpha.add_(.01)
    assert before != upstream_digest(runtime)
