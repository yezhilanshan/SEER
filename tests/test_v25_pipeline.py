import hashlib
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from continuous_mmwave.config import ModelConfig
from continuous_mmwave.models.event_former import TinyLLMAdapter, StreamingSemanticEventFormer
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_memory import EventMemory
from continuous_mmwave.v25.models import V25Heads, install_age_free_gate
from continuous_mmwave.v25.controller import V25Controller, ControllerConfig, FinalizationJob
from continuous_mmwave.v25.evidence import EvidenceConfig
from continuous_mmwave.v25.pipeline import V25Pipeline, FrozenQA
from continuous_mmwave.v25.stage3 import V25Stage3, fixed_budget_train, validate_feature_cache
from continuous_mmwave.v25.checkpoint import save_bundle, load_bundle, module_digest
from continuous_mmwave.v25.runtime import V25Runtime
from continuous_mmwave.v25.stage2 import Stage2Migration


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward_update(self,rd,md,**kwargs):
        self.calls += 1
        return SimpleNamespace(radar_tokens=rd,radar_token_mask=kwargs['frame_mask'],state=self.calls)


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        config = ModelConfig(radar_dim=8,llm_dim=16,llm_heads=4,llm_layers=2,cross_attn_dim=16,
                             cross_attn_heads=4,semantic_dim=8,dropout=0.,cross_attention_init_scale=.1)
        self.radar_encoder = Encoder()
        self.event_former = install_age_free_gate(StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4),config,torch.tensor([1,2])))
        self.semantic_projection = SemanticProjection(16,8)


def setup():
    torch.manual_seed(23)
    frontend = Frontend()
    heads = V25Heads(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    with torch.no_grad():
        for head in (heads.local.output[-1],heads.semantic.network[-1]):
            head.weight.zero_(); head.bias.fill_(20.)
        heads.radar.classifier[-1].weight.zero_(); heads.radar.classifier[-1].bias.fill_(-20.)
    c = V25Controller(frontend.event_former,frontend.semantic_projection,heads,
                      ControllerConfig(proposal_persistence=1,boundary_semantic_weight=0.,boundary_radar_weight=0.),
                      EvidenceConfig(warmup_s=.3,h_plus=.1,tau_stability=0.,evidence_weight=0.))
    compressor = QueryCompressor(16,internal_dim=16,heads=4,layers=1,dropout=0.)
    memory = EventMemory(compressor,frontend.semantic_projection,l0_capacity=2,l1_capacity=2,l2_capacity=2,group_size=2)
    return frontend,heads,c,memory


def job(uid,start,end):
    return FinalizationJob(uid,start,end,end+.2,.9,torch.randn(1,4,8),torch.ones(1,4,dtype=torch.bool),{})


def test_background_commit_keeps_memory_chronological_and_context_adds_active():
    frontend,heads,c,memory = setup()
    started, release = threading.Event(),threading.Event()
    with V25Pipeline(V25Runtime(frontend.radar_encoder,c),memory) as pipeline:
        original = pipeline._finalize
        def blocked(value):
            if value.event_id == 'a':
                started.set(); assert release.wait(5)
            return original(value)
        pipeline._finalize = blocked
        pipeline.submit(job('a',0.,1.))
        assert started.wait(2)
        pipeline.submit(job('b',1.,2.))
        c.step_tokens(torch.ones(1,4,8),torch.ones(1,4,dtype=torch.bool),2.)
        release.set()
        pipeline.flush()
        assert [u.uid for u in memory.levels[0]] == ['a','b']
        key = torch.randn(8)
        closed,selected = pipeline.context(key,now=2.,include_active=False)
        running,_ = pipeline.context(key,now=2.,include_active=True)
        assert running.shape[1] == closed.shape[1]+8
        assert [u.uid for u in selected] == ['a','b']


def test_commit_failure_blocks_later_jobs_and_is_reported():
    frontend,heads,c,memory = setup()
    started, release = threading.Event(),threading.Event()
    with V25Pipeline(V25Runtime(frontend.radar_encoder,c),memory) as pipeline:
        def fail(value):
            started.set(); assert release.wait(5)
            raise RuntimeError('commit exploded')
        pipeline._finalize = fail
        pipeline.submit(job('a',0.,1.))
        assert started.wait(2)
        pipeline.submit(job('b',1.,2.))
        release.set()
        with pytest.raises(RuntimeError,match='commit exploded'):
            pipeline.flush()
        assert not memory.levels[0]
        assert len(pipeline.failed_jobs) >= 1


def test_real_tiny_eventformer_encoder_once_memory_and_independent_qa():
    frontend,heads,c,memory = setup()
    original_p = module_digest(frontend.semantic_projection)
    qa_adapter = EventQAAdapter(16,8)
    reader = TinyLLMAdapter(32,16,2,4)
    def answerer(context,query):
        ids = torch.tensor([[3,4]])
        return reader.caption_logits(context,ids,torch.ones_like(ids,dtype=torch.bool)).argmax(-1).tolist()
    qa = FrozenQA(qa_adapter,answerer)
    with V25Pipeline(V25Runtime(frontend.radar_encoder,c),memory,qa=qa) as pipeline:
        for i in range(20):
            pipeline.step(torch.randn(1,4,8),torch.zeros(1),torch.tensor([0]),
                          torch.ones(1,4,dtype=torch.bool),i*.18)
        report = pipeline.finish(include_unconfirmed_tail=True)
        assert frontend.radar_encoder.calls == 20
        assert report['confirmed_events'] > 0
        assert report['eof_unconfirmed_events'] == 1
        assert not report['formal_go']
        assert memory.promotions[0] > 0
        saved = [u.tokens.clone() for level in memory.levels for u in level]
        response = pipeline.answer('what happened?',torch.randn(8),now=4.,include_active=False)
        assert response['answer']
        assert all(torch.equal(a,b.tokens) for a,b in zip(saved,[u for level in memory.levels for u in level]))
        assert module_digest(frontend.semantic_projection) == original_p


def test_stage3_fixed_budget_steps_are_actual_and_phase_isolation():
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    stage.set_phase('semantic')
    assert not any(p.requires_grad for p in stage.heads.evidence.parameters())
    rows = [dict(active=torch.randn(4,8),candidate=torch.randn(4,8),target=torch.tensor([0.,1.,0.,1.]),
                 age_bucket=torch.tensor([1,1,2,2]))]
    report = fixed_budget_train(stage,rows,phase='semantic',steps=3,lr=1e-3)
    assert report['actual_optimizer_steps'] == 3
    assert stage.optimizer_steps['semantic'] == 3
    with pytest.raises(ValueError,match='Verify'):
        stage.set_phase('verify')
    stage.set_phase('evidence')
    rows = [dict(tokens=torch.randn(2,8,16),target=torch.rand(2,4),mask=torch.ones(2,4,dtype=torch.bool))]
    assert fixed_budget_train(stage,rows,phase='evidence',steps=2)['actual_optimizer_steps'] == 2


def test_stage2_evidence_backpropagates_to_new_gate_without_age_feature():
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    state = frontend.event_former.initial_state(2)
    for i in range(3):
        state = frontend.event_former.stream_update(state,torch.randn(2,4,8),i/3.,torch.ones(2,4,dtype=torch.bool)).event_tokens
    loss = stage.stage2_evidence_loss(state,torch.rand(2,4),torch.ones(2,4,dtype=torch.bool))
    loss.backward()
    assert frontend.event_former.state_gate.network[-1].weight.grad.abs().sum() > 0


def test_checkpoint_requires_version_sha_and_explicit_development_opt_in(tmp_path):
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    path = tmp_path/'v25.pt'
    digest = save_bundle(path,frontend,stage,ControllerConfig(),EvidenceConfig(),
                         provenance={'frontend_config_sha256':'a'*64,'data_sha256':'b'*64})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='development'):
        load_bundle(path,frontend,stage,expected_sha256=digest)
    load_bundle(path,frontend,stage,expected_sha256=digest,allow_development=True)
    with pytest.raises(ValueError,match='SHA'):
        load_bundle(path,frontend,stage,expected_sha256='0'*64,allow_development=True)
    old = tmp_path/'old.pt'; torch.save({'phase':'verify','model_state':{}},old)
    with pytest.raises(ValueError,match='v2.5'):
        load_bundle(old,frontend,stage,expected_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),allow_development=True)


def test_old_feature_cache_and_wrong_source_are_rejected():
    source = 'a'*64
    with pytest.raises(ValueError,match='age-free'):
        validate_feature_cache({'schema_version':1},source)
    payload = {'schema_version':25,'architecture':'v25-age-free-tanh','frontend_digest':'b'*64,'split':'train'}
    with pytest.raises(ValueError,match='source'):
        validate_feature_cache(payload,source)


def test_physical_sequence_training_and_invalid_split():
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    row = dict(tokens=torch.randn(2,4,4,8),mask=torch.ones(2,4,4,dtype=torch.bool),
               local_target=torch.tensor([[0.,1.,0.],[1.,0.,1.]]),radar_target=torch.tensor([[0.,1.,0.],[1.,0.,1.]]))
    report = fixed_budget_train(stage,[row],phase='physical',steps=2)
    assert report['actual_optimizer_steps'] == 2
    assert all(p.grad is None for p in stage.heads.semantic.parameters())
    with pytest.raises(ValueError,match='train batches'):
        fixed_budget_train(stage,[{**row,'split':'dev'}],phase='physical',steps=1)


def test_stage2_migration_preserves_teacher_p_and_trains_gate_plus_evidence():
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    migration = Stage2Migration(frontend,stage.heads.evidence)
    assert all(not p.requires_grad for p in frontend.semantic_projection.parameters())
    assert all(p.requires_grad for p in frontend.event_former.state_gate.parameters())
    tokens = frontend.event_former.initial_state(2)
    for _ in range(3):
        tokens = frontend.event_former.stream_update(tokens,torch.randn(2,4,8),1.,torch.ones(2,4,dtype=torch.bool)).event_tokens
    fake = SimpleNamespace(stream_event_tokens=tokens,commit_event_tokens=tokens,
        stream_key=frontend.semantic_projection(tokens),commit_key=frontend.semantic_projection(tokens),
        losses=SimpleNamespace(text=tokens.square().mean(),align=tokens.abs().mean(),consistency=tokens.sum()*0.))
    result = migration.objective(fake,torch.rand(2,4),torch.ones(2,4,dtype=torch.bool),torch.randn(2,8),
        teacher_commit_key=torch.randn(2,8),commit_targets=torch.rand(2,4),commit_mask=torch.ones(2,4,dtype=torch.bool))
    result['loss'].backward()
    assert frontend.event_former.state_gate.network[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in frontend.semantic_projection.parameters())


def test_runtime_invalid_timestamp_does_not_advance_encoder():
    frontend,heads,c,memory = setup()
    runtime = V25Runtime(frontend.radar_encoder,c)
    args = (torch.randn(1,4,8),torch.zeros(1),torch.tensor([0]),torch.ones(1,4,dtype=torch.bool),0.)
    try:
        runtime.step(*args)
        with pytest.raises(ValueError,match='increasing'):
            runtime.step(*args)
        assert frontend.radar_encoder.calls == 1
    finally:
        c.close()


def test_stage2_token_objective_uses_radar_not_legacy_event_states():
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    migration = Stage2Migration(frontend,stage.heads.evidence)
    batch = dict(radar=torch.randn(2,3,4,8),radar_mask=torch.ones(2,3,4,dtype=torch.bool),
                 caption_ids=torch.tensor([[3,4,5],[6,7,8]]),caption_mask=torch.ones(2,3,dtype=torch.bool),
                 labels=torch.tensor([0,1]),stream_target=torch.rand(2,4),stream_target_mask=torch.ones(2,4,dtype=torch.bool),
                 commit_target=torch.rand(2,4),commit_target_mask=torch.ones(2,4,dtype=torch.bool),
                 teacher_stream_key=torch.randn(2,8),teacher_commit_key=torch.randn(2,8))
    result = migration.token_step(batch)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert frontend.event_former.state_gate.network[-1].weight.grad.abs().sum() > 0


def test_controller_jobs_are_retained_when_commit_queue_is_full():
    frontend,heads,c,memory = setup()
    started,release = threading.Event(),threading.Event()
    with V25Pipeline(V25Runtime(frontend.radar_encoder,c),memory,max_pending_jobs=1) as pipeline:
        original = pipeline._finalize
        def blocked(value):
            started.set(); assert release.wait(5)
            return original(value)
        pipeline._finalize = blocked
        c._ready_jobs.extend([job('a',0.,1.),job('b',1.,2.),job('c',2.,3.)])
        try:
            pipeline.poll()
            assert started.wait(2)
            assert [value.event_id for value in pipeline.pending_finalizations] == ['b','c']
            assert set(pipeline.retained_jobs) == {'a','b','c'}
        finally:
            release.set()
        pipeline.flush()
        assert [row['event_id'] for row in pipeline.completed_events] == ['a','b','c']
        assert not pipeline.retained_jobs


def test_failed_commit_retains_radar_payload_for_readback():
    frontend,heads,c,memory = setup()
    with V25Pipeline(V25Runtime(frontend.radar_encoder,c),memory) as pipeline:
        bad = job('a',0.,1.)
        def fail(value):
            raise RuntimeError('failed payload')
        pipeline._finalize = fail
        pipeline.submit(bad)
        with pytest.raises(RuntimeError,match='failed payload'):
            pipeline.flush()
        torch.testing.assert_close(pipeline.retained_jobs['a'].radar_tokens,bad.radar_tokens)


def test_calibrated_status_cannot_bypass_readiness_validation(tmp_path):
    from continuous_mmwave.v25.checkpoint import read_bundle
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    path = tmp_path/'original.pt'
    save_bundle(path,frontend,stage,ControllerConfig(),EvidenceConfig(),
                provenance={'frontend_config_sha256':'a'*64,'data_sha256':'b'*64})
    payload = torch.load(path,weights_only=True)
    payload['status'] = 'calibrated-development'
    forged = tmp_path/'false-status.pt'; torch.save(payload,forged)
    with pytest.raises(ValueError,match='readiness'):
        read_bundle(forged,expected_sha256=hashlib.sha256(forged.read_bytes()).hexdigest())


def test_post_training_probe_is_bound_to_current_frontend_and_head():
    from continuous_mmwave.v25.training import probe_semantic_head
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    rows = [dict(active=torch.randn(4,8),candidate=torch.randn(4,8),target=torch.tensor([0.,1.,0.,1.]),
                 age_bucket=torch.tensor([1,1,3,3]))]
    before = probe_semantic_head(stage.heads.semantic,rows,frontend_digest=module_digest(frontend),
        cache_sha256='a'*64,early_max_age=2,split='development')
    fixed_budget_train(stage,rows,phase='semantic',steps=2)
    after = probe_semantic_head(stage.heads.semantic,rows,frontend_digest=module_digest(frontend),
        cache_sha256='a'*64,early_max_age=2,split='development')
    assert before['semantic_head_digest'] != after['semantic_head_digest']
    assert after['semantic_head_digest'] == module_digest(stage.heads.semantic)
    assert after['passed'] is None  # no invented acceptance limits


def test_calibration_requires_passed_current_probe():
    from continuous_mmwave.v25.evidence import calibrate_episodes
    with pytest.raises(ValueError,match='probe'):
        calibrate_episodes([], [EvidenceConfig()])


def test_native_half_memory_context_passes_generation_adapter_without_mutation():
    context = torch.randn(1,16,16,dtype=torch.float16)
    original = context.clone()
    qa = FrozenQA(EventQAAdapter(16,8),lambda tokens,query:tokens.dtype)
    assert qa.answer(context,'what happened?') == torch.float32
    assert torch.equal(context,original)


def test_calibration_and_checkpoint_ready_path_and_stale_source(tmp_path):
    from continuous_mmwave.v25.training import probe_semantic_head,config_digest
    from continuous_mmwave.v25.evidence import Episode,calibrate_episodes
    frontend = Frontend()
    stage = V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8,evidence_dim=4)
    stage.optimizer_steps = dict(physical=1,semantic=1,evidence=1)  # synthetic certificate fixture only
    config,evidence = ControllerConfig(),EvidenceConfig()
    rows = [dict(active=torch.randn(80,8),candidate=torch.randn(80,8),target=torch.tensor([0.,1.]*40),
                 age_seconds=torch.tensor([.54]*40+[1.08]*40),episode_ids=[f'fixture-{i}' for i in range(80)])]
    criteria = dict(min_early_auroc=0.,min_mature_auroc=0.,max_same_early_fpr=1.,max_abs_same_age_correlation=1.)
    probe = probe_semantic_head(stage.heads.semantic,rows,frontend_digest=module_digest(frontend),
        cache_sha256='a'*64,early_range=(.54,.72),mature_range=(1.08,1.98),split='development',criteria=criteria)
    assert probe['passed'] is True
    source = dict(frontend_digest=module_digest(frontend),heads_digest=module_digest(stage),
        semantic_head_digest=module_digest(stage.heads.semantic),controller_config_digest=config_digest(vars_dict(config)),
        episodes_sha256='b'*64)
    positive,negative = Episode('c1','a1',0.),Episode('c2','a2',0.)
    for ep,label,p in ((positive,1,.99),(negative,0,.01)):
        ep.gt_transition_exists=label; ep.end_time=1.
        ep.observations=[dict(age_s=.5,difference=p,stability=1.,physical_difference=0.)]
    calibration = calibrate_episodes([positive,negative],[evidence],probe=probe,provenance=source)
    assert calibration['selected']['recall'] == 1.
    path = tmp_path/'ready.pt'
    sha = save_bundle(path,frontend,stage,config,evidence,provenance={'frontend_config_sha256':'a'*64,'data_sha256':'b'*64},
                      frontend_eventformer_steps=1,initialization_probe=probe,calibration=calibration)
    loaded = load_bundle(path,frontend,stage,expected_sha256=sha)
    assert loaded['status'] == 'calibrated-development'
    with torch.no_grad():
        stage.heads.semantic.network[-1].bias.add_(1.)
    stale = tmp_path/'stale.pt'
    sha = save_bundle(stale,frontend,stage,config,evidence,provenance={'frontend_config_sha256':'a'*64,'data_sha256':'b'*64},
                      frontend_eventformer_steps=1,initialization_probe=probe,calibration=calibration)
    with pytest.raises(ValueError,match='development'):
        load_bundle(stale,frontend,stage,expected_sha256=sha)


def vars_dict(config):
    from dataclasses import asdict
    return asdict(config)
