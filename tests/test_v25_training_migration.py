"""Same-path teachers, independent covered ranges and continuous state."""
from types import SimpleNamespace
import torch
from torch.nn import functional as F

from test_v25_pipeline import Frontend
from continuous_mmwave.v25.models import EvidenceProjection
from continuous_mmwave.v25.stage2 import Stage2Migration


def test_retention_compares_each_path_to_its_own_teacher():
    frontend = Frontend()
    migration = Stage2Migration(frontend, EvidenceProjection(16, 4))
    tokens = torch.randn(2, 8, 16)
    stream, commit = torch.randn(2, 8), torch.randn(2, 8)
    output = SimpleNamespace(stream_event_tokens=tokens, commit_event_tokens=tokens,
        stream_key=stream, commit_key=commit,
        losses=SimpleNamespace(text=torch.tensor(0.), align=torch.tensor(0.), consistency=torch.tensor(0.)))
    target = torch.rand(2, 4); mask = torch.ones(2, 4, dtype=torch.bool)
    result = migration.objective(output, target, mask, stream.clone(), teacher_commit_key=commit.clone(),
                                 commit_targets=1-target, commit_mask=mask)
    assert abs(float(result['retention_loss'])) < 1e-6
    expected = (F.smooth_l1_loss(migration.evidence_projection(tokens), target)
                + F.smooth_l1_loss(migration.evidence_projection(tokens), 1-target))/2
    torch.testing.assert_close(result['evidence_loss'], expected)


def test_chunk_continuation_uses_supplied_state_and_returns_detached_state():
    frontend = Frontend()
    migration = Stage2Migration(frontend, EvidenceProjection(16, 4), tbptt_steps=2)
    radar = torch.randn(2, 4, 4, 8)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    former = frontend.event_former
    with torch.no_grad():
        initial = former.initial_state(2)
        for i in range(2):
            initial = former.stream_update(initial, radar[:, i], 1., mask[:, i]).event_tokens
        expected = initial
        for i in range(2, 4):
            expected = former.stream_update(expected, radar[:, i], 1., mask[:, i]).event_tokens
    batch = dict(radar=radar[:, 2:], radar_mask=mask[:, 2:], initial_state=initial,
        commit_radar=radar.flatten(1, 2), commit_radar_mask=mask.flatten(1, 2),
        caption_ids=torch.tensor([[3,4,5], [6,7,8]]), caption_mask=torch.ones(2,3,dtype=torch.bool),
        labels=torch.tensor([0,1]), stream_target=torch.rand(2,4), stream_target_mask=torch.ones(2,4,dtype=torch.bool),
        commit_target=torch.rand(2,4), commit_target_mask=torch.ones(2,4,dtype=torch.bool),
        teacher_stream_key=torch.randn(2,8), teacher_commit_key=torch.randn(2,8))
    result = migration.token_step(batch)
    torch.testing.assert_close(result['next_state'], expected)
    assert not result['next_state'].requires_grad
    result['loss'].backward()
    assert any(p.grad is not None for p in former.state_gate.parameters())
    assert all(p.grad is None for p in frontend.semantic_projection.parameters())
