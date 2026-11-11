import importlib

import pytest
import torch

from continuous_mmwave.models.event_former import TinyLLMAdapter
from continuous_mmwave.parrad_temporal_qa import temporal_batches, temporal_loss
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks
from tests.test_temporal_qa import make_model
from tests.test_short_context_qa import events
from tests.test_stage4_qa import Tokenizer


def routing():
    name = 'continuous_mmwave.parrad_qa_reader'
    assert importlib.util.find_spec(name), 'missing protected single/multi QA routing'
    return importlib.import_module(name)


def test_multi_loss_uses_new_reader_and_preserves_single_reader_and_prefix_gradient():
    r = routing()
    old = TinyLLMAdapter(128, 16, 2, 4)
    new = TinyLLMAdapter(128, 16, 2, 4)
    readers = r.QAReaders(old, new)
    assert readers.for_size(1) is old
    assert readers.for_size(2) is new and readers.for_size(4) is new
    with pytest.raises(ValueError): readers.for_size(3)
    rows = events(); by_id = {e['uid']: e for e in rows}
    batch = [dict(t, query='q') for t in temporal_batches(build_boundary_tasks(rows, 42), 42)[1]]
    model = make_model()
    loss = temporal_loss(model, readers, Tokenizer(), batch, by_id, 'cpu', 'time')
    expected = temporal_loss(model, new, Tokenizer(), batch, by_id, 'cpu', 'time')
    assert torch.equal(loss, expected)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for reader in (old, new) for p in reader.parameters())


def test_sequential_three_loss_backward_matches_mean_with_one_optimizer_update():
    import continuous_mmwave.parrad_temporal_qa as runner
    assert hasattr(runner, 'temporal_backward'), 'three retained LLM graphs need sequential accumulation'
    rows = events(); by_id = {e['uid']: e for e in rows}
    batch = [dict(t, query='q') for t in temporal_batches(build_boundary_tasks(rows, 42), 42)[1]]
    teacher = TinyLLMAdapter(128, 16, 2, 4)
    joint = make_model(); sequential = make_model()
    loss = temporal_loss(joint, teacher, Tokenizer(), batch, by_id, 'cpu', 'order')
    loss.backward()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    value = runner.temporal_backward(sequential, teacher, Tokenizer(), batch, by_id, torch.device('cpu'), 'order', scaler)
    assert value == pytest.approx(float(loss.detach()), rel=1e-6)
    for (name, p), (_, q) in zip(joint.named_parameters(), sequential.named_parameters()):
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, rtol=1e-5, atol=1e-7), name
    optimizer = torch.optim.AdamW([p for p in sequential.parameters() if p.requires_grad])
    optimizer.step()
    assert {int(s['step']) for s in optimizer.state.values()} == {1}


def test_loader_rejects_wrong_sha_before_model_load(tmp_path):
    r = routing(); path = tmp_path/'reader.pt'; path.write_bytes(b'invalid')
    with pytest.raises(ValueError, match='SHA'):
        r.load_multi_reader(dict(qa_reader=str(path), qa_reader_sha256='0'*64),
                            TinyLLMAdapter(128, 16, 2, 4), {}, 'original', torch.device('cpu'))


def test_retrieval_metrics_count_all_positives_and_report_absent_class_as_missing():
    r = routing()
    assert hasattr(r, 'retrieval_metrics'), 'missing original retrieval before/after measurement'
    scores = torch.tensor([[.9, .8, .7, .6], [.8, .9, .7, .6], [.6, .7, .8, .9]])
    result = r.retrieval_metrics(scores, [0, 1, 0, 1])
    # Class 0 ranks relevant items at 1,3; class 1 at 1,4.
    assert result['classes']['0']['AP'] == pytest.approx(5/6)
    assert result['classes']['1']['AP'] == pytest.approx(3/4)
    assert result['classes']['0']['recall_at_1'] == .5
    assert result['classes']['1']['recall_at_5'] == 1.
    assert result['classes']['2'] is None
    assert result['mAP'] == pytest.approx(19/24)


def test_sealed_native_reader_loads_exact_lora_and_keeps_frozen_prefix_gradients(tmp_path):
    r = routing()
    transformers = pytest.importorskip('transformers'); pytest.importorskip('peft')
    from continuous_mmwave.parrad_qa_text_decoder import create_reader, adapter_state, frozen_base_digest
    from continuous_mmwave.parrad_stage4_runner import sha
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    base = transformers.Qwen2ForCausalLM(cfg); base_dir = tmp_path/'base'; base.save_pretrained(base_dir)
    reader = create_reader(base, rank=2)
    with torch.no_grad():
        for name, p in reader.named_parameters():
            if 'lora_' in name: p.fill_(.03)
    source = dict(original_upstream_digest='original', protected_sha256={}, base_model=str(base_dir),
        base_files={str(p.relative_to(base_dir)):sha(p) for p in base_dir.rglob('*') if p.is_file()},
        new_frozen_base_digest=frozen_base_digest(reader))
    path = tmp_path/'reader.pt'
    torch.save(dict(step=500, adapter_state=adapter_state(reader),
        reader_settings=dict(rank=2, layers=2, dim=32), source=source), path)
    old = TinyLLMAdapter(128, 32, 2, 4)
    restored, provenance = r.load_multi_reader(dict(qa_reader=str(path), qa_reader_sha256=sha(path)),
        old, {}, 'original', torch.device('cpu'))
    assert all(torch.equal(v, adapter_state(restored)[k]) for k, v in adapter_state(reader).items())
    assert provenance['sha256'] == sha(path) and provenance['updates'] == 500
    assert not {p.data_ptr() for p in restored.parameters()} & {p.data_ptr() for p in old.parameters()}
    assert all(not p.requires_grad for p in restored.parameters())
    prefix = torch.randn(1, 16, 32, requires_grad=True); ids = torch.tensor([[3, 4, 5]])
    restored.caption_logits(prefix, ids, torch.ones_like(ids)).square().mean().backward()
    assert prefix.grad.abs().sum() > 0 and all(p.grad is None for p in restored.parameters())
    source['original_upstream_digest'] = 'wrong'
    saved = torch.load(path, weights_only=False); saved['source'] = source; torch.save(saved, path)
    with pytest.raises(ValueError, match='source'):
        r.load_multi_reader(dict(qa_reader=str(path), qa_reader_sha256=sha(path)), old, {}, 'original', torch.device('cpu'))
