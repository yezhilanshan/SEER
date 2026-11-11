import hashlib
import torch
import pytest

from continuous_mmwave.radar_text.rd_transfer import normalize_rd
from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer, save_model, load_model


def inputs():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 1, 32, 32) * 12 + 50
    x[0, 0, 0, 0, 0] = -20
    x[1, 2:] = float('nan')
    return x, torch.tensor([4, 2])


def reference(x, lengths, log):
    result = torch.zeros_like(x, dtype=torch.float32)
    for i, n in enumerate(lengths.tolist()):
        a = x[i, :n].float()
        if log:
            a = a.sign() * a.abs().log1p()
        result[i, :n] = (a-a.mean()) / a.std(unbiased=False).clamp_min(1e-5)
    return result


@pytest.mark.parametrize('mode,log', [('signed_log_zscore', True), ('raw_zscore', False)])
def test_actual_normalization_matches_reference_and_preserves_input(mode, log):
    x, lengths = inputs(); before = x.clone()
    actual = normalize_rd(x, lengths, normalization=mode)
    assert torch.equal(actual, reference(x, lengths, log))
    assert actual.shape == x.shape and actual.dtype == torch.float32
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[1, 2:]) == 0
    changed = x.clone(); changed[1, 2:] = 1e12
    assert torch.equal(actual, normalize_rd(changed, lengths, normalization=mode))


def test_default_is_exact_legacy_behavior_and_invalid_modes_fail():
    x, lengths = inputs()
    assert torch.equal(normalize_rd(x, lengths), reference(x, lengths, True))
    with pytest.raises(ValueError, match='normalization'):
        normalize_rd(x, lengths, normalization='typo')


@pytest.mark.parametrize('mode', ['signed_log_zscore', 'raw_zscore'])
def test_training_and_inference_restore_actual_cnn_preprocessing(tmp_path, mode):
    torch.set_num_threads(2)
    model = RDResNetPretrainer(frame_chunk=8, amp=False, normalization=mode).eval()
    x, lengths = inputs(); captured = []
    hook = model.encoder.backbone.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    with torch.no_grad(): expected = model(x, lengths)
    hook.remove()
    valid = torch.arange(x.shape[1])[None] < lengths[:, None]
    assert torch.equal(torch.cat(captured), reference(x, lengths, mode == 'signed_log_zscore')[valid])
    path = tmp_path/'best.pt'; save_model(path, model, epoch=1)
    restored, payload = load_model(path)
    assert payload['model_config']['normalization'] == mode
    assert restored.encoder.normalization == mode
    with torch.no_grad(): assert torch.equal(restored(x, lengths), expected)


def test_legacy_checkpoint_without_normalization_uses_P0(tmp_path):
    model = RDResNetPretrainer(amp=False)
    p = tmp_path/'old.pt'; save_model(p, model)
    value = torch.load(p, weights_only=True); value['model_config'].pop('normalization', None); torch.save(value, p)
    restored, _ = load_model(p)
    assert restored.encoder.normalization == 'signed_log_zscore'


def test_parameter_fingerprint_and_sampler_order_ignore_preprocessing():
    from continuous_mmwave.radar_text.signed_log_ablation import parameter_sha, PairedWindowSampler
    models = []
    for mode in ['signed_log_zscore', 'raw_zscore']:
        torch.manual_seed(17); models.append(RDResNetPretrainer(amp=False, normalization=mode))
    assert parameter_sha(models[0]) == parameter_sha(models[1])
    with torch.no_grad(): models[1].classifier.bias[0] += 1
    assert parameter_sha(models[0]) != parameter_sha(models[1])
    class Data:
        windows = [(0,0,2,'a',[]),(0,2,4,'b',[]),(1,0,2,'c',[]),(1,2,4,'d',[])]
        def __len__(self): return len(self.windows)
    a, b = PairedWindowSampler(Data(),17), PairedWindowSampler(Data(),17)
    for _ in range(3):
        order=list(a);torch.rand(100)
        assert order == list(b) and sorted(order)==list(range(4))


def test_validation_diagnostics_use_full_recording_transition_context(tmp_path):
    import json
    import numpy as np
    from continuous_mmwave.radar_text.signed_log_ablation import validation_diagnostics
    # A transition just before the window must also mask its first valid frames.
    np.savez(tmp_path/'r.npz', labels=np.array([0,0,1,1,1,1]), timestamps=np.arange(6)*.09)
    class Recordings:
        manifest=tmp_path/'train.jsonl'
        rows=[dict(path='r.npz',sensor_id='radar',subject_id='person')]
    class Data:
        recordings=Recordings()
        windows=[(0,3,6,'window',[1])]
    scores=np.zeros((3,14));scores[:,1]=1
    np.savez(tmp_path/'logits.npz',logits=scores,labels=np.ones(3,dtype=int),offsets=np.array([0,3]))
    result=validation_diagnostics(Data(),tmp_path/'logits.npz')
    assert result['pooled']['frames']==3 and result['steady']['frames']==1
    assert result['by_sensor']['radar']['frames']==3


def test_completed_report_survives_plot_failure(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import subprocess
    import numpy as np
    from continuous_mmwave.radar_text.diagnostics import classification_metrics
    script=Path(__file__).resolve().parents[1]/'scripts/report_parrad_signed_log_ablation.py'
    spec=importlib.util.spec_from_file_location('ablation_report_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def dump(path,value):path.write_text(json.dumps(value))
    dump(tmp_path/'experiment-policy.json',dict(started='fixture-only',action_phrases={}))
    dump(tmp_path/'status.json',dict(state='complete'))
    metric=classification_metrics(np.eye(14,dtype=int));metric.update(subject_macro_f1=1.,unweighted_ce=.1)
    diagnostic=dict(steady=metric,bed_macro_f1=1.,by_subject={'s':metric},by_sensor={'r':metric})
    for arm in ['P0','P1']:
        p=tmp_path/arm;p.mkdir()
        dump(p/'results.json',dict(validation=metric));dump(p/'validation-diagnostics.json',diagnostic)
        dump(p/'selection.json',dict(epoch=1));dump(p/'status.json',dict(elapsed_seconds=1.))
        dump(p/'history.json',[dict(epoch=1,train_loss=.1,validation=metric,amp_overflows=0,optimizer_steps=1)])
    def failure(args,**kwargs):
        if kwargs.get('check'):raise subprocess.CalledProcessError(1,args)
        return subprocess.CompletedProcess(args,1,stdout='fixture plot failure')
    monkeypatch.setattr(module.subprocess,'run',failure)
    module.build_report(tmp_path)
    assert (tmp_path/'parrad_signed_log_ablation.md').exists()
    assert json.loads((tmp_path/'plot-status.json').read_text())['returncode']==1
