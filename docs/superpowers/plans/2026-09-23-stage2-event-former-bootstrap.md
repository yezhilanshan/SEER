# Stage 2 Event Former Bootstrap Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PARrad Stage 2 consume the action-dependent Stage 1 Radar Tokens from its first successful optimizer step, learn non-collapsed Event Tokens, and only then introduce radar-language caption training.

**Architecture:** Keep the completed Radar Encoder frozen during the initial Stage 2 work. Replace the blocked/unstable Event Former bootstrap with a small configurable cross-attention residual and a sigmoid convex state update, add a disposable 14-class Event Token head plus duplicate-label-safe alignment, and train in a semantic warmup phase followed by a gently ramped joint phase. Every transition is guarded by representation, gradient, AMP, and counterfactual diagnostics.

**Tech Stack:** Python 3.10, PyTorch, Transformers/PEFT, pytest, YAML, PARrad label-run data.

**Spec:** `.superpowers/sdd/2026-09-17-parrad-front-end/progress.md`

## Global Constraints

- Continue using `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage1_causal_md_gru_token_warmup/radar_encoder.pt`.
- Do not modify or overwrite any previous Stage 1 or Stage 2 output directory.
- Keep Stage 2 training windows label-run based.
- Keep the Qwen model at `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct`; do not require Hugging Face access.
- Do not repair or change the accepted Gemma dependency issue.
- The workspace has no Git repository; record checkpoints and verification in the SDD progress ledger instead of making Git commits.
- Use the local `interview` Conda environment for pytest and the isolated server SEER environment for CUDA smoke tests.

## Review Focus

- An invalid cross-attention initial scale must fail model validation, and a config regression test must ensure the production PARrad scale stays strictly positive.
- An all-padding sample must remain finite and must not inject a fake radar update.
- Repeated labels in one contrastive batch must be treated as additional positives, not false negatives.
- A skipped GradScaler step must not advance the global optimizer-step counter or a max-step stop condition.
- Warmup checkpoints must restore phase, optimizer, scaler, and freeze state without accidentally unfreezing Radar Encoder or Qwen LoRA.

---

### Task 1: Stable radar injection and bounded event-state updates

**Files:**
- Modify: `src/continuous_mmwave/config.py`
- Modify: `src/continuous_mmwave/models/event_former.py`
- Modify: `configs/parrad-md-gru-stage2.yaml`
- Modify: `configs/parrad-md-gru-stage2-smoke.yaml`
- Test: `tests/test_phase1.py`
- Test: `tests/test_parrad_stages.py`

**Interfaces:**
- Consumes: `ModelConfig`, `BottleneckRadarCrossAttention.forward`, and `EventStateGate.forward`.
- Produces: `ModelConfig.cross_attention_init_scale: float`, `ModelConfig.event_state_initial_update: float`, and bounded gate values in `[0, 1]`.

- [x] **Step 1: Write failing initialization and gradient tests**

```python
def test_configured_cross_attention_scale_opens_radar_gradient():
    config = cfg(cross_attention_init_scale=1e-3)
    module = StreamingSemanticEventFormer(
        TinyLLMAdapter(64, 32, 4, 4), config, torch.tensor([1, 2, 3])
    )
    radar = torch.randn(2, 4, config.radar_dim, requires_grad=True)
    result = module.stream_update(module.initial_state(2), radar, 0.5)
    result.event_tokens.square().mean().backward()
    assert torch.isfinite(radar.grad).all()
    assert torch.count_nonzero(radar.grad) > 0
    assert torch.tanh(module.cross_attention[0].alpha).item() == pytest.approx(1e-3)


def test_state_gate_is_a_convex_update_with_safe_initial_fraction():
    gate = EventStateGate(8, initial_update=0.1)
    previous = torch.randn(2, 3, 8)
    proposal = torch.randn(2, 3, 8)
    state, values = gate(previous, proposal, 0.5)
    assert torch.all((0 <= values) & (values <= 1))
    torch.testing.assert_close(values, torch.full_like(values, 0.1))
    torch.testing.assert_close(state, previous.lerp(proposal, values))


def test_parrad_stage2_config_keeps_radar_path_open():
    config = load_experiment_config("configs/parrad-md-gru-stage2.yaml")
    assert config.model.cross_attention_init_scale > 0
```

- [x] **Step 2: Run the tests and confirm they fail because the fields and bounded gate do not exist**

Run: `python -m pytest tests/test_phase1.py tests/test_parrad_stages.py -q`

- [x] **Step 3: Add validated configuration fields**

Add to `ModelConfig`:

```python
cross_attention_init_scale: float = 0.0
event_state_initial_update: float = 0.1
```

Validate `0 <= cross_attention_init_scale < 1` and `0 < event_state_initial_update < 1`. Keep the generic default residual at zero for backward compatibility, while both PARrad Stage 2 YAML files explicitly set `cross_attention_init_scale: 0.001`.

- [x] **Step 4: Initialize the residual by its effective tanh scale**

Construct each cross-attention layer with the configured scale and initialize:

```python
self.alpha = nn.Parameter(torch.tensor(math.atanh(init_scale), dtype=torch.float32))
```

This makes the effective multiplier exactly `0.001`, rather than setting the raw parameter to an approximate value.

- [x] **Step 5: Replace the signed state update with a convex interpolation**

Initialize the final gate layer with zero weights and `logit(0.1)` bias, then compute:

```python
gates = torch.sigmoid(self.network(features))
state = torch.lerp(previous, proposal, gates)
```

This removes negative gates and values above one, which were able to extrapolate and grow the state norm on every streaming update.

- [x] **Step 6: Run the focused tests**

Run: `python -m pytest tests/test_phase1.py tests/test_parrad_stages.py -q`
Expected: all tests pass; radar input and cross-attention key/value parameters receive finite nonzero gradients on the first backward pass.

### Task 2: Truthful AMP step accounting and a safe initial scale

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Modify: `configs/parrad-md-gru-stage2.yaml`
- Modify: `configs/parrad-md-gru-stage2-smoke.yaml`
- Test: `tests/test_parrad_training.py`

**Interfaces:**
- Consumes: `_step(loss, optimizer, parameters, clip, scaler)`.
- Produces: `_step(...) -> bool`, where `True` means `optimizer.step()` actually ran; batch metrics contain `optimizer_steps` and `optimizer_step_skipped`.

- [x] **Step 1: Write a fake-GradScaler regression test**

```python
class FakeGradScaler:
    def __init__(self, scale_before: float, scale_after: float, skip: bool) -> None:
        self.current_scale = scale_before
        self.scale_after = scale_after
        self.skip = skip

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self):
        self.current_scale = self.scale_after

    def get_scale(self):
        return self.current_scale


def test_skipped_amp_step_does_not_advance_optimizer_step():
    scaler = FakeGradScaler(scale_before=65536.0, scale_after=32768.0, skip=True)
    metrics = train_stage2_batch(module, batch, ids, mask, optimizer, 1.0, scaler=scaler)
    assert metrics["optimizer_steps"] == 0
    assert metrics["optimizer_step_skipped"] == 1.0
```

Also add a successful-scale test where the scale stays equal and `optimizer_steps == 1`.

- [x] **Step 2: Run the focused test and verify the current unconditional count fails**

Run: `python -m pytest tests/test_parrad_training.py -q`

- [x] **Step 3: Return execution status from `_step`**

For AMP, capture `scale_before = scaler.get_scale()`, call `scaler.step()` and `scaler.update()`, then return `scaler.get_scale() >= scale_before`. A scale decrease means non-finite gradients caused GradScaler to skip the optimizer update. Return `True` on the non-AMP path.

- [x] **Step 4: Advance global steps only after a real optimizer update**

Have each batch trainer return `optimizer_steps` as `0` or `1`. In the main loop, pop that value before `step += optimizer_steps`; retain `optimizer_step_skipped` in `metrics.jsonl`.

- [x] **Step 5: Make AMP bootstrap configurable and conservative**

Construct the CUDA scaler as:

```python
torch.amp.GradScaler("cuda", init_scale=float(config.training.get("amp_init_scale", 65536)))
```

Set `amp_init_scale: 512` in both PARrad Stage 2 configurations, matching the earlier probe's first finite radar-gradient scale.

- [x] **Step 6: Run focused training-loop tests**

Run: `python -m pytest tests/test_parrad_training.py -q`
Expected: skipped steps are logged but do not increment `step` or satisfy `max_steps`.

### Task 3: Action-supervised Event Token warmup and false-negative-safe alignment

**Files:**
- Modify: `src/continuous_mmwave/stages/stage2.py`
- Modify: `src/continuous_mmwave/stages/parrad_stage2.py`
- Modify: `src/continuous_mmwave/train.py`
- Test: `tests/test_parrad_stages.py`
- Test: `tests/test_parrad_training.py`

**Interfaces:**
- Produces: `supervised_symmetric_alignment_loss(events, text, labels, temperature)`.
- Produces: `PARradStage2Module.auxiliary_classifier: nn.Linear` and stream/commit class logits.
- Produces: `set_parrad_stage2_phase(module, "semantic_warmup" | "joint")`.

- [x] **Step 1: Write a repeated-label contrastive test**

```python
def test_supervised_alignment_treats_duplicate_labels_as_positives():
    keys = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    labels = torch.tensor([2, 2, 7])
    loss = supervised_symmetric_alignment_loss(keys, keys, labels, 0.07)
    assert torch.isfinite(loss)
    assert loss < symmetric_alignment_loss(keys, keys, 0.07)
```

The loss uses a same-label positive mask and a log-sum-exp numerator in both event-to-text and text-to-event directions.

- [x] **Step 2: Write warmup freeze and auxiliary-head tests**

```python
def test_semantic_warmup_freezes_radar_and_lora_but_trains_event_former():
    set_parrad_stage2_phase(module, "semantic_warmup")
    assert not any(p.requires_grad for p in module.radar_encoder.parameters())
    assert not any(p.requires_grad for p in module.event_former.adapter.parameters())
    assert any(p.requires_grad for p in module.event_former.cross_attention.parameters())
    assert all(p.requires_grad for p in module.auxiliary_classifier.parameters())
```

Add a backward test proving the average stream/commit 14-class cross-entropy gives nonzero gradients to cross-attention, state gate, semantic projection, and auxiliary classifier, but no gradients to Radar Encoder or LoRA.

- [x] **Step 3: Implement the disposable 14-class head**

Classify the normalized `stream_key` and `commit_key` with a shared `nn.Linear(semantic_dim, 14)`. Derive one target per label-run sample from valid frame labels and reject a mixed-label sample instead of silently choosing one class.

- [x] **Step 4: Implement semantic-warmup loss**

For `semantic_warmup`, skip caption logits and text encoding. Optimize:

```text
0.5 * (CE(stream_class, label) + CE(commit_class, label))
+ 0.1 * consistency(stream_key, commit_key)
```

Keep the completed Radar Encoder and all Qwen/LoRA parameters frozen. Train only Event Former, Semantic Projection, and the auxiliary head.

- [x] **Step 5: Implement joint loss without discarding action supervision**

In the joint phase, enable Qwen LoRA but keep Radar Encoder frozen. Optimize:

```text
1.0 * supervised_alignment
+ 0.1 * consistency
+ caption_weight * caption_CE
+ 0.5 * auxiliary_classification
```

Ramp `caption_weight` linearly from `0.0` to `0.1` over the first two joint epochs. This prevents the initially large language-model loss from recreating the shared-prefix shortcut before Event Tokens become action-separable.

- [x] **Step 6: Run Stage 2 model and training tests**

Run: `python -m pytest tests/test_parrad_stages.py tests/test_parrad_training.py -q`
Expected: the original three production losses remain reported, auxiliary metrics are additional diagnostics, duplicate labels are not negatives, and phase-specific freeze states are exact.

### Task 4: Phase runner, diagnostics, and stop gates

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Modify: `src/continuous_mmwave/parrad_stage2_diagnostics.py`
- Modify: `configs/parrad-md-gru-stage2.yaml`
- Modify: `configs/parrad-md-gru-stage2-smoke.yaml`
- Test: `tests/test_parrad_stage2_diagnostics.py`
- Test: `tests/test_parrad_training.py`

**Interfaces:**
- Produces: Stage 2 configuration keys `semantic_warmup_epochs`, `auxiliary_weight`, `text_weight_final`, and `text_weight_ramp_epochs`.
- Produces: validation metrics `aux_accuracy`, `aux_macro_f1`, representation rank/cosine, real-vs-shuffled caption gap, and successful/skipped optimizer steps.

- [x] **Step 1: Write phase-transition and resume tests**

Test that epochs before `semantic_warmup_epochs` select warmup freeze/loss settings, the next epoch enables LoRA and the joint objective, and a resumed checkpoint restores the same phase before building the optimizer.

- [x] **Step 2: Add explicit production phase settings**

Use these initial values:

```yaml
training:
  amp_init_scale: 512
stage:
  semantic_warmup_epochs: 3
  auxiliary_weight: 0.5
  text_weight_final: 0.1
  text_weight_ramp_epochs: 2
  align_weight: 1.0
  consistency_weight: 0.1
```

The smoke configuration uses the same objective but limits batches/epochs rather than changing the model behavior.

- [x] **Step 3: Extend diagnostics with the new evidence**

Record gate min/mean/max, cross-attention effective scales, first-step radar/cross-attention gradient norms, Event Token class metrics, successful/skipped steps, and caption losses for real, zeroed, and shuffled Event Tokens.

- [x] **Step 4: Run the full local regression suite**

Run with the local interview environment:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest -q --ignore=tests/test_parrad_convert.py
```

Expected: no failures. The h5py-dependent conversion test remains excluded only because h5py is absent from the local interview environment.

- [x] **Step 5: Run a server CUDA bootstrap smoke before any epoch training**

Run 20 successful optimizer updates with the smoke config. Accept only if:

- all losses, parameters, and gradients are finite;
- zero optimizer steps are skipped at AMP scale 512;
- all cross-attention key/value/output groups receive nonzero gradients on the first update;
- Radar Encoder and Qwen LoRA remain bit-identical during semantic warmup;
- gates remain in `[0, 1]` and Event State norms do not show monotonic explosive growth.

- [x] **Step 6: Run the three-epoch semantic warmup and enforce a hard stop gate**

Proceed to joint training only if validation auxiliary Macro-F1 exceeds `0.20`, both stream and commit effective rank exceed `2.0`, and their mean off-diagonal cosine is below `0.98`. Otherwise stop and inspect the Event Former rather than adding caption loss.

- [x] **Step 7: Run a one-epoch joint smoke and enforce the conditioning gate**

Proceed to formal Stage 2 only if positive-negative alignment margin is positive, retrieval exceeds the `1/14` chance baseline, and real Event Tokens produce lower caption loss than shuffled Event Tokens. Treat a caption gap of less than `0.02` as inconclusive and stop for diagnosis.

- [x] **Step 8: Record verification and preserve artifacts**

Append exact commands, output paths, checksums, and gate results to `.superpowers/sdd/2026-09-17-parrad-front-end/progress.md`. Use new output directories for warmup, joint smoke, and formal training.
