# PARrad Front-End Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PARrad-specific RD+MD streaming front end that emits the existing four-token Event Former contract and supports synthetic P1/P2 training validation without changing the original RD+Point Cloud path.

**Architecture:** A separate PARrad data contract feeds a single-channel asymmetric RD CNN and a lightweight MD Conv1d+causal-TCN branch. MD produces identity-initialized FiLM parameters and a motion state; enhanced RD Doppler tokens receive causal temporal context and are resampled by four learned queries into `[B,4,256]` Radar Tokens.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, NumPy, PyYAML, pytest; tests run with `C:\User\anaconda\envs\interview\python.exe` and `PYTHONPATH=D:\lab\SEER\src`.

**Spec:** `docs/superpowers/specs/2026-09-17-parrad-adaptation-design.md`

## Global Constraints

- Preserve all existing RD+Point Cloud classes, configs, tests, and behavior.
- PARrad online updates consume exactly 2 frames with `frame_period_s=0.09`, so `delta_t=0.18 s`.
- Each PARrad update emits exactly four 256-dimensional Radar Tokens.
- MD is a motion prior: it may modulate RD and produce `motion_state`, but it does not independently emit semantic Radar Tokens.
- FiLM projections are zero-initialized so the initial modulation is an exact identity.
- No interpolation to 20 fps or 10 fps in the first implementation.
- This plan covers P0/P1/P2 front-end foundations only. P3 Proposal/Verify/Fork-and-Shadow starts only after this plan passes.
- The workspace is not a Git repository; replace commit steps with verified test checkpoints and do not initialize Git implicitly.

---

## File Map

**Create**

- `src/continuous_mmwave/parrad_config.py`: frozen PARrad timing, shape, and dimension contract.
- `src/continuous_mmwave/data_parrad.py`: canonical PARrad sample/batch types, NPZ manifest loader, collate, alignment validation, and split leakage checks.
- `src/continuous_mmwave/parrad_audit.py`: read-only dataset summary CLI for canonical manifests.
- `src/continuous_mmwave/models/parrad_encoder.py`: RD encoder, MD encoder, FiLM, causal state, Doppler tokens, four-query resampler, and public output types.
- `src/continuous_mmwave/stages/parrad_stage1.py`: temporary framewise closed-set activity head and weighted loss.
- `src/continuous_mmwave/stages/parrad_stage2.py`: adapter from `PARradBatch` to the existing Event Former Stage 2 loss path.
- `configs/parrad-tiny-stage1.yaml`: synthetic CPU P1 configuration.
- `configs/parrad-tiny-stage2.yaml`: synthetic CPU P2 configuration.
- `scripts/make_synthetic_parrad_data.py`: deterministic canonical PARrad fixtures.
- `tests/test_parrad_data.py`: data contract, collate, leakage, and audit tests.
- `tests/test_parrad_encoder.py`: shape, FiLM, state, masking, and Event Former interface tests.
- `tests/test_parrad_stages.py`: P1/P2 forward/backward tests.

**Modify**

- `src/continuous_mmwave/models/__init__.py`: export the PARrad encoder types.
- `src/continuous_mmwave/stages/__init__.py`: export PARrad stage types.
- `README.md`: document the validation-only PARrad variant and commands.

**Do not modify in this plan**

- `src/continuous_mmwave/models/radar_encoder.py`
- `src/continuous_mmwave/data.py`
- `src/continuous_mmwave/models/event_former.py`
- `src/continuous_mmwave/stages/stage1.py`
- `src/continuous_mmwave/stages/stage2.py`
- `configs/stage1.yaml`
- `configs/stage2.yaml`

---

### Task 1: PARrad Configuration and Data Contract

**Files:**

- Create: `src/continuous_mmwave/parrad_config.py`
- Create: `src/continuous_mmwave/data_parrad.py`
- Test: `tests/test_parrad_data.py`

**Interfaces:**

- Produces: `PARradConfig`, `PARradSample`, `PARradBatch`, `PARradManifestDataset`, `parrad_collate`, `build_sensor_vocabulary`, `assert_disjoint_splits`.
- Canonical NPZ fields: `rd[T,1,H,W]`, `md[T,M]`, `labels[T]`, `timestamps[T]`.
- Manifest metadata: `path`, `subject_id`, `session_id`, `sensor_id`, `environment_id`.

- [ ] **Step 1: Write failing configuration and shape tests**

```python
def test_parrad_frozen_contract():
    config = PARradConfig()
    assert config.frames_per_update == 2
    assert config.delta_t == pytest.approx(0.18)
    assert config.radar_tokens_per_update == 4
    assert (config.rd_channels, config.rd_height, config.rd_width) == (1, 93, 128)
    assert (config.md_bins, config.radar_dim, config.motion_dim) == (128, 256, 128)

def test_parrad_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        PARradConfig(frames_per_update=0)
    with pytest.raises(ValueError):
        PARradConfig(frame_period_s=0)
    with pytest.raises(ValueError):
        PARradConfig(radar_tokens_per_update=3)
```

- [ ] **Step 2: Run tests and verify imports fail**

Run:

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests\test_parrad_data.py -q
```

Expected: collection failure because `continuous_mmwave.parrad_config` and `data_parrad` do not exist.

- [ ] **Step 3: Implement `PARradConfig`**

```python
@dataclass(slots=True)
class PARradConfig:
    frame_period_s: float = 0.09
    frames_per_update: int = 2
    rd_channels: int = 1
    rd_height: int = 93
    rd_width: int = 128
    md_bins: int = 128
    radar_dim: int = 256
    motion_dim: int = 128
    radar_tokens_per_update: int = 4
    sensor_vocab_size: int = 2
    use_sensor_embedding: bool = True
    fusion_heads: int = 8
    rd_tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    md_tcn_dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1

    @property
    def delta_t(self) -> float:
        return self.frame_period_s * self.frames_per_update
```

Validate all positive dimensions, token count exactly four, head divisibility, and dropout range.

- [ ] **Step 4: Write failing canonical NPZ and collate tests**

```python
def test_manifest_loads_aligned_canonical_npz(tmp_path):
    path = write_npz(tmp_path, frames=3)
    manifest = write_manifest(tmp_path, path)
    sample = PARradManifestDataset(manifest)[0]
    assert sample.rd.shape == (3, 1, 93, 128)
    assert sample.md.shape == (3, 128)
    assert sample.labels.shape == sample.timestamps.shape == (3,)

def test_parrad_collate_pads_time_and_builds_mask():
    batch = parrad_collate([sample(frames=2), sample(frames=4)])
    assert batch.rd.shape == (2, 4, 1, 93, 128)
    assert batch.frame_mask.tolist() == [[True, True, False, False], [True] * 4]

def test_alignment_mismatch_fails_closed(tmp_path):
    path = write_npz(tmp_path, frames=3, md_frames=2)
    with pytest.raises(ValueError, match="RD, MD, labels, and timestamps"):
        PARradManifestDataset(write_manifest(tmp_path, path))[0]
```

- [ ] **Step 5: Implement canonical data types and loader**

```python
@dataclass
class PARradSample:
    rd: Tensor
    md: Tensor
    labels: Tensor
    timestamps: Tensor
    subject_id: str
    session_id: str
    sensor_id: str
    environment_id: str

@dataclass(frozen=True)
class PARradBatch:
    rd: Tensor
    md: Tensor
    labels: Tensor
    timestamps: Tensor
    frame_mask: Tensor
    sensor_index: Tensor
    metadata: tuple[dict[str, str], ...]
```

Validation must reject incorrect rank, incorrect configured spatial dimensions, non-finite input, non-monotonic timestamps, misaligned lengths, and empty sequences. Padding labels use `-100`; padding timestamps use zero and are ignored by `frame_mask`.

`build_sensor_vocabulary` sorts the union of sensor IDs across the already-separated manifests and returns a stable mapping. Every dataset split receives that same mapping; unknown sensor IDs fail closed.

- [ ] **Step 6: Write and implement split leakage tests**

```python
def test_assert_disjoint_splits_rejects_subject_overlap():
    train = [{"subject_id": "s1", "session_id": "a"}]
    test = [{"subject_id": "s1", "session_id": "b"}]
    with pytest.raises(ValueError, match="subject overlap"):
        assert_disjoint_splits(train=train, validation=[], test=test)

def test_assert_disjoint_splits_rejects_session_overlap():
    train = [{"subject_id": "s1", "session_id": "a"}]
    test = [{"subject_id": "s2", "session_id": "a"}]
    with pytest.raises(ValueError, match="session overlap"):
        assert_disjoint_splits(train=train, validation=[], test=test)
```

- [ ] **Step 7: Run the focused data suite**

Expected: all `tests/test_parrad_data.py` tests pass.

- [ ] **Step 8: Checkpoint**

Record the passing command and changed files in the task summary. No commit is possible because the workspace is not a Git repository.

---

### Task 2: Dataset Audit CLI and Synthetic Fixtures

**Files:**

- Create: `src/continuous_mmwave/parrad_audit.py`
- Create: `scripts/make_synthetic_parrad_data.py`
- Modify: `tests/test_parrad_data.py`

**Interfaces:**

- Consumes: `PARradManifestDataset` from Task 1.
- Produces: `audit_manifest(path: str | Path) -> dict[str, object]` and CLI JSON output.

- [ ] **Step 1: Write the failing audit summary test**

```python
def test_audit_reports_shapes_timing_classes_and_metadata(tmp_path):
    manifest = make_two_recording_manifest(tmp_path)
    result = audit_manifest(manifest)
    assert result["recordings"] == 2
    assert result["frames"] == 6
    assert result["rd_shapes"] == [[1, 93, 128]]
    assert result["md_bins"] == [128]
    assert result["subjects"] == ["s1", "s2"]
    assert result["frame_interval_seconds"]["median"] == pytest.approx(0.09)
```

- [ ] **Step 2: Run the test and verify `audit_manifest` is missing**

- [ ] **Step 3: Implement deterministic audit aggregation**

The returned JSON-compatible dictionary must contain record count, frame count, unique RD shapes, MD bins, label histogram excluding `-100`, subject/session/sensor/environment sets, minimum/median/maximum positive frame interval, and finite-value status.

- [ ] **Step 4: Implement the synthetic generator**

Generate four deterministic recordings with variable lengths divisible or non-divisible by two, two subjects, two sensors, monotonic timestamps, per-frame labels, and a manifest. MD is computed as the RD mean over Range so tests can verify correlation.

- [ ] **Step 5: Run audit CLI against synthetic data**

Run:

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' scripts\make_synthetic_parrad_data.py
& 'C:\User\anaconda\envs\interview\python.exe' -m continuous_mmwave.parrad_audit data\synthetic_parrad\manifest.jsonl
```

Expected: valid JSON summary and exit code 0.

- [ ] **Step 6: Checkpoint**

Run `tests/test_parrad_data.py` and record the passing output.

---

### Task 3: MD-Guided PARrad Streaming Encoder

**Files:**

- Create: `src/continuous_mmwave/models/parrad_encoder.py`
- Modify: `src/continuous_mmwave/models/__init__.py`
- Create: `tests/test_parrad_encoder.py`

**Interfaces:**

- Consumes: `PARradConfig`, tensors `rd[B,2,1,93,128]`, `md[B,2,128]`, optional `sensor_index[B]`, optional `frame_mask[B,2]`, optional `PARradStreamState`.
- Produces: `PARradEncoderOutput(radar_tokens, semantic_state, motion_state, frame_hidden, state, radar_token_mask, frame_mask)`.
- Produces: `PARradStreamState(rd_tcn, md_tcn, update_index)`.

- [ ] **Step 1: Write failing output contract tests**

```python
def test_parrad_encoder_emits_four_token_contract():
    encoder = PARradRadarEncoder(tiny_config()).eval()
    output = encoder.forward_update(rd(), md())
    assert output.radar_tokens.shape == (2, 4, 16)
    assert output.semantic_state.shape == (2, 16)
    assert output.motion_state.shape == (2, 8)
    assert output.frame_hidden.shape == (2, 2, 16)
    assert output.radar_token_mask.shape == (2, 4)
```

- [ ] **Step 2: Run the test and verify missing implementation**

- [ ] **Step 3: Implement asymmetric residual RD blocks**

Expose `forward_maps(rd) -> Tensor[B,T,C,R',D']`. The first two strides are `(2,1)` and the third is `(2,2)`. Tests inspect intermediate shapes or block strides to assert that Doppler remains 128 through the first two reductions.

- [ ] **Step 4: Write and implement FiLM identity test**

```python
def test_film_is_exact_identity_at_initialization():
    film = MotionFiLM(motion_dim=8, channels=16)
    feature = torch.randn(2, 16, 4, 8)
    motion = torch.randn(2, 8)
    torch.testing.assert_close(film(feature, motion), feature)
```

Implement separate zero-initialized gamma and beta linear projections.

- [ ] **Step 5: Implement MD frame encoder and Stateful Causal TCN**

Reuse the existing `StatefulCausalTCN` class by composition; do not duplicate causal cache logic. Map each `[128]` MD frame to `motion_dim`, then apply dilations `(1,2,4)`.

- [ ] **Step 6: Implement Doppler tokens and context injection**

After FiLM, pool only the Range axis, yielding `[B,T,C,D']`. Project channel vectors at each Doppler location to `radar_dim`, add sinusoidal Doppler position encoding, attention-pool to `frame_hidden`, run the RD TCN, then add a zero-gated temporal context projection back to every Doppler token from the same frame.

- [ ] **Step 7: Implement the four-query resampler**

Flatten `[T,D',radar_dim]` within the current update and attend with exactly four learned queries. When `use_sensor_embedding=true`, add the configured sensor embedding to every RD Doppler token before resampling; passing `sensor_index=None` disables it for cross-sensor diagnostics. Do not create MD semantic tokens. Pool the four output tokens with a learned semantic query.

- [ ] **Step 8: Write stream equivalence and masking tests**

```python
def test_chunked_encoder_matches_forward_sequence():
    full = encoder.forward_sequence(rd_4_frames, md_4_frames)
    first = encoder.forward_update(rd_4_frames[:, :2], md_4_frames[:, :2])
    second = encoder.forward_update(rd_4_frames[:, 2:], md_4_frames[:, 2:], first.state)
    torch.testing.assert_close(torch.cat([first.radar_tokens, second.radar_tokens], 1), full.radar_tokens)

def test_all_padding_update_is_finite_and_zero():
    output = encoder.forward_update(rd(), md(), frame_mask=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isfinite(output.radar_tokens).all()
    assert not output.radar_token_mask.any()
    assert output.radar_tokens.count_nonzero() == 0
```

- [ ] **Step 9: Test Event Former compatibility**

Instantiate the tiny existing Event Former and call `stream_update(initial_state, output.radar_tokens, maturity=0.5, radar_mask=output.radar_token_mask)`. Assert output shape `[B,8,llm_dim]` and finite values.

- [ ] **Step 10: Run encoder and existing regression suites**

Run:

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests\test_parrad_encoder.py tests\test_phase1.py -q -k 'not qwen_front_four_layers_qv_lora_and_bfloat16'
```

Expected: all selected tests pass.

- [ ] **Step 11: Checkpoint**

Record model parameter count, focused test output, and changed files.

---

### Task 4: PARrad Stage P1 Closed-Set Pretraining

**Files:**

- Create: `src/continuous_mmwave/stages/parrad_stage1.py`
- Modify: `src/continuous_mmwave/stages/__init__.py`
- Create: `tests/test_parrad_stages.py`
- Create: `configs/parrad-tiny-stage1.yaml`

**Interfaces:**

- Consumes: `PARradBatch`, `PARradRadarEncoder`, `num_classes`, optional class weights.
- Produces: `PARradStage1Module.training_step(batch) -> PARradStage1Loss`.
- The temporary head maps `frame_hidden[B,T,radar_dim]` to logits `[B,T,num_classes]`.

- [ ] **Step 1: Write failing masked loss test**

```python
def test_parrad_stage1_ignores_padding_labels():
    logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    labels = torch.tensor([[0, -100]])
    loss = frame_classification_loss(logits, labels)
    assert loss < 0.001
```

- [ ] **Step 2: Implement loss and temporary head**

Use `cross_entropy(..., ignore_index=-100, weight=class_weights)`. Reject batches with no supervised frames instead of returning NaN.

- [ ] **Step 3: Write failing optimizer-step test**

Assert one optimizer step returns finite `loss` and `accuracy`, and changes at least one RD encoder parameter.

- [ ] **Step 4: Implement `PARradStage1Module`**

`forward` calls `encoder.forward_sequence`; `training_step` applies the head to `frame_hidden` and reports masked frame accuracy.

- [ ] **Step 5: Verify export excludes the temporary classifier**

Provide `export_encoder_state()` whose keys are only from `encoder.state_dict()`; assert no `classifier` key is present.

- [ ] **Step 6: Run focused P1 tests**

Expected: all P1 tests pass and existing Stage 1 tests remain green.

- [ ] **Step 7: Checkpoint**

Record the passing output. Do not add a production training CLI until the model-level training step is stable.

---

### Task 5: PARrad Stage P2 Event Token Alignment Adapter

**Files:**

- Create: `src/continuous_mmwave/stages/parrad_stage2.py`
- Modify: `tests/test_parrad_stages.py`
- Create: `configs/parrad-tiny-stage2.yaml`

**Interfaces:**

- Consumes: `PARradBatch`, `PARradRadarEncoder`, existing `StreamingSemanticEventFormer`, existing `SemanticProjection`, tokenized class templates.
- Produces: `PARradStage2Module.training_step(batch, caption_ids, caption_mask) -> Stage2Output`.
- Reuses: `Stage2Loss`, `symmetric_alignment_loss`, `cosine_consistency_loss`, and `causal_caption_cross_entropy` from the existing Stage 2 module.

- [ ] **Step 1: Write failing P2 contract test**

```python
def test_parrad_stage2_has_stream_commit_and_three_losses():
    output = module.training_step(batch, ids, mask)
    assert output.stream_event_tokens.shape == output.commit_event_tokens.shape == (2, 8, 32)
    assert set(output.losses.__dict__) == {"text", "align", "consistency"}
    assert torch.isfinite(output.losses.total(1.0, 1.0))
```

- [ ] **Step 2: Implement stream view using two-frame updates**

Run `PARradRadarEncoder.forward_sequence` once to obtain four tokens per update. Iterate update groups of four tokens through the existing Event Former and preserve per-update state norms.

- [ ] **Step 3: Implement commit view and semantic losses**

Pass all valid Radar Tokens to `commit_refine`; compute shared semantic projection for stream, commit, and text; reuse the existing three-loss definition unchanged.

- [ ] **Step 4: Add deterministic class-template mapping**

Define a versioned mapping function `templates_for_label(label_name: str) -> tuple[str, str, str]`. Template selection is `(epoch + sample_index) % 3`; unknown labels raise `KeyError`.

- [ ] **Step 5: Write optimizer-step and gradient tests**

Assert one step updates the PARrad encoder query parameters and semantic projection, while the tiny LLM backbone remains frozen.

- [ ] **Step 6: Run P2 and regression suites**

Expected: PARrad P2 tests pass; existing Stage 2 tests pass except the already-known environment-only Qwen/PEFT test when PEFT is unavailable.

- [ ] **Step 7: Checkpoint**

Record the exact skipped/failed dependency condition separately from code failures.

---

### Task 6: Documentation and Full Foundation Verification

**Files:**

- Modify: `README.md`
- Verify: all files from Tasks 1–5.

**Interfaces:**

- Produces: documented synthetic data, audit, P1, and P2 commands.

- [ ] **Step 1: Document validation-only scope**

Add a PARrad section stating that this path validates Event Tokens and future continuous event formation, does not replace RD+Point Cloud, and does not support pose or free-form captions.

- [ ] **Step 2: Document canonical NPZ and manifest formats**

Include exact field shapes and metadata keys. State that a raw PARrad conversion reader will be selected only after the downloaded files are audited.

- [ ] **Step 3: Run static compilation**

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' -m compileall -q src tests scripts
```

Expected: exit code 0.

- [ ] **Step 4: Run all PARrad tests**

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests\test_parrad_data.py tests\test_parrad_encoder.py tests\test_parrad_stages.py -q
```

Expected: all pass.

- [ ] **Step 5: Run existing regression tests**

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests\test_phase1.py -q
```

Expected: all code-level tests pass. If the sole Qwen test fails because `peft` is not installed, report it as an environment dependency and do not install packages without user authorization.

- [ ] **Step 6: Run the synthetic smoke path**

Generate canonical fixtures, run the audit, and execute one P1 and one P2 optimizer step through test helpers. Verify all recorded losses are finite.

- [ ] **Step 7: Review against the design spec**

Confirm four-token output, 180 ms update, identity FiLM, separate motion state, original-path compatibility, subject/session leakage checks, and absence of P3 claims.

- [ ] **Step 8: Checkpoint**

Summarize completed files, test evidence, the missing raw PARrad reader if data has not been provided, and the explicit go/no-go before starting a separate P3 implementation plan.
