# PARrad Stage P3 Continuous Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the frozen-front-end PARrad Stage P3 `Proposal -> Candidate -> Semantic Verify -> Fork-and-Shadow` pipeline on continuous subject-disjoint recordings.

**Architecture:** Reconstruct the accepted Stage P2 module and freeze it, cache configuration-fingerprinted per-update Radar Tokens and oracle semantic-key trajectories, then train only four small Stage P3 heads in three explicit phases. Run the learned heads through a batch-one online controller whose Radar Encoder executes once per update and whose Reject path adopts a continuously maintained Shadow without replay.

**Tech Stack:** Python 3.10, PyTorch, NumPy NPZ/JSONL, PyYAML, pytest; local tests use the `interview` Conda environment and CUDA verification uses `/home/xibeidaxue/SEER/.conda/envs/seer`.

**Spec:** `docs/superpowers/specs/2026-09-24-parrad-stage3-transition-design.md`

## Global Constraints

- Preserve the accepted Stage P2 checkpoint at `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage2_event_joint_formal_v1/best.pt`, SHA-256 `939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a`.
- Freeze the causal MD-GRU Radar Encoder, Event Former, Semantic Projection, Qwen backbone, LoRA, and Stage P2 auxiliary classifier during every Stage P3 training phase.
- Use `frame_period_s=0.09`, `frames_per_update=2`, and the derived `update_period_s=0.18`; no 20-fps or four-frame assumption may enter Stage P3 code.
- Preserve complete recording order in validation/test; labels are metrics-only during online controller execution.
- Treat `-100` as ignored supervision, never as Idle or a semantic class.
- Keep train/validation/test subject and session groups disjoint and keep both sensor views of one session in the same split.
- Tune thresholds on validation only and execute the test split exactly once after thresholds are frozen.
- Write new cache, output, log, and audit artifacts only below `/home/xibeidaxue/newdisk2/WJ/PARrad`; do not alter raw or canonical inputs.
- Do not implement Hierarchical Memory, Idle/Presence, free-form captioning, new radar backbones, or Stage P5 fine-tuning.
- The workspace has no Git repository. Replace every commit boundary with a dated SDD ledger entry containing RED/GREEN commands, hashes, and the review ruling.

## Review Focus

1. A valid label transition separated by `-100` must not become a synthetic direct boundary; Task 1 pins ignored-gap behavior.
2. A Candidate shorter than three updates must produce zero Stability without NaN and still be eligible for timeout/Reject; Tasks 4 and 7 pin this path.
3. Proposals on the first or last valid update must not create an empty old/new event buffer; Tasks 1 and 7 pin recording-edge behavior.
4. A cache produced from a different Stage P2 checkpoint, config, or manifest must fail closed instead of being reused; Task 3 pins all three fingerprints.
5. A true boundary arriving while a false Candidate is active must remain deterministic and must not cause a second fork or radar replay; Task 7 pins the defined behavior and Task 8 measures its boundary consequence.

---

### Task 1: Continuous transition targets and audit

**Files:**
- Create: `src/continuous_mmwave/parrad_stage3_data.py`
- Create: `tests/test_parrad_stage3_data.py`
- Create: `.superpowers/sdd/2026-09-24-parrad-stage3-transition/progress.md`

**Interfaces:**
- Consumes: `PARradSample`, complete frame labels/timestamps, `frame_period_s`, and `frames_per_update`.
- Produces: `EventRun`, `TransitionAudit`, `extract_event_runs(...)`, `gaussian_boundary_targets(...)`, and `audit_stage3_manifests(...)`.

- [ ] **Step 1: Write failing event-run and ignored-gap tests**

```python
def test_extract_event_runs_does_not_bridge_ignored_gap():
    labels = torch.tensor([0, 0, -100, -100, 1, 1])
    times = torch.arange(6, dtype=torch.float64) * 0.09
    runs = extract_event_runs(labels, times)
    assert [(r.label, r.start_frame, r.stop_frame) for r in runs] == [(0, 0, 2), (1, 4, 6)]
    assert runs[1].boundary_from_previous is None

def test_extract_event_runs_records_direct_transition_time():
    labels = torch.tensor([0, 0, 1, 1])
    times = torch.tensor([0.0, 0.09, 0.18, 0.27], dtype=torch.float64)
    runs = extract_event_runs(labels, times)
    assert runs[1].boundary_from_previous == pytest.approx(0.18)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_data.py -v
```

Expected: collection fails because `continuous_mmwave.parrad_stage3_data` does not exist.

- [ ] **Step 3: Implement immutable run/audit contracts and extraction**

```python
@dataclass(frozen=True)
class EventRun:
    label: int
    start_frame: int
    stop_frame: int
    start_time: float
    stop_time: float
    boundary_from_previous: float | None

def extract_event_runs(labels: Tensor, timestamps: Tensor, ignore_index: int = -100) -> tuple[EventRun, ...]:
    """Return valid constant-label runs; ignored gaps break adjacency."""

@dataclass(frozen=True)
class TransitionAudit:
    recordings: int
    valid_runs: int
    direct_transitions: int
    ignored_gaps: int
    class_run_counts: dict[int, int]
    sensor_transition_counts: dict[str, int]
    duration_seconds: tuple[float, ...]
```

Reject non-1D tensors, length mismatch, non-finite/non-monotonic timestamps, and empty recordings with explicit `ValueError` messages.

- [ ] **Step 4: Add failing Gaussian-target and recording-edge tests**

```python
def test_gaussian_boundary_targets_use_seconds_and_mask_edges():
    update_times = torch.tensor([0.09, 0.27, 0.45], dtype=torch.float64)
    target, valid = gaussian_boundary_targets(
        update_times, boundary_times=torch.tensor([0.27]), sigma_seconds=0.36
    )
    assert target[1] == pytest.approx(1.0)
    assert target[0] == pytest.approx(math.exp(-(0.18**2) / (2 * 0.36**2)))
    assert valid.tolist() == [True, True, True]

def test_recording_edges_are_not_boundaries():
    runs = extract_event_runs(torch.tensor([2, 2]), torch.tensor([0.0, 0.09]))
    assert runs[0].boundary_from_previous is None
```

- [ ] **Step 5: Implement time-based targets and split audit**

```python
def gaussian_boundary_targets(
    update_times: Tensor,
    boundary_times: Tensor,
    sigma_seconds: float,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be positive")
    distance = (update_times[:, None] - boundary_times[None, :]).abs()
    nearest = distance.min(1).values if boundary_times.numel() else torch.full_like(update_times, torch.inf)
    target = torch.exp(-(nearest.square()) / (2 * sigma_seconds**2))
    valid = torch.ones_like(update_times, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    return torch.where(valid, target, torch.zeros_like(target)), valid

def audit_stage3_manifests(
    manifests: Mapping[str, str | Path], config: PARradConfig
) -> dict[str, TransitionAudit]:
    """Audit train/validation/test without expanding random windows."""
```

Call `assert_disjoint_splits` before loading arrays. Fail if a split has zero direct transitions or if any training class has zero valid run.

- [ ] **Step 6: Verify GREEN and existing data regressions**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_data.py tests/test_parrad_data.py tests/test_parrad_split.py -v
```

Expected: all pass.

- [ ] **Step 7: Record the no-Git task checkpoint**

Append the RED/GREEN commands, test counts, exported interfaces, and ruling to `.superpowers/sdd/2026-09-24-parrad-stage3-transition/progress.md` using `apply_patch`.

---

### Task 2: Physical and semantic transition primitives

**Files:**
- Create: `src/continuous_mmwave/models/transition.py`
- Create: `tests/test_parrad_stage3.py`
- Modify: `src/continuous_mmwave/models/__init__.py`

**Interfaces:**
- Consumes: accepted Radar Tokens `[B,4,256]`, masks `[B,4]`, semantic keys `[B,128]`, and `update_period_s`.
- Produces: `LocalChangeHead`, `PrototypeState`, `RadarTransitionHead`, `SemanticDifferenceHead`, `candidate_stability`, and `MonotonicVerifyHead`.

- [ ] **Step 1: Write failing Local Change tests**

```python
def test_local_change_head_shapes_masks_and_rejects_all_masked():
    head = LocalChangeHead(radar_dim=16, hidden_dim=8)
    previous, current = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    assert head(previous, current, mask).shape == (2,)
    with pytest.raises(ValueError, match="valid radar token"):
        head(previous[:1], current[:1], torch.zeros(1, 4, dtype=torch.bool))
```

- [ ] **Step 2: Verify Local Change RED**

Run `conda run -n interview pytest tests/test_parrad_stage3.py::test_local_change_head_shapes_masks_and_rejects_all_masked -v`.

Expected: import or symbol failure.

- [ ] **Step 3: Implement masked attention pooling Local Change**

```python
class LocalChangeHead(nn.Module):
    def __init__(self, radar_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1): ...
    def forward(self, previous: Tensor, current: Tensor, mask: Tensor) -> Tensor:
        previous, current = self.norm(previous), self.norm(current)
        feature = torch.cat((current - previous, (current - previous).abs(), current * previous), -1)
        hidden = self.input(feature)
        pooled = self.pool(hidden, mask).squeeze(1)
        return self.output(pooled).squeeze(-1)
```

Return logits, not probabilities; callers apply `sigmoid` once.

- [ ] **Step 4: Write failing prototype/EMA/Welford tests**

```python
def test_prototype_state_uses_physical_time_constant():
    state = PrototypeState.initialize(torch.tensor([[1.0, 0.0]]))
    updated = update_prototype(state, torch.tensor([[0.0, 1.0]]), 0.18, 2.0)
    assert updated.alpha == pytest.approx(math.exp(-0.18 / 2.0))
    assert updated.count == 2

def test_radar_transition_is_finite_for_zero_variance():
    head = RadarTransitionHead(radar_dim=16, prototype_dim=8)
    tokens = torch.ones(2, 4, 16)
    logits, state = head(tokens, torch.ones(2, 4, dtype=torch.bool), None, 0.18)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(state.variance).all()
```

- [ ] **Step 5: Implement physical prototype state and Radar Transition**

```python
@dataclass(frozen=True)
class PrototypeState:
    fast: Tensor
    mean: Tensor
    m2: Tensor
    count: int
    age_seconds: float
    alpha: float

def update_prototype(
    state: PrototypeState, value: Tensor, update_period_s: float, tau_fast_s: float
) -> PrototypeState: ...

class RadarTransitionHead(nn.Module):
    def forward(
        self,
        radar_tokens: Tensor,
        token_mask: Tensor,
        state: PrototypeState | None,
        update_period_s: float,
    ) -> tuple[Tensor, PrototypeState]: ...
```

Use population Welford variance `m2 / max(count, 1)`, `eps=1e-6`, and L2-normalized projected prototypes.

- [ ] **Step 6: Write failing semantic/stability/monotonic tests**

```python
def test_candidate_stability_requires_three_keys():
    assert candidate_stability(torch.randn(2, 8), tau=0.1).item() == 0

def test_verify_probability_is_monotonic_in_each_input():
    head = MonotonicVerifyHead()
    base = torch.tensor([[0.2, 0.3, 0.4]])
    for column in range(3):
        higher = base.clone(); higher[:, column] += 0.2
        assert head(higher).item() >= head(base).item()
    assert all(weight.item() >= 0 for weight in head.positive_weights())
```

- [ ] **Step 7: Implement semantic difference, Stability, and Verify**

```python
class SemanticDifferenceHead(nn.Module):
    def forward(self, active: Tensor, candidate: Tensor, maturity: Tensor) -> Tensor: ...

def candidate_stability(keys: Tensor, tau: float) -> Tensor:
    """Accept [B,H,D] or [H,D]; use the last three keys and return [B] or scalar."""

class MonotonicVerifyHead(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        weights = functional.softplus(self.raw_weights)
        return (inputs * weights).sum(-1) + self.bias
    def positive_weights(self) -> Tensor:
        return functional.softplus(self.raw_weights)
```

Verify returns logits; monotonicity is checked after `sigmoid` or directly on logits.

- [ ] **Step 8: Run Task 2 GREEN and regress Event Former interfaces**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3.py tests/test_parrad_encoder.py tests/test_parrad_stages.py -v
```

Expected: all pass with no existing Stage 1/2 failures.

- [ ] **Step 9: Update the SDD ledger**

Record parameter counts, RED/GREEN evidence, and the ruling that `motion_state` remains unused in P3 v1.

---

### Task 3: Fingerprinted frozen feature cache

**Files:**
- Modify: `src/continuous_mmwave/parrad_stage3_data.py`
- Modify: `tests/test_parrad_stage3_data.py`
- Test: `tests/test_parrad_stage3_training.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `_build_parrad_stage2`, Stage P2 trainable-only checkpoint, complete manifests.
- Produces: `CacheFingerprint`, `Stage3CacheRecord`, `build_stage3_cache(...)`, and `load_stage3_cache(...)`.

- [ ] **Step 1: Write failing cache round-trip and fingerprint tests**

```python
def test_stage3_cache_round_trip_and_fingerprint(tmp_path):
    record = Stage3CacheRecord(
        radar_tokens=np.ones((3, 4, 8), np.float16),
        radar_mask=np.ones((3, 4), np.bool_),
        frame_labels=np.array([[0,0], [0,1], [1,1]], np.int64),
        timestamps=np.array([[0.0,0.09], [0.18,0.27], [0.36,0.45]], np.float64),
        oracle_keys=np.ones((3, 6), np.float16),
        oracle_maturity=np.array([0.5, 1.0, 1.0], np.float32),
        oracle_run_id=np.array([0, 0, 1], np.int64),
    )
    fingerprint = CacheFingerprint("checkpoint", "config", "manifest")
    write_stage3_cache(tmp_path, "recording", record, fingerprint, {"subject_id": "s1"})
    assert load_stage3_cache(tmp_path, "recording", fingerprint).radar_tokens.shape == (3,4,8)
    with pytest.raises(ValueError, match="fingerprint"):
        load_stage3_cache(tmp_path, "recording", replace(fingerprint, checkpoint_sha256="other"))
```

- [ ] **Step 2: Verify cache RED**

Run `conda run -n interview pytest tests/test_parrad_stage3_data.py -k cache -v`.

- [ ] **Step 3: Implement NPZ + JSON cache format**

```python
@dataclass(frozen=True)
class CacheFingerprint:
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str

@dataclass(frozen=True)
class Stage3CacheRecord:
    radar_tokens: np.ndarray
    radar_mask: np.ndarray
    frame_labels: np.ndarray
    timestamps: np.ndarray
    oracle_keys: np.ndarray
    oracle_maturity: np.ndarray
    oracle_run_id: np.ndarray

def write_stage3_cache(root: Path, key: str, record: Stage3CacheRecord,
                       fingerprint: CacheFingerprint, metadata: Mapping[str, str]) -> None: ...
def load_stage3_cache(root: Path, key: str,
                      expected: CacheFingerprint) -> Stage3CacheRecord: ...
```

Use `np.savez_compressed`, `allow_pickle=False`, atomic temporary-file replacement, finite checks, and exact dtype/shape validation.

- [ ] **Step 4: Write a failing tiny-module continuous-state test**

```python
def test_cache_builder_carries_encoder_state_and_resets_oracle_event_state_only_at_runs(tmp_path):
    cache = build_stage3_cache_for_recording(fake_stage2, sample, fingerprint, tmp_path)
    assert fake_stage2.radar_encoder.update_indices == [0, 1, 2]
    assert cache.oracle_run_id.tolist() == [0, 0, 1]
    assert cache.oracle_maturity[0] < cache.oracle_maturity[1]
```

- [ ] **Step 5: Implement frozen cache extraction**

```python
@torch.inference_mode()
def build_stage3_cache_for_recording(
    module: PARradStage2Module,
    sample: PARradSample,
    fingerprint: CacheFingerprint,
    output_root: Path,
) -> Stage3CacheRecord: ...

def build_stage3_cache(
    config: ExperimentConfig,
    stage2_checkpoint: Path,
    output_root: Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> Path: ...
```

Reconstruct through `_build_parrad_stage2(config)`, overlay `load_training_checkpoint(stage2_checkpoint, module)`, call `eval()`, set every parameter `requires_grad=False`, carry Radar Encoder state across the recording, and reset oracle Event Former state only at valid training-label runs. Validation/test cache may contain oracle keys for post-hoc metrics, but online evaluation must not read them.

- [ ] **Step 6: Run cache GREEN and verify determinism**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_data.py tests/test_parrad_stage3_training.py -k "cache or fingerprint" -v
```

Expected: repeated cache builds with seed 42 produce identical arrays and metadata hashes.

- [ ] **Step 7: Update the SDD ledger**

Record cache schema version `1`, fingerprints, and the rule that raw/canonical data remain read-only.

---

### Task 4: Stage P3 supervised datasets and losses

**Files:**
- Modify: `src/continuous_mmwave/parrad_stage3_data.py`
- Create: `src/continuous_mmwave/stages/parrad_stage3.py`
- Modify: `tests/test_parrad_stage3_data.py`
- Modify: `tests/test_parrad_stage3.py`
- Modify: `src/continuous_mmwave/stages/__init__.py`

**Interfaces:**
- Consumes: `Stage3CacheRecord` and Task 2 heads.
- Produces: `PhysicalTransitionDataset`, `SemanticPairDataset`, `VerifyTrajectoryDataset`, `PARradStage3Module`, and phase-specific loss dictionaries.

- [ ] **Step 1: Write failing deterministic dataset tests**

```python
def test_semantic_pairs_are_balanced_deterministic_and_split_local():
    first = SemanticPairDataset(records, seed=42)
    second = SemanticPairDataset(records, seed=42)
    assert first.pair_index == second.pair_index
    assert Counter(target for _, _, target, _ in first.pair_index) == {0: 2, 1: 2}
    assert all(first.records[a].split == first.records[b].split for a,b,_,_ in first.pair_index)

def test_verify_trajectories_include_false_proposals_and_short_candidates():
    dataset = VerifyTrajectoryDataset(records, seed=42)
    kinds = {sample.kind for sample in dataset.samples}
    assert {"true_boundary", "shifted", "within_event", "short_spike"} <= kinds
```

- [ ] **Step 2: Verify dataset RED**

Run `conda run -n interview pytest tests/test_parrad_stage3_data.py -k "semantic_pairs or verify_trajectories" -v`.

- [ ] **Step 3: Implement the three datasets**

```python
@dataclass(frozen=True)
class PhysicalBatch:
    previous: Tensor
    current: Tensor
    mask: Tensor
    local_target: Tensor
    radar_target: Tensor
    reset: Tensor

@dataclass(frozen=True)
class SemanticPairBatch:
    active: Tensor
    candidate: Tensor
    maturity: Tensor
    target: Tensor

@dataclass(frozen=True)
class VerifyBatch:
    difference: Tensor
    key_history: Tensor
    maturity: Tensor
    target: Tensor
    weight: Tensor

class PhysicalTransitionDataset(Dataset):
    """Return previous/current Radar Tokens, mask, local/radar soft target, and recording reset flag."""

class SemanticPairDataset(Dataset):
    """Return active key, candidate key, maturity, and same/different target."""

class VerifyTrajectoryDataset(Dataset):
    """Return p_diff, three-key Stability inputs, maturity, target, and sample weight."""
```

Balance boundary-near/ordinary physical samples, four semantic pair types, and true/false Candidate types using train-only indices. Never duplicate validation/test samples to improve metrics.

- [ ] **Step 4: Write failing phase-freezing and loss tests**

```python
@pytest.mark.parametrize("phase,trainable_prefixes", [
    ("physical", ("local_change", "radar_transition")),
    ("semantic", ("semantic_difference",)),
    ("verify", ("verify",)),
])
def test_stage3_phase_exposes_only_expected_parameters(phase, trainable_prefixes): ...

def test_stage3_losses_are_logits_based_and_finite():
    module = tiny_stage3_module()
    metrics = module.physical_step(tiny_physical_batch())
    assert set(metrics) >= {"loss", "local_loss", "radar_loss"}
    assert torch.isfinite(metrics["loss"])
```

- [ ] **Step 5: Implement Stage P3 module and phase contract**

```python
class PARradStage3Module(nn.Module):
    def __init__(self, radar_dim=256, semantic_dim=128, prototype_dim=128,
                 update_period_s=0.18, tau_fast_s=2.0, tau_stability=0.1): ...
    def set_phase(self, phase: Literal["physical", "semantic", "verify"]) -> None: ...
    def physical_step(self, batch: PhysicalBatch) -> dict[str, Tensor]: ...
    def semantic_step(self, batch: SemanticPairBatch) -> dict[str, Tensor]: ...
    def verify_step(self, batch: VerifyBatch) -> dict[str, Tensor]: ...
```

Use `binary_cross_entropy_with_logits` for all learned binary heads. Apply sample weights inside the unreduced loss and normalize by their sum. Stability remains detached and parameter-free.

- [ ] **Step 6: Run Task 4 GREEN**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_data.py tests/test_parrad_stage3.py -v
```

Expected: all pass; phase transitions leave frozen Stage P2 outside this module untouched.

- [ ] **Step 7: Update the SDD ledger**

Record dataset counts for synthetic fixtures, phase parameter reports, and all loss definitions.

---

### Task 5: Stage P3 runner, checkpoints, and configuration

**Files:**
- Create: `src/continuous_mmwave/parrad_stage3_runner.py`
- Create: `tests/test_parrad_stage3_training.py`
- Create: `configs/parrad-md-gru-stage3-smoke.yaml`
- Create: `configs/parrad-md-gru-stage3.yaml`
- Modify: `src/continuous_mmwave/train.py`

**Interfaces:**
- Consumes: Task 3 cache index and Task 4 module/datasets.
- Produces: `run_parrad_stage3(...)`, `train_stage3_phase(...)`, phase checkpoints, JSONL metrics, and CLI `stage3` dispatch.

- [ ] **Step 1: Write failing three-phase runner tests**

```python
def test_stage3_runner_trains_phases_in_order_and_selects_validation_best(tmp_path):
    result = run_parrad_stage3(tiny_config(tmp_path), max_steps_per_phase=1)
    assert [row["phase"] for row in result.phase_summaries] == ["physical", "semantic", "verify"]
    assert (result.output_dir / "best_physical.pt").exists()
    assert (result.output_dir / "best_semantic.pt").exists()
    assert (result.output_dir / "best_verify.pt").exists()

def test_stage3_checkpoint_contains_heads_only_and_cache_fingerprint(tmp_path): ...
```

- [ ] **Step 2: Verify runner RED**

Run `conda run -n interview pytest tests/test_parrad_stage3_training.py -v`.

- [ ] **Step 3: Implement typed Stage P3 settings validation**

```python
@dataclass(frozen=True)
class Stage3TrainingConfig:
    cache_dir: Path
    stage2_checkpoint: Path
    output_dir: Path
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs_physical: int = 30
    epochs_semantic: int = 30
    epochs_verify: int = 20
    early_stopping_patience: int = 5

@dataclass(frozen=True)
class PhaseTrainingResult:
    phase: Literal["physical", "semantic", "verify"]
    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    best_metrics: Mapping[str, float]

@dataclass(frozen=True)
class Stage3TrainingResult:
    output_dir: Path
    phase_summaries: tuple[Mapping[str, Any], ...]
    final_checkpoint: Path
    cache_fingerprint: str
```

Validate positive epochs/batch size, non-negative weight decay, checkpoint existence, matching cache fingerprint, and a new output directory. Resume requires matching phase and fingerprint.

- [ ] **Step 4: Implement reusable phase loop and metric selection**

```python
def train_stage3_phase(
    module: PARradStage3Module,
    phase: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: Stage3TrainingConfig,
    resume: Path | None = None,
    max_steps: int | None = None,
) -> PhaseTrainingResult: ...

def run_parrad_stage3(config_path: Path, resume: Path | None = None,
                      max_steps_per_phase: int | None = None) -> Stage3TrainingResult: ...
```

Select physical best by validation PR-AUC then loss, semantic best by AUROC then balanced accuracy, and Verify best by AUROC then false-confirm rate. Write `metrics-<phase>.jsonl`, `validation-<phase>.jsonl`, `best_<phase>.pt`, and `last_<phase>.pt`.

- [ ] **Step 5: Add Stage P3 CLI dispatch without extending Stage 1/2 loops**

```python
if args.stage == "stage3":
    from continuous_mmwave.parrad_stage3_runner import run_parrad_stage3
    run_parrad_stage3(args.config, args.resume, args.max_steps)
    return
```

Keep Stage P3 out of the existing monolithic Stage 1/2 training loop.

- [ ] **Step 6: Add smoke and production configs**

Both configs must reference the accepted Stage P2 checkpoint and new directories:

```yaml
stage3:
  cache_dir: /home/xibeidaxue/newdisk2/WJ/PARrad/cache/stage3_transition_v1
  stage2_checkpoint: /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage2_event_joint_formal_v1/best.pt
  sigma_boundary_s: 0.36
  tau_fast_s: 2.0
  tau_stability: 0.1
  reject_timeout_s: 1.8
  maturity_seconds: 1.2
```

Smoke outputs to `outputs/stage3_transition_smoke_v1`; production outputs to `outputs/stage3_transition_v1`.

- [ ] **Step 7: Verify runner GREEN and Stage 1/2 CLI regression**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_training.py tests/test_parrad_training.py -v
```

Expected: all pass; existing Stage 1/2 config loading and resume behavior are unchanged.

- [ ] **Step 8: Update the SDD ledger**

Record config hashes, checkpoint schema, optimizer/early-stopping choices, and RED/GREEN results.

---

### Task 6: Fork-and-Shadow controller

**Files:**
- Create: `src/continuous_mmwave/parrad_controller.py`
- Create: `tests/test_parrad_controller.py`

**Interfaces:**
- Consumes: frozen `StreamingSemanticEventFormer`, `SemanticProjection`, trained Stage P3 heads, one update of Radar Tokens, mask, and timestamp.
- Produces: `ActiveEventState`, `FrozenActiveReference`, `CandidateBranch`, `ShadowBranch`, `FinalizationJob`, `ControllerOutput`, and `PARradOnlineController.step_tokens(...)`.

- [ ] **Step 1: Write failing state-transition tests**

```python
def test_proposal_creates_frozen_reference_candidate_and_shadow():
    controller = tiny_controller(force_proposal=True)
    before = controller.active.event_tokens.clone()
    output = controller.step_tokens(tokens(), mask(), timestamp=0.18)
    assert output.state == "candidate"
    assert torch.equal(controller.reference.event_tokens, before)
    assert torch.equal(controller.shadow.event_tokens_before_update, before)
    assert torch.equal(controller.candidate.event_tokens_before_update,
                       controller.event_former.initial_state(1))

def test_second_proposal_is_ignored_while_candidate_exists(): ...
```

- [ ] **Step 2: Verify controller RED**

Run `conda run -n interview pytest tests/test_parrad_controller.py -v`.

- [ ] **Step 3: Implement controller state contracts**

```python
@dataclass(frozen=True)
class ActiveEventState:
    event_tokens: Tensor
    prototype: PrototypeState
    start_time: float
    radar_chunks: tuple[Tensor, ...]

@dataclass(frozen=True)
class FinalizationJob:
    start_time: float
    end_time: float
    radar_tokens: Tensor
    radar_mask: Tensor
    provisional_event_tokens: Tensor
    metadata: Mapping[str, str]

@dataclass(frozen=True)
class ControllerOutput:
    state: Literal["active", "candidate"]
    proposal_probability: float
    difference_probability: float | None
    stability: float | None
    verify_probability: float | None
    decision: Literal["none", "proposal", "confirm", "reject"]
    finalization_job: FinalizationJob | None
```

Clone tensors at snapshot boundaries. Validate scalar timestamps are finite and strictly increasing.

- [ ] **Step 4: Implement proposal, branch updates, Confirm, and Reject**

```python
class PARradOnlineController:
    @torch.inference_mode()
    def step_tokens(self, radar_tokens: Tensor, radar_mask: Tensor,
                    timestamp: float, metadata: Mapping[str, str] | None = None) -> ControllerOutput: ...
```

Active mode performs one Event Former update. Candidate mode performs exactly two updates—Candidate and Shadow—from the same input tokens. Frozen Reference never runs. Physical scores are computed only for monitoring during Candidate and cannot enter Verify.

- [ ] **Step 5: Write failing no-replay and buffer-edge tests**

```python
def test_reject_matches_uninterrupted_shadow_without_replay():
    uninterrupted, forked = paired_controllers()
    for index, update in enumerate(sequence()):
        uninterrupted.step_tokens(*update)
        forked.step_tokens(*update)
    assert forked.replay_count == 0
    torch.testing.assert_close(forked.active.event_tokens, uninterrupted.active.event_tokens)
    assert forked.active.prototype == uninterrupted.active.prototype

def test_confirm_at_recording_edge_never_emits_empty_finalization_job(): ...

def test_short_candidate_has_zero_stability_and_can_reject(): ...

def test_true_boundary_during_false_candidate_does_not_refork_or_replay():
    controller = tiny_controller(force_proposal_at={1, 2}, force_reject_at=3)
    for update in sequence_with_true_boundary_at(2):
        controller.step_tokens(*update)
    assert controller.fork_count == 1
    assert controller.replay_count == 0
    assert controller.decision_trace == ["none", "proposal", "none", "reject"]
```

- [ ] **Step 6: Implement exact Reject adoption and Confirm ownership**

On Reject assign `active = shadow.as_active()` and clear Candidate/Reference/Shadow in O(1). On Confirm create the old-event `FinalizationJob` ending at the original proposal time, promote Candidate with `start_time=proposal_time`, and discard Shadow. A proposal at an edge with no sealable old event is suppressed.

- [ ] **Step 7: Add a runtime wrapper that calls the Radar Encoder once**

```python
class PARradStage3Runtime:
    def step(self, rd: Tensor, md: Tensor, sensor_index: Tensor,
             frame_mask: Tensor, timestamp: float) -> ControllerOutput:
        encoded = self.radar_encoder.forward_update(..., state=self.encoder_state)
        self.encoder_state = encoded.state
        return self.controller.step_tokens(encoded.radar_tokens,
                                           encoded.radar_token_mask, timestamp)
```

The counting-encoder test asserts exactly one `forward_update` invocation per external update in Active and Candidate states.

- [ ] **Step 8: Run controller GREEN**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_controller.py tests/test_phase1.py -v
```

Expected: all controller invariants and existing Event Former tests pass.

- [ ] **Step 9: Update the SDD ledger**

Record the state diagram, exact equality tolerances, pass counts, and cost if the no-replay invariant is wrong.

---

### Task 7: Continuous evaluation and validation-only calibration

**Files:**
- Create: `src/continuous_mmwave/parrad_stage3_diagnostics.py`
- Modify: `tests/test_parrad_controller.py`
- Modify: `tests/test_parrad_stage3_training.py`

**Interfaces:**
- Consumes: trained Stage P3 checkpoint, cached or raw complete recordings, frozen Stage P2 module, and threshold grid.
- Produces: `match_boundaries_seconds(...)`, `evaluate_controller(...)`, `calibrate_thresholds(...)`, `latency_benchmark(...)`, JSON audits, and the frozen threshold artifact.

- [ ] **Step 1: Write failing one-to-one boundary matching tests**

```python
def test_boundary_matching_is_one_to_one_in_seconds():
    result = match_boundaries_seconds(
        predicted=[1.1, 1.2, 3.4], truth=[1.0, 3.0], tolerance_s=0.5
    )
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.mean_absolute_error_s == pytest.approx(0.25)
```

- [ ] **Step 2: Verify diagnostics RED**

Run `conda run -n interview pytest tests/test_parrad_stage3_training.py -k boundary_matching -v`.

- [ ] **Step 3: Implement metrics and full-recording evaluator**

```python
@dataclass(frozen=True)
class BoundaryMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    absolute_errors_s: tuple[float, ...]

    @property
    def mean_absolute_error_s(self) -> float: ...

@dataclass(frozen=True)
class ControllerThresholds:
    proposal: float
    verify_confirm: float
    verify_reject: float
    candidate_timeout_updates: int
    fingerprint: str

def match_boundaries_seconds(predicted: Sequence[float], truth: Sequence[float],
                             tolerance_s: float) -> BoundaryMatch: ...

@torch.inference_mode()
def evaluate_controller(runtime: PARradStage3Runtime,
                        recordings: Sequence[PARradSample],
                        thresholds: ControllerThresholds) -> dict[str, Any]: ...
```

Use chronological greedy one-to-one matching because both lists are sorted. Labels are read only after inference to compute truth boundaries, event semantics, and groups.

- [ ] **Step 4: Write failing threshold-freeze and test-immutability tests**

```python
def test_calibration_uses_validation_only_and_freezes_thresholds(tmp_path):
    chosen = calibrate_thresholds(validation_results(), fixed_grid(), tmp_path / "thresholds.json")
    before = (tmp_path / "thresholds.json").read_bytes()
    evaluate_test_once(test_recordings(), chosen)
    assert (tmp_path / "thresholds.json").read_bytes() == before
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_test_once(test_recordings(), chosen)
```

- [ ] **Step 5: Implement fixed-grid calibration and single-use test marker**

```python
@dataclass(frozen=True)
class ThresholdGrid:
    proposal: tuple[float, ...]
    verify_confirm: tuple[float, ...]
    verify_reject: tuple[float, ...]
    candidate_timeout_updates: tuple[int, ...]

@dataclass(frozen=True)
class ValidationSweepResult:
    thresholds: ControllerThresholds
    boundary_f1_036: float
    boundary_f1_072: float
    over_segmentation: float
    median_confirmation_latency_s: float

def calibrate_thresholds(results: Iterable[ValidationSweepResult],
                         grid: ThresholdGrid, output_path: Path) -> ControllerThresholds: ...

def evaluate_test_once(..., marker_path: Path) -> dict[str, Any]: ...
```

Filter candidates to median confirmation latency `<=1.2`; sort by Boundary F1 at `0.72`, negative over-segmentation, then Boundary F1 at `0.36`. Atomically write thresholds plus validation metrics and a config/checkpoint fingerprint. Refuse a second test run unless the user explicitly supplies a new output directory.

- [ ] **Step 6: Implement baselines and grouped reports**

Run the same recordings through Proposal-only, Proposal+fixed-delay, and full controller. Report overall plus subject/sensor/recording/class/duration groups; include state norms, Candidate duty cycle, proposal-to-confirm ratio, and false-proposal rate.

- [ ] **Step 7: Implement CUDA latency benchmark**

```python
@torch.inference_mode()
def latency_benchmark(runtime, updates, warmup=50, repetitions=500) -> dict[str, float]:
    """Synchronize CUDA around timed normal, candidate, control-commit, and refiner paths."""
```

Report P50/P95, peak allocated/reserved memory, and Event Former passes per update. Evaluate the hard 180 ms deadline separately from the 70/140 ms paper targets.

- [ ] **Step 8: Run Task 7 GREEN**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_controller.py tests/test_parrad_stage3_training.py -v
```

- [ ] **Step 9: Update the SDD ledger**

Record matching semantics, threshold ordering, test-once enforcement, and latency measurement method.

---

### Task 8: Integration regression and remote staged rollout

**Files:**
- Modify: `src/continuous_mmwave/models/__init__.py`
- Modify: `src/continuous_mmwave/stages/__init__.py`
- Modify: `tests/test_parrad_training.py`
- Modify: `.superpowers/sdd/2026-09-24-parrad-stage3-transition/progress.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: accepted local suite, remote transition audit/cache, CUDA smoke, pilot checkpoint, validation thresholds, final test report, and a Stage P3 go/no-go audit.

- [ ] **Step 1: Add the failing end-to-end tiny CLI test**

```python
def test_stage3_cli_builds_cache_trains_heads_calibrates_and_evaluates(tmp_path):
    cli_main(["stage3", "--config", str(tiny_stage3_config(tmp_path)),
              "--max-steps", "1"])
    assert (tmp_path / "stage3" / "best_verify.pt").exists()
    assert (tmp_path / "stage3" / "validation_thresholds.json").exists()
```

- [ ] **Step 2: Verify end-to-end RED, then implement minimal orchestration**

The CLI sequence is `audit -> cache if absent and fingerprint-valid -> train requested phases -> optional calibration/evaluation`. Production test evaluation requires an explicit `--evaluate-test` flag and a fresh output directory.

- [ ] **Step 3: Run all local related tests**

Run:

```powershell
conda run -n interview pytest tests/test_parrad_stage3_data.py tests/test_parrad_stage3.py tests/test_parrad_controller.py tests/test_parrad_stage3_training.py tests/test_parrad_training.py tests/test_parrad_stages.py tests/test_phase1.py -v
```

Expected: all pass; Gemma tests remain out of scope per user instruction.

- [ ] **Step 4: Run the full local suite**

Run:

```powershell
conda run -n interview pytest tests -v
```

If `test_parrad_convert.py` is unavailable because local `h5py` is absent, run the full suite excluding only that file and record the exact exclusion. Do not install into unrelated environments.

- [ ] **Step 5: Sync only Stage P3 files to `/home/xibeidaxue/SEER`**

Before transfer, list exact local paths and verify every remote target resolves under `/home/xibeidaxue/SEER`. Never copy or delete other directories.

- [ ] **Step 6: Run remote related and full regressions**

Run in `/home/xibeidaxue/SEER/.conda/envs/seer` with `PYTHONPATH=src`. Record accepted pre-existing Gemma failures separately and do not attempt to fix Gemma.

- [ ] **Step 7: Run transition audit and cache smoke**

Write new artifacts to:

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/audit/stage3_transition_dataset_v1.json
/home/xibeidaxue/newdisk2/WJ/PARrad/cache/stage3_transition_v1
```

Verify split disjointness, counts, fingerprints, finite arrays, and exact file hashes before training.

- [ ] **Step 8: Run one-step CUDA smokes for P3-A/B/C**

Use `configs/parrad-md-gru-stage3-smoke.yaml`, a new output directory, AMP, and one real optimizer update per phase. Require finite losses/gradients/checkpoints and confirm zero Stage P2 tensor changes.

- [ ] **Step 9: Run a short pilot and evaluate hard gates**

Train enough epochs for every head to complete at least one validation pass. Stop rather than launching production if Semantic Difference does not beat its validation baseline, Verify collapses to one decision, physical proposal recall is zero, or CUDA latency exceeds 180 ms before controller overhead.

- [ ] **Step 10: Run production heads and validation calibration**

Use `configs/parrad-md-gru-stage3.yaml`; keep batch size conservative, output to `outputs/stage3_transition_v1`, and calibrate the fixed grid into `validation_thresholds.json`. Do not execute test until all validation gates are reviewed.

- [ ] **Step 11: Run the single final test and write go/no-go audit**

Write:

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/audit/stage3_transition_v1_gate.json
```

Include baseline/full metrics, threshold fingerprint, test-once marker, latency, memory, state-norm diagnostics, checkpoint hashes, and explicit pass/fail for every spec gate.

- [ ] **Step 12: Final verification and ledger closure**

Re-run focused tests after the server results, verify the accepted checkpoint and audit hashes, update the SDD ledger, and report whether Hierarchical Memory is permitted. Do not start Stage P4 without a new user instruction.

---

## Plan Self-Review

- Spec coverage: Tasks 1-8 cover supervision, cache, four heads, staged training, Fork-and-Shadow, finalization contract, calibration, baselines, latency, and go/no-go; explicitly excluded systems remain absent.
- Placeholder scan: no `TBD`, `TODO`, “similar to”, or unspecified implementation step remains.
- Type consistency: cached Radar Tokens flow into Task 2/4; `PARradStage3Module` checkpoints feed Task 6/7; `ControllerThresholds` is created by Task 7 and consumed only after validation freeze.
- Review Focus coverage: ignored gaps and edges are Task 1/6 tests; short Candidates are Task 2/6; fingerprints are Task 3; active-Candidate collision behavior is Task 6 and measured in Task 7.
- Scope: the plan ends at Stage P3 go/no-go and does not authorize Hierarchical Memory.
