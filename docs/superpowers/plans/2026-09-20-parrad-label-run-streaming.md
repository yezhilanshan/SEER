# PARrad Label-Run and Streaming Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Stage 1 first on boundary-trimmed single-label runs and then on continuous state-carrying TBPTT sequences, while evaluating only on continuous recordings.

**Architecture:** `PARradWindowDataset` gains an explicit label-run mode while retaining fixed windows. The encoder and Stage 1 module accept an initial stream state. The training entry runs a label-run phase followed by an optional streaming phase; continuous validation/test use a state-carrying evaluator whose chunk boundaries are independent of labels.

**Tech Stack:** Python 3.10, PyTorch 2.4, pytest, YAML configuration.

**Spec:** `docs/superpowers/specs/2026-09-20-parrad-label-run-streaming.md`

## Global Constraints

- Do not modify PARrad manifests, canonical data, model checkpoints, or unrelated folders.
- Preserve the existing fixed-window dataset behavior as the default for backward compatibility.
- Real ignored transition frames remain in streaming state evolution but do not enter cross-entropy.
- Label-run boundaries are training-only and must not select validation/test windows.
- The repository has no `.git`; record progress in the adjacent progress file instead of committing.

## Review Focus

- A label run shorter than twice the boundary radius must be dropped without creating an empty sample.
- Existing `-100` gaps must never be returned as real frames by label-run mode.
- A long label run must split at 64 frames without mixing labels or losing its tail.
- TBPTT state tensors must be detached while `update_index` remains continuous.
- Continuous evaluation must produce the same frame count regardless of chunk size and must not use labels to choose chunks.

---

### Task 1: Label-run windows

**Files:**
- Modify: `src/continuous_mmwave/data_parrad.py`
- Test: `tests/test_parrad_data.py`

**Interfaces:**
- Produces: `PARradWindowDataset(..., window_mode="fixed" | "label_run")`.
- Produces: label-run samples containing one non-ignored class and at most `window_frames` frames.

- [ ] Add failing tests for transition trimming, long-run splitting, ignored gaps, and short-run dropping.
- [ ] Run the focused data tests and verify failures are caused by the missing `window_mode` behavior.
- [ ] Implement label-run window enumeration while preserving fixed mode.
- [ ] Run the focused data tests to green.

### Task 2: Stateful chunk continuation

**Files:**
- Modify: `src/continuous_mmwave/models/parrad_encoder.py`
- Modify: `src/continuous_mmwave/stages/parrad_stage1.py`
- Test: `tests/test_parrad_encoder.py`
- Test: `tests/test_parrad_stages.py`

**Interfaces:**
- Produces: `detach_parrad_stream_state(state: PARradStreamState) -> PARradStreamState`.
- Produces: `PARradRadarEncoder.forward_sequence(..., state: PARradStreamState | None = None)`.
- Produces: `PARradStage1Module.forward(batch, state=None)`.

- [ ] Add failing tests proving chunked state continuation preserves `update_index` and detach removes graph history.
- [ ] Run the focused encoder/stage tests and verify expected failures.
- [ ] Add initial-state forwarding and the detach helper.
- [ ] Run the focused tests to green.

### Task 3: TBPTT training and continuous evaluation

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Test: `tests/test_parrad_training.py`

**Interfaces:**
- Consumes: state-aware Stage 1 forwarding from Task 2.
- Produces: `train_parrad_streaming_batch(...)` with per-chunk optimizer steps.
- Produces: `evaluate_parrad_stage1_streaming(...)` over recording-ordered chunks.

- [ ] Add failing tests for state-carrying TBPTT, ignored-label chunks, and chunk-size-invariant continuous evaluation counts.
- [ ] Run focused training tests and verify expected failures.
- [ ] Implement batch slicing, state detach at chunk boundaries, and recording-stream evaluation.
- [ ] Run focused training tests to green.

### Task 4: Two-phase entry and production configuration

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Modify: `configs/parrad-stage1.yaml`
- Modify: `tests/test_parrad_training.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: label-run datasets, TBPTT trainer, and continuous evaluator.
- Produces: `epochs` label-run epochs followed by `streaming_finetune_epochs` continuous epochs.

- [ ] Add a failing integration test showing training uses label-run samples while validation/test use all continuous supervised frames.
- [ ] Run the integration test and verify the failure.
- [ ] Wire explicit train/evaluation modes and the two training phases into `run_training`.
- [ ] Set production batch size 16, evaluation recording batch size, stream sequence length, and fine-tune phase in YAML; document the keys.
- [ ] Run the focused integration tests to green.

### Task 5: Verification and deployment preparation

**Files:**
- Verify: all project tests
- Verify: production data inventory on the remote server

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a tested code/config set ready to copy to `/home/xibeidaxue/SEER` without touching other remote folders.

- [ ] Run the complete pytest suite in the `interview` Conda environment.
- [ ] Run a one-step synthetic Stage 1 smoke test.
- [ ] Audit the production label-run/continuous window statistics without starting full training.
- [ ] Review the complete change set against the specification and record any rulings or deferred minor issues.

### Task 6: Minimum label-run duration

**Files:**
- Modify: `src/continuous_mmwave/data_parrad.py`
- Modify: `src/continuous_mmwave/train.py`
- Test: `tests/test_parrad_data.py`

**Interfaces:**
- Produces: `PARradWindowDataset(..., min_label_run_frames=N)`.
- Produces: balanced non-overlapping chunks no longer than `window_frames`, with no tiny tail when the retained run is long enough.

- [x] Add failing tests for dropping short runs and redistributing a one-frame tail.
- [x] Implement validation and balanced run partitioning.
- [x] Run data and training regression tests.

### Task 7: Recording-continuous streaming training

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Test: `tests/test_parrad_training.py`

**Interfaces:**
- Produces: streaming training that accepts and returns detached `PARradStreamState`.
- Produces: ordered recording groups whose state resets only between groups/recordings, never at a 256-frame sequence boundary.

- [x] Add a failing state-continuation test spanning two 256-frame sequences.
- [x] Replace shuffled streaming windows with recording-ordered sequence batches.
- [x] Run focused and integration tests.

### Task 8: Phase schedules, early stopping, and metrics

**Files:**
- Modify: `src/continuous_mmwave/train.py`
- Modify: `configs/parrad-stage1.yaml`
- Modify: `tests/test_parrad_training.py`
- Modify: `README.md`

**Interfaces:**
- Produces: independent label-run/streaming learning rates, warmup-cosine scheduling, phase-local early stopping, and `best_label_run.pt` / `best_streaming.pt`.
- Produces: continuous evaluation with macro, raw, segment, class, and sensor metrics.

- [x] Add failing unit tests for the learning-rate schedule and phase-local early stopping.
- [x] Add failing evaluation tests for confusion-derived metrics and chunk invariance.
- [x] Implement phase controls and extended metrics.
- [x] Update production config and documentation.
- [x] Run the 3+1 epoch remote CUDA pilot before full training.
