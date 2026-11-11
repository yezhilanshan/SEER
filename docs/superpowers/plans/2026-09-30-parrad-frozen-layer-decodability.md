# PARrad Frozen Layer Decodability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen, train-only, subject-disjoint layer decodability audit for F, H4, R, Ê, E, and z, without modifying the formal Stage P3 model or controller behavior.

**Architecture:** A diagnostic trace tap observes the existing Event Former forward and captures pre/post-gate tensors. A separate train-only replay module binds each Candidate to the controller's exact frozen Active state and emits natural and forced-horizon shards. A probe evaluator applies fixed 128-dimensional random projections and a capacity-matched logistic probe to the two independent labels.

**Tech Stack:** Python, PyTorch, NumPy, pytest, PARrad Stage P2 cache, Stage P3 controller.

**Spec:** `docs/superpowers/specs/2026-09-30-parrad-frozen-layer-decodability-design.md`

## Global Constraints

- Read only `train` records: subject_02/03/04, 24 recordings. Never load validation/test.
- No production weight update, controller policy change, Verify training, threshold calibration, or existing-output overwrite.
- Exact path `F → H4 → R → Ê → E → z`; trace/controller tolerance `1e-5`.
- Record `candidate_id`, `proposal_update`, `active_ref_update`, `active_ref_state_id`; assert the reference matches the frozen controller state.
- Fixed seeds `[3407, 7211, 12893]`; 130 probe inputs/131 parameters; fixed CPU LBFGS, lambda `1e-3`, max_iter `200`, tolerance_grad `1e-7`, tolerance_change `1e-9`, threshold `0.5`.
- Natural survivor and forced horizon are distinct cohorts. Proposal and update labels are distinct tasks.
- This workspace has no `.git`; record verification locally, but skip commit-specific steps from the generic skill.

## Review Focus

- Prompt-cache and non-cache paths must expose the same tensor boundaries without changing the result; test both paths.
- A proposal whose Active trace is stale or absent must abort, not silently fall back to `proposal_update-1`; test this mismatch.
- `-100`, mixed-frame, ignored-gap, and multiple-boundary updates must be excluded; test each case.
- No label or standardization leakage from held-out subject; test a shifted held-out distribution.
- Existing output directories must be refused before writing; test immutability.

---

### Task 1: Diagnostic Event Former trace

**Files:** Modify `src/continuous_mmwave/models/event_former.py`; test `tests/test_parrad_stage3_frozen_layers.py`.

**Interfaces:** Produce an optional diagnostic trace callback on `stream_update` with F-independent `H4`, `R`, `E_hat`, `alpha`, and `E` tensors; default `None` and no behavior change. The replay owns `F` and computes `z` from the exact controller projection.

- [ ] Write tests that trace values obey `R=H4-E_prev`, `E_hat=W_out(R)`, `E=lerp(E_prev,E_hat,alpha)` and equal untraced outputs, using both prompt-cache and non-cache fixtures.
- [ ] Run targeted pytest; observe missing trace interface failure.
- [ ] Implement the optional trace callback only on the existing forward path.
- [ ] Run targeted and existing Event Former tests; confirm pass.

### Task 2: Strict reference and label contract

**Files:** Create `src/continuous_mmwave/parrad_stage3_frozen_layers.py`; test `tests/test_parrad_stage3_frozen_layers.py`.

**Interfaces:** Produce functions for clean frame-run classification, update-level same/different labels, and an assertion-checked `ActiveReferenceTrace` carrying the four required identifiers and all six layers. Replay in Task 3 consumes these functions.

- [ ] Write tests for `-100`, mixed update, ignored gap, multiple boundaries, stale state ID, and E/z mismatch.
- [ ] Run targeted pytest; observe expected missing behavior failure.
- [ ] Implement minimal pure label/reference functions; do not trust `oracle_run_id` alone.
- [ ] Run targeted pytest and confirm pass.

### Task 3: Train-only natural and forced extraction

**Files:** Extend `src/continuous_mmwave/parrad_stage3_frozen_layers.py`; test `tests/test_parrad_stage3_frozen_layers.py`.

**Interfaces:** A CLI loads only `train` cache records and frozen runtime; writes one recording per tensor shard plus manifest under a new output directory. Emits both cohort IDs, both label tasks, ages 1/4/7/13, eligibility reasons, and exact reference IDs. Task 4 reads this manifest/shards.

- [ ] Write tests for split refusal, duplicate output refusal, reference/controller mismatch abort, candidate lifetime mapping, and natural/forced separation.
- [ ] Run targeted pytest; observe expected failure.
- [ ] Implement batch-one replay using existing Stage P3 runtime and forced-horizon copy; do not change controller decisions.
- [ ] Run targeted tests, then one-recording remote smoke. Check checkpoint/cache fingerprints and exact trace agreement before full extraction.

### Task 4: Capacity-matched linear probe

**Files:** Create `src/continuous_mmwave/parrad_stage3_linear_probe.py`; test `tests/test_parrad_stage3_linear_probe.py`.

**Interfaces:** Evaluate train-only shards in three leave-one-subject-out folds; fixed sign-projection seeds, train-only standardization/class weights, fixed weighted LBFGS logistic, held-out AUROC/AP/balanced accuracy, raw geometry and prevalence baselines; secondary raw-dimensional sensitivity labeled separately.

- [ ] Write tests that every layer produces 130 inputs/131 parameters, seeds are deterministic, held-out perturbations do not alter training transforms, and undefined one-class folds are reported not fabricated.
- [ ] Run targeted pytest; observe expected failure.
- [ ] Implement evaluator and manifest/report writer with fixed settings.
- [ ] Run targeted pytest and full project pytest; record unrelated failures by name.

### Task 5: Full frozen audit and interpretation

**Files:** Create `docs/stage3-frozen-layer-decodability-2026-09-30.md` and new remote output directory only.

**Interfaces:** Consume verified extraction and probe results; report per-subject/sensor/session metrics and seed ranges by layer, age, cohort, and label, then identify only reproducible information cliffs.

- [ ] Verify one-recording smoke is finite and E/z matches controller at every checked update within `1e-5`.
- [ ] Run 24 train recordings; do not access validation/test or alter existing outputs.
- [ ] Fit/evaluate frozen probes; verify manifests, class counts, split separation, and report all weak/undefined findings honestly.
- [ ] Write interpretation and explicitly state that no production model/controller was changed.
