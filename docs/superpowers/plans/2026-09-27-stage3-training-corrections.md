# Stage P3 Training Corrections Plan

**Goal:** Correct the Stage P3 supervision and train/inference contracts exposed by the first formal run, without reading the test split.

**Spec:** `docs/superpowers/specs/2026-09-24-parrad-stage3-transition-design.md`

## Constraints

- Keep the accepted Stage P2 checkpoint and cache byte-identical.
- Use validation only for model selection and controller calibration.
- Keep the existing pairwise physical API for compatibility, but make production training use event-local chronological sequences.
- Preserve `last_verify.pt` as a historical fallback; new training writes to a new output directory.

## Task 1: Semantic pair coverage

- Generate four balanced pair types for every eligible event, deterministically and within one split.
- Prefer cross-subject/session peers for both same-class and different-class pairs.
- Test event coverage, balance, determinism, and split isolation.

## Task 2: Chronological physical supervision

- Add event-local physical sequences containing the full current run plus the first update of the next directly adjacent run.
- Maintain Radar Transition fast/slow/Welford state across the sequence and reset only between oracle events.
- Apply a training-split-derived soft positive weight and mask padding from loss and metrics.
- Report Local, Radar, and noisy-OR metrics plus recall at FPR <= 10%.

## Task 3: Verify train/inference consistency

- Train Verify from the frozen Semantic Difference probability used by the controller, not raw cosine distance.
- Repair trajectory examples so Active key, Candidate history, maturity, and targets are explicit.
- Prefer checkpoints satisfying false-confirm <= 10%; within that constraint rank by PR-AUC and balanced accuracy.

## Task 4: Integration and remote smoke

- Run the focused and full local suites in the `interview` Conda environment.
- Sync only changed Stage P3 files to `/home/xibeidaxue/SEER`, after timestamped backups.
- Run the remote focused suite and a CUDA one-step smoke using a fresh output directory.
- Do not start formal training until the smoke metrics and dataset counts are reviewed.
