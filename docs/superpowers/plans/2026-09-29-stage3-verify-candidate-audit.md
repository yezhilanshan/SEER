# Stage 3 Verify Candidate-Level Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the frozen Stage 3 Verify inputs contain usable information about independently counted online Candidates before changing the model or training again.

**Architecture:** Finish the already-running validation-only post-training calibration, then replay frozen Stage 3 recordings into one row per Candidate. Compare decision-time and retrospective feature separability with small diagnostic probes; use the evidence to choose either a candidate-level Verify training design or an upstream Proposal/Semantic investigation.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, existing PARrad Stage 3 cache and controller.

**Spec:** `docs/superpowers/specs/2026-09-24-parrad-stage3-transition-design.md` (Sections 6-8, 10, 13), supplemented by the 2026-09-29 discussion of proposal-level labels and update-level samples.

## Global Constraints

- Use only `train` for probe fitting and `validation` for selection; do not load or evaluate `test`.
- Keep Stage 1/2, Physical, Semantic, both Verify checkpoints, and existing calibration artifacts immutable.
- Do not begin another long training run before the diagnostic decision gate.
- Keep sustained GPU temperature below 84°C; run replay jobs sequentially if needed.
- The workspace has no Git metadata; record commands, test results, checkpoint SHA-256 values, and artifact paths in the diagnostic report instead of committing.

## Review Focus

- Identical proposal timestamps in different recordings must remain separate Candidates (Task 2 test).
- Candidate updates after maturity saturates must retain their exact update order and count (Task 2 test).
- A Candidate that ends with no Confirm/Reject must still appear with an explicit terminal reason (Task 2 test).
- Ignored `-100` label gaps must not create artificial true boundaries (Task 2 test).
- A sensor group with no positive or no negative Candidates must report an undefined AUC, not a fabricated score (Task 3 test).

---

### Task 1: Close the existing post-training calibration

**Files:** No source edits. Read the three shards under `timeout-reject-post-hard-negative-v1`; write merged validation artifacts beside them.

**Interfaces:** Consumes `verify-hard-negative-v1/best_verify.pt` and the original merged timeout/reject thresholds. Produces a validation-only selected threshold artifact and an old-versus-new closed-loop comparison.

- [ ] Check that the sequential runner finished successfully and each shard contains exactly six sweep rows, with no traceback or CUDA OOM.
- [ ] Verify all contexts agree on `source_split=validation`, validation fingerprint, grid, and the new checkpoint SHA-256; verify `test_evaluation_allowed=false`.
- [ ] Merge using `merge_validation_sweeps` and `timeout_reject_grid(original_merged_thresholds, proposal=0.55)`; fail closed on any mismatch.
- [ ] Report F1@0.36, F1@0.72, precision, recall, over/under-segmentation, decision counts, latency, and state-norm finiteness against the original checkpoint on the same validation split.
- [ ] Record artifact hashes and the exact merge command in the diagnostic report.

### Task 2: Audit one independent Candidate per proposal

**Files:** Create `src/continuous_mmwave/parrad_stage3_candidate_audit.py`; test in `tests/test_parrad_stage3_candidate_audit.py`. Do not change the existing miner's training behavior in this task.

**Interfaces:** Implement `collect_candidate_episodes(runtime, records, thresholds, boundary_tolerance_s) -> tuple[CandidateEpisode, ...]`. Each episode identifies `(recording_key, proposal_time)`, sensor, post-hoc boundary distance/label, ordered `(difference, stability, maturity)` updates, decision, and terminal reason. It must replay only the caller-specified split.

- [ ] Write tests for one label per proposal, cross-record identity, maturity saturation, unfinished Candidates, and `-100` gaps; run each test and confirm it fails for the missing interface.
- [ ] Implement the smallest read-only collector using the controller's existing `step_tokens` path; do not infer Candidate boundaries from feature similarity.
- [ ] Run the tests in the isolated SEER environment and confirm they pass without changing Stage 1/2 or the original miner.
- [ ] Replay `train` and `validation` separately; save candidate counts, lifetime distributions, positive prior, and per-sensor counts with a source split and checkpoint SHA in every output artifact.

### Task 3: Measure decision-time information, then decide

**Files:** Extend `src/continuous_mmwave/parrad_stage3_candidate_audit.py`; extend `tests/test_parrad_stage3_candidate_audit.py`; write a validation-only diagnostic report under a new output directory.

**Interfaces:** A probe consumes Task 2 episodes and produces metrics for (a) the first scored Candidate update, including its legitimate zero stability when history is short, (b) the final update as a retrospective upper bound, and (c) an episode summary. It never fits on validation data.

- [ ] Test that short Candidates are counted rather than silently discarded, train-derived normalization is never fitted on validation, and one episode contributes one target to each probe.
- [ ] Compute each feature's univariate AUROC/PR-AUC, positive/negative distributions, and Candidate counts overall and by sensor; use candidate prevalence as PR baseline. Score both predeclared label tolerances, +/-0.36 s and +/-0.72 s, from the stored boundary distance.
- [ ] Fit only inexpensive diagnostic heads on `train`: `Linear(3,1)` and `Linear(3,16)->ReLU->Linear(16,1)`, seed 42, AdamW learning rate 1e-3, 20 fixed epochs, train-fitted standardization, and no validation-tuned hyperparameters. Evaluate once on subject-disjoint `validation`, with recording-level bootstrap intervals. Compare against the current non-negative linear Verify.
- [ ] **Go gate:** Propose new Verify training only if the first-scored-update probe reaches validation AUROC >= 0.60, PR-AUC at least 0.05 above Candidate prevalence, and AUROC >= 0.55 on each sensor with defined metrics. If only the retrospective endpoint separates, investigate timing/feature accumulation instead of treating it as an online solution.
- [ ] **No-go gate:** If even the retrospective probe is near chance, stop Verify hyperparameter tuning and trace Proposal alignment and Semantic Difference outputs by sensor and boundary distance. Do not use `test` to break a tie.
- [ ] Publish the evidence, the selected branch, commands, split identities, and hashes; request review before any new long training run.
