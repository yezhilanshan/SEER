# Stage 3 Semantic Hard-Pair Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This checkout has no `.git`; record RED/GREEN and hashes in a plan-specific ledger instead of commits.

**Goal:** Test whether same-session adjacent oracle pairs can make Semantic Difference distinguish transitions without retraining Stage 1/2, Physical, or Verify.

**Architecture:** Extend only the Stage 3 semantic-pair dataset with balanced adjacent positive and within-run negative pairs at matched maturity. Train the Semantic Difference head alone from the original frozen Stage 3 checkpoint in a new output directory; select on held-out validation adjacent pairs, then run one validation-only online Candidate diagnostic. No test access.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest; isolated server SEER Conda environment.

**Spec:** `docs/stage3-upstream-audit-2026-09-29.md` and `docs/superpowers/specs/2026-09-24-parrad-stage3-transition-design.md` §6–8.

## Global Constraints

- Keep Stage 2 checkpoint, original Stage 3 checkpoint, Radar Encoder, Event Former, Semantic Projection, Physical heads, and Verify head byte-identical.
- Build training pairs from train cache only; validation pairs from validation cache only; never load test.
- Exclude transitions across `-100` gaps; keep paired sensor views in their existing subject-disjoint split.
- Use new output directory under `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2` and record checkpoint/cache hashes.
- GPU temperature must not stay above 84°C; use a short pilot before any longer run.

## Review Focus

- One-update or missing-run event: skip invalid adjacent pair without losing balance.
- Ignored gap: never call a pair adjacent across `-100`.
- Maturity mismatch: positive and its matched negative receive exactly the same scalar maturity.
- Split leakage: no train pair contains validation or test key.
- Checkpoint selection: do not choose the pilot from the old cross-session metric alone.

---

### Task 1: Deterministic same-session hard pairs

**Files:** Modify `src/continuous_mmwave/parrad_stage3_data.py`; test `tests/test_parrad_stage3_data.py`.

**Interfaces:** `SemanticPairDataset(records: tuple[Stage3CacheItem, ...], seed: int = 42, include_adjacent: bool = False)` retains the old behavior by default. With `include_adjacent=True`, add two balanced positive/negative pairs per direct GT-run boundary, using the preceding run's mature key as reference and explicit matched maturity. Expose pair kind for auditing.

- [ ] Write tests: one direct boundary yields matched nascent/mature positive-negative pairs; `-100` gap yields none; one-update runs are handled; fixed seed is deterministic and split-local.
- [ ] Run targeted pytest and see expected RED failures before code.
- [ ] Implement only pair generation and explicit maturity handling; preserve existing four base pair kinds and default behavior.
- [ ] Run targeted and adjacent Stage 3 tests to GREEN; audit real train/validation pair counts and same-session percentages.

### Task 2: Semantic-only short pilot and frozen validation gate

**Files:** Create `src/continuous_mmwave/parrad_stage3_semantic_pilot.py`; test `tests/test_parrad_stage3_semantic_pilot.py`; optionally reuse read-only metric functions in `parrad_stage3_upstream_audit.py`.

**Interfaces:** A CLI consumes the config, original `best_verify.pt`, and train/validation caches; outputs a new Semantic-only checkpoint and per-epoch validation JSONL. It must fail on source SHA/cache mismatch or pre-existing nonempty output; only `semantic_difference` parameters require gradients. Use seed 42, batch 64, learning rate `1e-4`, at most 3 epochs. Select by validation adjacent mature paired-win rate, breaking ties with adjacent nascent rate and old cross-session AUROC.

- [ ] Write RED tests for trainable-parameter isolation, checkpoint provenance, validation-only selection, and rejection of test split/output overwrite.
- [ ] Implement the minimal pilot runner and run unit/integration tests to GREEN.
- [ ] Run a one-batch CUDA smoke and inspect finite loss, gradients, GPU temperature, and checkpoint hashes.
- [ ] Run at most 3 epochs on GPU 0; report each epoch's adjacent nascent/mature paired-win and AUROC by sensor, plus cross-session AUROC. Do not continue automatically if loss is nonfinite or temperature sustains >84°C.
- [ ] Only if validation mature paired-win ≥0.60 on both sensors, nascent paired-win ≥0.55 on both sensors, and old cross-session AUROC ≥0.65, run one frozen validation-only online Candidate replay with the pilot Semantic head. Otherwise stop and report no-go.
- [ ] For any online replay, compare original Candidate first-update GT-run-switch proxy AUROC 0.399 with the pilot and report both sensors; do not train Verify or touch test.

### Task 3: Verification and handoff

**Files:** Add a dated diagnostic report under `docs/`; update plan-specific ledger.

- [ ] Run full remote pytest; name every failure, including known Gemma failures, without changing Gemma.
- [ ] Check `source_split=validation`, `test_evaluation_allowed=false`, original checkpoint SHA, and output SHA.
- [ ] State clearly whether the hard-pair pilot improved online evidence; do not infer improvement from oracle-pair validation alone.
