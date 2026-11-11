# Stage 3 Proposal Component Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This checkout has no `.git`; record RED/GREEN and hashes in a plan-specific ledger instead of commits.

**Goal:** Measure closed-loop consequences of Local-only, Radar-only, and existing noisy-OR Proposal with all other Stage 3 behavior frozen.

**Architecture:** Replay the same eight validation cache recordings through the existing controller three times. Diagnostic wrappers suppress one physical score without changing prototype updates, Event Former state, Semantic Difference, Verify, thresholds, or checkpoint weights. Use the existing one-to-one boundary matching implementation, not the earlier single-update attribution.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest; isolated server SEER Conda environment.

**Spec:** `docs/stage3-upstream-audit-2026-09-29.md` and `docs/superpowers/specs/2026-09-24-parrad-stage3-transition-design.md` §5, §9–10.

## Global Constraints

- Use original `best_verify.pt` SHA `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6` and its original validation-only thresholds; no per-mode retuning.
- Keep all model weights and source artifacts byte-identical. Run only validation cache; do not read test or create a test-evaluation allowance.
- Sequential modes on GPU 0; pause on nonfinite state or sustained temperature >84°C.
- Treat unmatched-confirm fraction as an ablation metric, not the official false-confirm definition.

## Review Focus

- Radar suppression must still update physical prototype states identically.
- Local suppression must leave previous-radar tracking and masking intact.
- Each recording and mode must reset controller state but preserve continuous state within the recording.
- All modes must use the same validation fingerprint, one-to-one matching tolerance, and thresholds.
- A mode with zero confirmations must report defined precision/recall conventions, not divide by zero.

---

### Task 1: Diagnostic component suppression

**Files:** Create `src/continuous_mmwave/parrad_stage3_proposal_ablation.py`; test `tests/test_parrad_stage3_proposal_ablation.py`.

**Interfaces:** `proposal_mode(runtime, mode: Literal['or','local','radar'])` is a scoped diagnostic context manager. In local mode the Radar Transition head still returns its real updated prototype but its logit is masked to a finite low value; in radar mode Local Change's logit is masked. On context exit restore original modules. Default/or mode is unchanged.

- [ ] Write RED tests comparing prototype updates and default outputs across modes, including context cleanup after an exception.
- [ ] Implement minimal wrappers and run targeted tests to GREEN.

### Task 2: Matched validation-only closed-loop replay

**Files:** Continue `src/continuous_mmwave/parrad_stage3_proposal_ablation.py`; test `tests/test_parrad_stage3_proposal_ablation.py`.

**Interfaces:** `evaluate_cached_ablation(runtime, records, thresholds, mode)` accepts only validation `Stage3CacheItem` records; returns overall and per-sensor F1/precision/recall at ±0.36/0.72 s, confirm/proposal/reject counts, over/under-segmentation, median confirmation delay, and finite state norm. Reuse `match_boundaries_seconds` rather than inventing a different matcher.

- [ ] Write RED tests for validation-only guard, tiny recording matching, reset-on-recording, zero-confirm edge case, and identical OR counts to the previous cached-token Candidate audit on the same fixture.
- [ ] Implement read-only replay; run targeted and controller suite to GREEN.
- [ ] Run OR/Local/Radar sequentially on eight validation recordings; save separate immutable JSON artifacts and SHA-256 markers under a new output root.
- [ ] Compare per-sensor recall as well as F1: no component is declared safe to remove solely because it reduces far-away Proposals.

### Task 3: Verification and handoff

**Files:** Add a dated result report under `docs/`; update plan-specific ledger.

- [ ] Run full remote pytest and report known unrelated failures by name.
- [ ] Verify no test reads, checkpoint hashes unchanged, finite state, and all three modes shared the same validation fingerprint.
- [ ] Recommend a later configuration change only if the closed-loop trade-off is unambiguously favorable; this plan itself does not change the production Proposal rule.
