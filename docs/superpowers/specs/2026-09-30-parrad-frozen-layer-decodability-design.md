# PARrad Frozen Layer Decodability Audit — Design

**Status:** Approved with four constraints, 2026-09-30  
**Date:** 2026-09-30  
**Workspace:** `D:\lab\SEER`  
**Purpose:** Locate where a frozen online Stage P3 Candidate loses or retains information about a real event transition, without changing the model or training Verify.

## 1. Scope and interpretation

The primary question is not whether one hand-chosen cosine direction works, but whether a fixed-capacity linear readout can decode a GT event change at each existing layer. This is a **diagnostic**: it may train logistic-regression probes on frozen features, but it must not update Radar Encoder, Event Former, Semantic Projection, Stage P3 heads, thresholds, or formal controller. It must not read validation or test data during extraction, probe fitting, hyperparameter choice, or model selection. Existing validation findings motivate the experiment but are not counted as independent evidence.

The current cache has 24 train recordings, subject_02/03/04 × four sessions × two sensors (80,170 updates). Three leave-one-subject-out folds are possible. The frozen Stage P2/P3 checkpoints were themselves trained/selected with existing data; the subject holdout is **only for the new probe**, not an unseen-subject estimate of the entire system.

## 2. Exact computational boundaries

Use frozen Stage P2 cache `radar_tokens` as `F_t ∈ R^{4×256}`; its fingerprint must match the Stage P2 checkpoint. Instrument the existing online `StreamingSemanticEventFormer._update` during the same forward that the controller uses. The actual code path is:

```text
F_t
  → LLM layers 1–4 with Radar Cross-Attention
  → H4_t = final event-token hidden, shape [8,896]
  → R_t = float32(H4_t) − float32(E_{t−1})
  → Ê_t = W_out(R_t), shape [8,896]
  → α_t = sigmoid(gate_network(E_{t−1}, Ê_t−E_{t−1}, maturity_t))
  → E_t = lerp(E_{t−1}, Ê_t, α_t), shape [8,896]
  → z_t = SemanticProjection(E_t), shape [128]
```

`H4_t`, `R_t`, `Ê_t`, gate values, `E_t`, and `z_t` are all retained. The proposed `tanh(λ)` formula and direct `W_out(H4)` are **not** the current implementation; using either would invalidate layer attribution. The gate input delta `Ê_t−E_{t−1}` is also retained or reconstructible. Instrumentation must observe the existing operation, not run a second approximation. With the same batch-one/prompt-cache path, traced `E_t` and `z_t` must match controller Candidate/Shadow tensors at every checked update within `1e-5`; any mismatch aborts the audit.

At each natural Local/Radar Proposal, freeze the Active reference and its last successful `F/H4/R/Ê/E/z` trace. Map all later Candidate traces to this reference. Active updates, Candidate updates, Shadow updates, and confirm/reject transitions must be tagged explicitly; on confirm the last Active trace becomes Candidate, on reject it becomes Shadow. Natural controller decisions remain untouched. A separately marked forced-horizon copy may supply age 4/7/13 comparisons for a fixed candidate cohort, exactly as in the earlier validation-only probe; it must not change natural decisions. Report natural-survivor and forced-horizon results separately to expose censoring.

## 3. Two non-interchangeable labels

1. **Proposal-level boundary label:** retain the established `clean_proposal_label`: one GT boundary within `±0.72 s` of Proposal and no second boundary through the 2.16 s horizon is positive; boundary-free trajectories are negative; other cases are excluded. The label is attached once per Proposal. Do not treat its repeated update rows as independent samples.
2. **Update-level same/different run label:** reference run is the clean GT run of the immediately preceding update `t_p^-`. At age `t`, label 0 if the current clean update remains in that run; label 1 if exactly one valid GT boundary occurred and the current clean update belongs to the new run. Exclude an update when either frame is `-100`, its two frames straddle a boundary, the reference update is mixed/ignored, an ignored gap intervenes, or two or more boundaries occurred since the reference. The cached `oracle_run_id` alone is insufficient because it records the last valid frame in an update; cross-check frame labels and `extract_event_runs`. This is a GT proxy for Frozen Active's event identity, not proof that the learned Active state represents that run.

For both tasks, evaluate predeclared ages 1, 4, 7, and 13 (0, 0.54, 1.08, 2.16 s). Report eligible positives/negatives and missingness at every age. Each age-specific fit uses at most one row per Candidate, so age and candidate lifetime cannot act as hidden duplicate-sample shortcuts. No future GT enters a feature; GT is used only for retrospective labels.

## 4. Comparable geometry and linear probes

For each layer and Candidate/reference pair, save fixed cosine distance and L2 distance, plus finite/norm checks. For the probe, flatten layer tensors in a fixed token order and build relational input from the current-minus-reference vector; append its fixed cosine and L2 scalars. This makes the same/different task actually relational rather than asking a single current vector to reveal its unknown reference. A current-only 128-dimensional probe for the Proposal label is a secondary check.

Raw layer dimensions differ (F 1024; H4/R/Ê/E 7168; z 128). A logistic regression on each raw flattened vector would **not** have equal learned capacity. The primary comparison therefore uses a label-free, fixed-seed random sign projection of the relational vector to 128 dimensions, then appends the same two geometry scalars: **130 input features, 131 learned logistic parameters including bias, at every layer**. The z layer uses a seeded 128×128 projection as well for an identical protocol. Predeclare three projection seeds; report median and full range rather than selecting the best seed. A high-dimensional ridge/logistic probe may be shown separately as a sensitivity upper bound, never mixed into the capacity-matched layer ranking. A weak projected result alone cannot prove information absent from the full representation.

Per fold, calculate standardization means/variances and class weights on the two probe-training subjects only. Fit the same L2-regularized logistic objective, optimizer tolerance, maximum iterations, and fixed regularization for all layers, labels, ages, and seeds; do not tune on the held-out subject. If a fold lacks both classes, report AUROC as undefined instead of inventing a value. Save coefficient norms, convergence status, finite checks, train metrics, and held-out scores. Geometry-only and prevalence/age-only baselines use identical held-out rows. No layer or score direction is selected on validation.

Primary split: leave one entire train subject out per fold, including all its sessions and both sensors. Report held-out AUROC, AP, prevalence, counts, and balanced accuracy per fold; session- and sensor-stratified metrics are descriptive. Aggregate with subject-macro mean and range, not pooled-update confidence intervals. A three-subject fold range is not a formal generalization guarantee.

## 5. Decision rules and deferred sequence check

Layer localization requires a **material, repeatable** gap in capacity-matched held-out probe performance across subjects and projection seeds, with adequate class counts and nondegenerate training. Do not infer a bottleneck from raw cosine alone, one favorable fold, a single projection seed, or tiny near-zero distances. Compare F→H4, H4→R→Ê, Ê→E, and E→z against the exact runtime operations above.

If single-update probes are weak or inconsistent, a separate predeclared follow-up will test 3- and 8-update trajectory features with subject-disjoint evaluation and the same learned parameter budget. The validation raw-radar cosine rise at 2.16 s motivates that follow-up, but is not evidence that the trajectory probe will succeed. No Event Former retraining or controller change is authorized by this diagnostic alone.

## 6. Implementation and safety contract

- Add a diagnostic-only trace path to the existing Event Former forward that leaves its returned tensors and controller behavior unchanged when disabled. Test exact trace values and no-trace equivalence with both prompt-cache and non-cache paths.
- Build a new Stage P3 train-split extraction command; refuse validation/test records, checkpoint/cache fingerprint mismatches, existing output directories, nonfinite tensors, or trace/controller mismatch. Use batch-one replay. Save a manifest with source hashes, split, subject/session/sensor, candidate/age, label-exclusion reason, and whether a row is natural or forced.
- Keep large tensor shards and probe outputs under a new directory on the 4 TB PARrad volume, never in prior result directories. Avoid loading all high-dimensional vectors into RAM; shard by recording and project/fit fold-wise. Record tensor dtype and numeric scale before interpreting near-zero cosine distances.
- Build a separate probe evaluator that reads only the new train-split shards, enforces subject-disjoint folds and train-only transforms, and writes all per-fold/per-seed predictions and aggregate metrics. It must never auto-promote a probe into the formal Verify head.
- Test label edge cases (`-100`, mixed update, repeated label, multiple boundaries), branch trace mapping, exact network equivalence, fold leakage, dimensional equality, and output immutability. Run a one-recording smoke before the 24-recording extraction; only after the smoke passes run the full audit.

## 7. Non-goals and review question

No Stage P2/P3 training, Verify retraining, threshold recalibration, validation/test evaluation, or new event-start token. The first deliverable is a frozen train-only diagnostic with reproducible per-layer and per-subject results; any architecture change is a later decision.

**Review question:** approve the capacity-matched 128-dimensional fixed random projection (three predeclared seeds) as the primary linear-probe protocol, with raw-dimensional probes only as secondary sensitivity analysis? This choice determines how strongly a weak probe can be interpreted.

## 8. Approved pre-run constraints (2026-09-30)

- Trace the actual `F → H4 → R → Ê → E → z` runtime path: `R=H4−E_prev`, `Ê=W_out(R)`, `α=sigmoid(gate(...))`, `E=lerp(E_prev,Ê,α)`. Abort if traced `E/z` differs from controller values by more than `1e-5` at any checked update.
- Record `candidate_id`, `proposal_update`, `active_ref_update`, and `active_ref_state_id`. Resolve the reference from the controller's *actual* frozen Active state, not an assumed preceding update; assert reference trace `E/z` equals `reference.event_tokens`/`reference.semantic_key` within `1e-5` and all relational features use that state ID.
- Fix projection seeds to `[3407, 7211, 12893]` before seeing results; use them for every fold/layer/age. Primary probe always has 130 inputs and 131 learned logistic parameters.
- Fix the probe to weighted binary cross-entropy plus `0.5 × 1e-3 × ||w||²` (bias unpenalized), deterministic CPU `torch.optim.LBFGS` with `line_search_fn="strong_wolfe"`, `max_iter=200`, `tolerance_grad=1e-7`, `tolerance_change=1e-9`, and a classification threshold of `0.5`. Estimate standardization and inverse-frequency class weights only from the two probe-training subjects. Record these values and software versions in the manifest. No held-out-subject threshold or hyperparameter search.

The user approved this design with these constraints and authorized the diagnostic run. Natural and forced-horizon cohorts, and proposal- versus update-level labels, remain separate. No production model, Verify head, or controller change is authorized.
