# Oracle → Proposal domain-gap diagnostic (approved 0-update execution)

Scientific target: explain frozen Oracle50's Oracle/replay gap without training or selecting a checkpoint/threshold. O = existing clean random Oracle k=6 pairs; T = fixed-grid time anchors; P = existing frozen Proposal replay. Every reference has 12 updates and candidate 7. All upstream modules and Oracle50 remain frozen. No official holdout, Semantic/Candidate adaptation, trigger tuning or new production model.

## Locked methods

- Use the existing 26 TRAIN-derived streams and their unchanged train/calibration/dev partitions. Dev was already seen; this is retrospective attribution, not independent validation.
- T anchors j=18,21,24,... while j+7<=N. Reference tokens [j-18:j-6), Candidate [j:j+7). Geometry/context determine key extraction; protocol GT only labels/masks outputs afterward. Decision delay 1.08s; no new controller or operational event metrics.
- Mask nonhomogeneous or invalid supports and any frame within 0.36s of a protocol boundary, identical to prior pilot. Same/different denotes action class, not event identity. Retain all masked anchors in raw NPZ. Flag exact Proposal/grid overlap; also report T excluding exact P anchors as sensitivity.
- Load O/P keys and Oracle50 checkpoint from immutable pilot v2. New T keys use the identical compatible frontend and 64-window batches. Verify identical operators/inputs/batch shape exactly; preserve prior cross-batch FP16 error separately. Do not recertify its failed 1e-5 cross-batch test.
- Natural point metrics: AUROC, AP, class prevalence, same/different score distributions; weighted AP at 50/50 class mass. Keep cosine-distance baseline and swapped-input Head scores without orientation selection.
- Matching strata: (stream recording ID including sensor, reference action A, candidate action B). For domains being compared, retain only strata present in all; assign stratum mass equal to the minimum sample count across domains and divide that mass equally among each domain's samples. Normalize positive and negative mass to 1/2. Missing support is excluded and disclosed, never filled with zero. Report pairwise O/T, T/P, O/P and joint O/T/P coverage, counts and metrics. If either class absent, metrics are null. Do not match on any predicted score.
- Reference-action analysis compares A→A negatives against all A→B positives; a fixed different A→B pair alone has undefined AUROC. Disclose full action-pair counts/score distributions.
- Oracle temporal diagnostics: candidate before/after reference; center-time gaps <=5s, (5,30]s, >30s; same-label same contiguous GT occurrence vs repeated occurrence. No bins chosen from scores.
- All eight dev subject×sensor cells; leave-one-recording-out sensitivity drops both sensors for that subject-session. These are sensitivity ranges, not independent-sample confidence intervals or causal percentages.
- O→T and T→P differences suggest local temporal geometry versus selection effects; they do not identify additive causal contributions. Do not infer P/Encoder failure from one frozen Head.

## Tasks / completion contracts

1. RED→GREEN tests for weighted tied metrics, matching coverage and masks, causal anchor geometry, occurrence identity and extraction operator. Pin sources and create exclusive capsule/remote output.
2. Frozen T extraction + O/P fixed-Head forward, same-batch/operator/reset/future-perturbation checks, source/checkpoint/frontend digest before/after; optimizer calls = 0.
3. Compute all predeclared natural/matched/grouped/temporal/swapped analyses and complete raw CSV/NPZ. No score-based sample selection.
4. Independently recompute labels, scores, metrics and frozen digests; archive raw JSON/NPZ/log/code with member SHA; verify archive SHA and safe paths before local extraction. Report limitations and practical routing, one fresh whole-change review.

Deliver docs/stage3b-oracle-proposal-domain-gap-results-2026-10-10.md, plots, CSV, JSON/NPZ/log and SHA manifest. Prior artifacts remain immutable.
