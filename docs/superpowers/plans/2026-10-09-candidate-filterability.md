# Candidate Filterability Diagnostic Implementation Plan

> Execute inline with executing-plans, TDD and final fresh read-only review. The user approved the preceding four-step experiment; this document locks remaining implementation choices before training.

**Goal:** Test whether frozen causal Difference/Stability evidence can filter the fixed Radar+GRU candidate pool over 0.18–1.08s.
**Spec:** docs/radar+gru融合方案.md stages4–8, narrowed by the user-approved chat plan. No formal Verify Head or controller installation.

## Locked design

- Radar0.38 + independent GRU0.44, original lag1/NMS0.36/neighbor-high, immutable first-arrival cross-source merge0.36. Original ±0.72s matcher and protocolGT unchanged.
- Exactly26 pinned records:18fit/8dev. Diagnostic train=fit set000 (10 sensor streams), calibration=fit set001 (8 streams); both sensors from one recording stay together. Upstream models previously saw these TRAIN subjects; calibration is only held out from these new probes.
- Causal shadow replay, not closed-loop performance: one frozen EventFormer active trajectory per recording, reset only at recording start and never GT. Before proposal update j, snapshot Active=P(E[j-1]); never change it during that candidate. Candidate starts from fitted initial event tokens and consumes tokens j,j+1,…,j+6. Multiple candidates are independent shadows. GRU never resets within recording; reference=h[j-1]. This continuously advancing active reference is a diagnostic convention, not a calibrated Confirm-driven active branch.
- Use the already verified positive-coordinate compatible frontend/P, not random v2.5 gate. All frontend/Encoder/P/Evidence/oldHeads remain frozen. Frozen bundle SHA57af183acfdcc1da33c76573b60b57ae60897fd9aca715484058b37340378602, automatically read from the independently verified prior evidence archive. Source/code identities verified before/after.
- At each k=1..6 (delay k*.18s after emission), Difference=cosine distance and relative L2 from frozen reference in GRU and semantic-key spaces (4 scalars). Stability=normalized GRU variance and semantic-key variance over last up to3 observed updates, plus consecutive GRU and semantic-key movement (4 scalars). No source/score/label/absolute time/recording ID/maturity input. Each readout padded to8 inputs: Difference,Stability,Combined. All linear8→1, zero initialization, seed42, unweighted candidate BCE, full diagnostic-train batch, FP32, AdamW500steps/lr0.01/wd0.01/clip1. No sweep/restarts/early stopping. 18 probes, 9,000 updates confined to those probes; 0 original updates.
- Fit normalization only diagnostic-train. Training/AP maturity comparisons use the common complete6-update cohort to avoid changing difficulty with k. Online/event metrics retain the entire pool denominator and conservatively reject candidates missing the required prefix; report censoring and rejects. Future availability is only offline cohort accounting, never an input feature.
- Protocol labels from original full-pool one-to-one matching; additionally retain matched/nearGT-unmatched/remote-unmatched groups. Re-match the retained full candidate pool after filtering. Do not report candidate classification recall as event recall.
- Calibration grid0:.001:1, perarm/k. Lock three policies before dev: primary minimum FP subject to eventRecall loss≤5pp (ties greaterRecall then greaterthreshold); auxiliary maximumRecall at FP≤50% original and at FP≤8/min. If no feasible primary, explicit missing. Primary method=Combined,k6, no dev model/k selection. AP/AUC all18, all8devcells, actualFP/density, proposal latency plus decision latency and recall loss. Filterability gate=FP reduction≥50% and recall loss≤5pp; operational gate=FP≤8/min and same recall constraint. Report both, no automatic formalGo.
- Final matching uses immutable proposal timestamps for detection coverage; report decision time=t_p+k*.18 separately. This prevents using observation delay to silently widen/change matching. No future feature beyond that decision time.
- Physical audit remains proxy only: time basis/frame-grid/origin limitation retained, true physical annotations remain missing. Do not change GT or fabricate physical onset. Any proxy signal review is separate from probe labels.
- No official holdout, production code/model writes, new Semantic/Sequential-Verify training or automation. Isolated files under .audit-staging and docs; this directory has noGit metadata, preserve evidence rather than pretend a worktree.

## Tasks

1. Write RED tests for frozen reference/prefix causality, split pairing, rematching duplicate replacement, delayed decision accounting, censoring, fit-only normalization and threshold locks; implement core and watch GREEN.
2. Build immutable capsule, verify source SHA, run frozen GPU shadow extraction with future-perturbation/prefix and recording reset checks; save episode state/key trajectories, feature arrays, metadata, checks, source digests. Test and retain original model hashes.
3. Run18 fixed-budget probes, save0/500 weights+optimizer/log/predictions, calibrate and save all locks before dev. Independently re-read actual steps/predictions, rematch outcomes, render curves/report, SHA archive. One fresh read-only review and one fix pass for material findings.

## Review focus

Emission-vs-peak index; post-emission observation off-by-one; no GT reset; source metadata availability; paired sensor split; full-pool rematching; duplicate negatives; tail censoring; continuously advancing Active limitation; train-only normalization and all locks before dev; AP improvement alone is not a temporal causality proof.
