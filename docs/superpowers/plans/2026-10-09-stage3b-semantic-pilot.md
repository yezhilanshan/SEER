# Stage3B Semantic Difference Pilot Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; TDD and a fresh final review.

**Goal:** Test whether frozen compatible P(E) keys support semantic same/different discrimination, with Oracle 0/50 and conditional 50-step causal Proposal replay adaptation.

**Architecture:** Existing SemanticDifference(128,256), no upstream changes. Oracle independently reset EventFormer windows use continuously encoded radar tokens. Proposal replay uses a fixed, causal, historical reference window; it is a diagnostic shadow convention, not a deployed controller.

**Tech Stack:** Python/NumPy/PyTorch, interview pytest locally, seer Python remotely.

**Spec:** User-approved in-chat Stage3B pilot, 2026-10-09; v2.5 semantic supervision sections 13–15/23. No additional training authorization inferred.

## Global constraints

- set000 train (10 streams), set001 calibration (8), set002 TRAIN development (8); paired sensors stay together; no official holdout access.
- Freeze exact positive-coordinate compatibility frontend, Encoder/EventFormer/P/gate/Evidence/physical heads; current compatible bridge is not signed final-v2.5 gate.
- New SemanticDifference seed42 FP32 AdamW lr=.001 wd=.01 clip1, full fixed train-pair cohort; precisely 50 successful updates, save0/50 and optimizer/log. No 200/500 extension, no architecture/trigger/Proposal sampling changes.
- Oracle same means same protocol action class, not event identity. Same/different pairs use distinct, nonoverlapping windows in the same recording; both labels share each reference and candidate maturity. Not identical-key copies.
- Oracle: clean homogeneous frame-label runs, exclude support within .36s of any protocol GT change; fixed 12-update reference, candidate endpoints after 2/3/4/5/6/7 updates. Window bank starts at clean-run start+3, stride12, only complete12-update windows. Both branches use same bank; reference/candidate supports disjoint. For each eligible reference choose one same and one different candidate uniformly/deterministically with seed42; reuse that pair at six ages. Labels balance exactly per record×age, and normalized weights equalize records and ages.
- Oracle uses GT to construct OFFLINE clean windows only. Encoder never GT-reset. EventFormer window resets are Oracle diagnostic construction, not real replay event boundaries. Keys remain frozen.
- Gate evaluated on calibration step50 only: balanced AP>=.80 and AUROC>=.80; complete both classes in all eight cal cells required. Cell/age/action results disclosed, no unregistered cell numerical pass criterion. Dev never decides continuation.
- Reference in causal replay: for emission index j, consume tokens[j-18:j-6] (12 updates), last reference observation j-7; freeze until decision j+6. Candidate consumes j..j+6. Fixed historical lag avoids assigning the continuous mixed Active state a current GT label; has a different reference convention from the previous scalar-probe experiment. Never claim a single-variable comparison to that experiment.
- Replay reference is computed for ALL candidates with valid causal context, WITHOUT GT. Candidate keys reuse exact verified prior replay. Supervision only if BOTH entire reference and seven-update Candidate supports homogeneous nonnegative labels and every support frame >.36s from protocol GT. Label is class inequality. Mixed/near-GT/invalid supports mask. No matched/unmatched training labels. Full-pool predictions never use this offline mask.
- Adapt only if Oracle gate passes AND every train recording has at least one clean same and different replay pair; otherwise preserve all evidence and stop without pretending a viable cohort. Primary model oracle50 if adaptation not allowed, adapted50 otherwise. Loss = independent balanced replay BCE + .2 independent balanced Oracle BCE. One optimizer reset for adaptation, 50 new updates, save0/50; no optional branch selected by dev.
- Baselines: untrained exact same Head/key construction and frozen cosine distance; preserve random-head direction, no post-hoc sign flip. Three evaluation layers; conditional stage is not Verify, no Confirm decisions.
- Thresholds at 0:.001:1 locked on calibration for retain-recall-loss<=5pp, halve-FP, and FP<=8/min before all formal dev inference. Original full-pool matcher ±.72, timestamp unchanged, rematch retained candidates; missing causal windows rejected and full GT denominator retained. Delay1.08s beyond proposal emission; report decision latency separately.
- Report Oracle/replay AP/AUROC, paired-cell/age/class metrics, mask coverage, full-pool event results and recall/FP gates (>=50% FP reduction and <=5pp recall loss). Oracle success cannot install a Verify classifier.
- Physical-onset annotation remains absent: all labels protocol action GT, no physical onset claim. All subjects overlap upstream TRAIN; no independent unseen-subject Go.
- Freeze audit, source SHA, key causality checks, independent weight/optimizer/prediction/matcher readback, full JSON/CSV/NPZ/log/checkpoints archive SHA verified before local extraction.

## Review focus

1. Pair leakage: different cannot systematically change recording/sensor; same cannot copy or overlap supports.
2. Reference purity cannot be inferred from only last frame of a mixed continuous Active trajectory.
3. GT/availability masks never determine online predicted semantic scores or candidate resets.
4. Oracle calibration gate precedes adaptation and all dev readout; no best-checkpoint choice.
5. Event matching after filtering is independent of semantic training labels; report masks and full denominators.

### Task 1: Core labels/pairing/gates and plan
- [ ] Tests fail for absent core, then pass: ambiguous spans, pair source/disjointness/age balance, source-disjoint split, gate ignores dev, loss independent normalization.
- [ ] Record source/schema and replay-reference rulings in progress ledger.

### Task 2: Frozen extraction and training runner
- [ ] Verify pinned source SHAs; build keys, pair manifest, coverage and causal checks; no optimizer updates yet.
- [ ] Save0, run50 Oracle, calibration gate, conditional50 adaptation, calibration locks, then dev readout once.
- [ ] Save all checkpoint/optimizer states, predictions, logs and trained/frozen digests.

### Task 3: Execution, independent verification and report
- [ ] Isolated capsule and single exclusive remote worker; preserve failure archives.
- [ ] Independent readback and source/weight digests; final fresh review and fixes.
- [ ] Download complete evidence, verify SHA/members before extraction; generate Chinese report and notify result. No auto next training.
