# Frozen Short-Horizon Sequential Audit — Results

Read-only analysis on 2026-10-02, Asia/Shanghai. This run replayed frozen representations and fitted diagnostic Logistic Regression; it did not train Stage 2/3 production modules, edit the controller, or evaluate validation/test.

## Completion and integrity

- Chain: smoke 01:23:52 → full extraction 01:29:20 → probes 03:41:47 → complete 04:40:21 CST. Total 3h16m29s; CPU probes approximately 58m34s.
- 24 train recordings, 80,170 natural updates, 4,244 Candidates. Three train subjects, four sessions each, two sensors. The frozen upstream network has already seen these train subjects: this is a probe-only subject-disjoint holdout, not independent generalization testing of the whole model.
- All 22 groups and 4,692 records complete: 3,132 fitted probes, 1,560 fixed-score records; zero undefined records and zero nonconverged fits.
- Fresh `load_sequence_source` verified source code, source checkpoint/config/threshold hashes, all recording artifact hashes, cohort completeness, causal/reference metadata, and trace/controller integrity. Maximum E/z parity errors are both exactly 0.0.
- Source manifest SHA: `5d9439e5b4c8cb533bc0de067faf7d2626307ec4a98c9a50e90e1b23638af440`.
- Downloaded summary SHA: `c0557917ad6ddb1e5ecaee4b1edb42709813949ffe62fc45422dde923fbe1aae`; results SHA: `f1a18008c1f7f96270883d677591f7e0534891bf98314a02915b6028fb8de3a6`. Both match the completed probe manifest.
- Output: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/frozen-short-horizon-sequential-v1/probes`.
- Local copies: `.audit-staging/results-sequential-v1`; reproducible summary analysis: `.audit-staging/analyze-sequential-v1.ps1`.

## Primary task and aggregation

Primary label is whether the current clean GT run differs from the frozen Active reference run, not whether Proposal time was close to an annotated boundary. All layer/source comparisons and history comparisons within a group share endpoint IDs and labels. History 3 is the current and previous two updates, spanning 0.36s; Candidate age 7 is 1.08s after Proposal.

Every primary probe has 130 inputs and 131 coefficients: 128 fixed projected residual dimensions plus cosine/L2. Three preregistered projection seeds; train-fold-only scaling, standardization, class weights. Reported values are subject-macro averages followed by seed median, not pooled or best-seed scores.

### Natural fixed-age AUROC: one update → three updates

| Source | 0.54s, N=3,619 | 0.90s, N=2,312 | 1.08s, N=1,996 |
| --- | ---: | ---: | ---: |
| F | 0.5820 → 0.5847 | 0.6701 → 0.6561 | 0.6885 → 0.6788 |
| E | 0.5600 → 0.5570 | 0.6313 → 0.6126 | 0.6550 → 0.6174 |
| z | 0.5386 → 0.5418 | 0.5727 → 0.5793 | 0.5887 → 0.5919 |
| F+E | 0.5861 → 0.5755 | 0.6630 → 0.6478 | 0.6664 → 0.6634 |
| F+z | 0.5752 → 0.5742 | 0.6564 → 0.6421 | 0.6749 → 0.6675 |

Pooled positive prevalence respectively 14.23%, 21.32%, 25.70%. Different ages have different survivors, label values, and excluded updates; their increase cannot be attributed entirely to representation maturation.

### Actual natural terminal endpoints

`natural-terminal-3-update` means terminal age at least 3, not a fixed age-3 observation. Each Candidate contributes one endpoint immediately before confirm/reject clears it. N=3,442, 712 positives and 2,730 negatives. Subject-macro prevalence/AP constant baseline 0.2066.

| Source | Single AUROC | Three-update AUROC | Single AP | Three-update AP | Single balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 0.6794 | 0.6723 | 0.3490 | 0.3443 | 0.6302 |
| E | 0.6349 | 0.6271 | 0.3162 | 0.3087 | 0.6075 |
| z | 0.5962 | 0.6079 | 0.2806 | 0.2865 | 0.5746 |
| F+E | 0.6667 | 0.6670 | 0.3479 | 0.3505 | 0.6192 |
| F+z | 0.6719 | 0.6659 | 0.3447 | 0.3421 | 0.6277 |

Paired differences are computed within subject/seed before aggregation; their median is not necessarily the difference of the independently aggregated table entries. Terminal history3-minus1 AUROC deltas: F −0.0070, E −0.0052, z +0.0135, F+E −0.0032, F+z −0.0054. z improves in 7/9 subject/seed pairs but remains weak; these are not nine independent subject replications. Single F+E/F+z beat F in only 2/9 and 1/9 pairs, respectively.

## Confounding and generalization checks

- Terminal single-F seed-macro AUROC range is 0.67923–0.68011: not a lucky projection seed.
- Terminal single-F per-subject seed-median AUROC: subject_02 0.6328, subject_03 0.7134, subject_04 0.6938. Subject_02 is clearly harder.
- Single-F same-recording conditional AUROC, pair-support weighted inside each fold and then subject-macro/seed-median: 0.6749, close to ordinary 0.6794. The signal is not explained solely by recording identity.
- Terminal single-F sensor-stratified subject-macro/seed-median AUROC: xWR14xx 0.6849, xWR68xx 0.6724. No evidence in this comparison that xWR68xx alone causes the weakness.
- Terminal single-F training AUROC 0.7434 vs held-out 0.6794. E 0.7517 vs 0.6349; z 0.7012 vs 0.5962. There is a larger probe generalization gap in E/z. This does not diagnose production Stage 2 as overfit or underfit.
- Age-only terminal AUROC 0.5840, AP 0.2449. F exceeds this; age is not the whole explanation, but the age mixture confounds terminal scores. Fixed-age controls remain important.
- At natural 1.08s, F geometry-only AUROC 0.5040 vs vector readout 0.6885; at actual terminal, geometry-only 0.5399 vs vector readout 0.6794. Scalar distances miss useful directional information.

## Low-FPR operating-point sensitivity, no refitting

Read the nine existing terminal single-F NPZ fits. For each fold/seed, set a threshold using only **probe-training negative scores**, the higher empirical quantile at 1−target FPR, moved upward by `np.nextafter`; apply it unchanged to the held-out subject. No held-out cutoff optimization and no new model fitting.

| In-sample probe-training FPR target | Held subject-macro FPR, seed median | Held recall, seed median | Held precision, seed median |
| --- | ---: | ---: | ---: |
| 5% | 7.35% | 15.80% | 40.03% |
| 10% | 12.59% | 27.42% | 39.88% |

For the 5% target, subject_02 held FPR is 14.65–15.83%, whereas subject_03 is 2.86–3.20% and subject_04 3.59–3.80%. The threshold does not transfer uniformly.

Important limitations: training scores here are in-sample, not an independent calibration subset; this sensitivity can be optimistic and is **not calibrated controller performance**. These FPR/precision values refer to update-level GT-run labels at retrospective natural terminal endpoints. They are not interchangeable with the old proposal-level false_confirm_rate, nor with a new controller's closed-loop precision. Terminal timing itself was chosen by the unchanged old controller.

## Late horizon and secondary Proposal task

- Natural decisions unchanged: 4,186 confirms, 49 rejects, 9 recording ends. Of 4,244 Candidates, only 144 reach terminal age at least 8; valid update-labelled late endpoints contain only 134 Candidates and 16 positives. The late natural groups are sparse and censored.
- Eight-update history first exists at age 8 / 1.26s, outside the primary 0.5–1.1s band. Do not pad or claim it as an early online result.
- Forced age13 / 2.16s: F AUROC single/three/eight = 0.7881/0.7831/0.7719. z = 0.7054/0.7019/0.6954. Even late forced replay has no general history-concatenation gain; natural Candidates do not survive to age13.
- Secondary strict Proposal-level terminal task: F single/three AUROC 0.7225/0.7233; E 0.6507/0.6456; z 0.6122/0.6077. This answers a different question and must not replace the semantic same/different-run result. It uses retrospective strict-cohort exclusions.

## Interpretation and next-step recommendation

1. Frozen radar residuals contain moderate, cross-recording linearly readable signal; neither F nor E/z is entirely random.
2. Three-update concatenation and fixed-capacity fusion do not reliably solve early verification. This protocol does **not** prove that nonlinear temporal readers or the full-dimensional representations contain no additional information: projection compresses more dimensions into the same 128 slots when history/sources grow.
3. The existing simple distance Verify inputs are inadequate to read the directional signal, but replacing them with the current linear F probe would still produce poor low-FPR recall and uneven threshold transfer. Do not deploy it or promote a threshold from this analysis.
4. Do not continue warm-up/Fork sweeps or begin broad Event Former retraining on the strength of these metrics. A bounded next pilot should freeze Radar/Event Former, keep the controller unchanged, and compare a regularized raw radar-relational linear readout with one small learned readout on real online Candidate labels. Use candidate-balanced contributions, training-subject folds, and separate train-session calibration; evaluate low-FPR recall, AP and per-subject transfer rather than AUROC alone. Only if this limited readout pilot is useful should a separate validation-only closed-loop comparison be authorized. Failure would justify investigating transition-aware representation supervision, not more scalar-threshold searches.

This recommendation does not authorize or start that pilot. No model, production configuration, threshold, training process, or automation was changed in this result-check turn.
