# Split BiRNN Controlled Training Plan

User V2 review authorizes a 200-500 step pilot, then freezing the method protocol and running fivefold Split/BiGRU if the training chain passes. Execute inline; keep one fresh review for the new training/evaluation code. No search for exact CNN counts is required before the declared reconstructed experiment.

## Frozen method decisions

- Experiment name `split_birnn_reconstructed`; existing CNN7283, no architecture change.
- Available23subject/153recording release, nine classes, approximate .09s command clock; Hospital/physical onset unconfirmed.
- Reuse audited seed42 subject folds. Outer test never selects checkpoint or diagnostics. Inner validation subjects are disjoint from train and test.
- Recording-uniform sampling, one uniformly sampled integer length100-1000 per batch8; offsets uniform within each recording. No class resampling. Preload MD/labels once per role; record expected subject/record probabilities and actual sampled class/subject/record counts.
- Adam constantLR.001; explicit constant scheduler persisted. Split loss is forward Combo + corrected Combo, equal weights; baseline one Combo. Global Dice primary, smoothing .8/.025. Aggregate loss by valid frame counts; branch components reported separately, loss magnitudes not compared as equivalent objectives across models.
- Eval mode for streaming/forward equality. BN training statistics span B*T; acknowledge train/eval difference, monitor snapshots and compare cloned CNN train/eval features without modifying actual training BN state.
- Pilot fold0,300steps,val every100. Resume from150 to300 against uninterrupted300; preserve full model/Adam/scheduler/Python/NumPy/TorchCPU/CUDA/sampler RNG and sampling counters, require actual exact comparisons or report statistical-only resume honestly.
- After pilot passes, formal budget max6400steps,batch8,val every128,minimum1280 before patience15 checks; identical budgets and data samples per corresponding fold across models. Seeds42+fold; constantLR scheduler. Best checkpoint by pooled **coverage-matched final weightedF1**; ties lower valid-frame aggregated valCombo.
- Baseline recomputes L512/H171 windows. Split persistent forward. Report coverage-matched classification and full-stream abstention separately, plus forward/first/final views and coverage. Unavailable tails remain explicit.
- Predesignated diagnostics: label shifts +/-3/6/10 with vacated frames ignored; collars +/-3/6/10 around internal CSV command boundaries; extrapolated-tail/non-tail metrics. No alteration of canonical labels or checkpoint selection using these diagnostics.
- Split first/final improve/harm/net fractions over all valid frames and per class, plus boundary/stable masks. Record per-window harmful/improved changes and first/final delays/counts. Delay relative to approximate frame arrival, not physical onset.

## Tasks

1. Tests and implementation: efficient cached recording sampler, full-state checkpoint/resume and training loop; immutable provenance and append-only logs. Watch targeted tests fail then pass in interview.
2. Tests and implementation: batched-window evaluation equivalent to existing replay, coverage-matched/full metrics, label/collar/tail diagnostics, improvement/harm; save all test logits/source/labels/endpoints for independent recomputation.
3. Native tests, source capsule, BN audit and fold0 pilot/resume/checkpoint/prediction replay. One fresh code review; fix material issues with regression tests before formal launch.
4. Freeze full formal configuration and a verified copy of all folds, train both models across five folds sequentially on2080Ti. Monitor owned processes and GPU; no unrelated jobs or environment changes.
5. Independent predictions/metrics/provenance readback, fold mean/sample SD/pooled OOF/per-subject stats, label/correction/latency reports and model artifacts. Return actual results and fidelity limits.

Do not create a new Codex chat/automation. No APK installation or package installation. New outputs/capsules only; never overwrite V1/V2 data or past experiments.
