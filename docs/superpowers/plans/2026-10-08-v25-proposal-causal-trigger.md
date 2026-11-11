# Stage3A frozen causal-trigger experiment

User-defined bounded research extension of the existing score diagnosis. Execute natively in this session. No deployed controller change, no Head/loss/encoder/gate/P training, no Semantic/Candidate advancement.

## Locked before fit-score inspection

- Calibration: all 18 original role=fit recording views from encoder index SHA205d96c8993c1570c13b0cdb0a708faa7025d094d1b974b5a8ecc5420bbbdc60. These trained the Head; calibration is in-sample, not independent validation. No subject/sensor view splitting of the same recording. Dev remains original8views/704changes, already explored by previous diagnostic; it is regression development evaluation, not a fresh holdout.
- Primary selection uses500checkpoint fit scores ONLY. Identical selected parameters applied to fixed1000checkpoint. Report1000fit/dev achieved FP rates; never silently recalibrate1000 or substitute the winning checkpoint/path.
- Three score paths independently: Local/Radar/OR. No loss correction, OR replacement, or duration features. No fixed refractory main experiment.
- Thresholds0.10..0.95 step0.05. One-shot: persistence2, re-arm immediately on below-high score. Historical fixed0.6 also reported.
- Hysteresis: same persistence2, high thresholds above; low=high-gap, gaps0.10/0.20, low>=0; re-arm after1/2consecutive valid scores strictly below low. Invalid/noncontiguous gap resets all history. Hysteresis cannot re-arm on an uninterrupted high plateau; do not promise it will.
- Peak: lookaheadL1/2updates. At currenti inspect centered window[i-2L,i], centeri-L. Center strictly exceeds every left score and is>=every right score (first point of tied plateau); all2L+1updates valid/contiguous. Center>=high and at least one immediately adjacent score>=high (two adjacent high updates). Emit timestamp=current observed update, not backdated center; store center separately. Unconfirmed tail is not flushed using unobserved future. Added lookahead0.18/0.36s included in latency.
- Online greedy NMS: suppress a newly confirmed peak if its center is<0.18/0.36/0.72s from the last accepted peak center. Never revoke/replace an already emitted peak after seeing a larger future peak. Reset across missing-data gaps. This local duplicate suppression uses neither GT event duration nor future ranks. Not offline global NMS.
- Event matching uses actual emission time, same chronological earliest unmatchedGT within±0.72s, eligibleGTsamevalidmask, exposurevalidupdates×0.18s. Signed median/P95 latency, post-boundaryRecall, hit counts, false/min and density/min all reported. Negative latency alone is not failure.
- For each(path,family,budget) with budget3/5/8false/min: among500fit configurations within budget choose highestRecall, then lowestfalse/min, then lowestdensity, then smallestP95latency, then sorted configurationJSON. No feasible configuration=>NA, never missing-as-zero. Freeze selected JSON/SHA before accessing dev in this experiment.
- Dev one pass of locked selections, all paired500/1000 runs plus historical0.6baseline. Budget means calibration cap, not guaranteed dev cap: print achievedFP and explicitdev_budget_met. Do not call over-budget devRecall 'Recall at fixed3/5/8FP/min'; no dev interpolation or parameter selection. Primary paired1000 control may exceed its calibration budget; flag it too.
- Complete calibration grid and all selected dev metrics retained, grouped byrecord/subject/sensor/cell. Count/prediction timestamps auditable. No posthoc high-recall pass threshold, no formalGo. If results remain weak, propose supervision investigation rather than claim all possible trigger problems eliminated.

## Execution and evidence

1. Freeze config/source hashes in proposal-trigger-spec-v1.json before inference; obtain fit-only saved scores from verified server code/checkpoints under inference_mode, allrequires_gradFalse, tensor digests before/after equal, optimizerupdates0. Keep originalroles and log everyrecord; do not relabel fit as dev.
2. Test causality (prefix invariance), actual emission latency, tied peak/tail, gap reset, hysteresis release, online NMS irreversibility, calibration-only selection andbudget failure. Implement diagnostic-only pureNumPy triggers.
3. Verify fit archiveSHA/member paths and members locally, calibrate on500fit, write exclusive selected.json plusSHA. Only then evaluate selected configurations on existing dev scores; verify original one-shot0.6 reproduces old metrics. Preserve full calibration grids, selected parameters, emittedtimestamps and grouped metrics.
4. Independently replay selected events/matching; render scientific comparisons and Chinese report. Update progress and stop; do not deploy, train or advance closedloop.
