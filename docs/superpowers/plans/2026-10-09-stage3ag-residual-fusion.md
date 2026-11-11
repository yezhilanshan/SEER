# Stage3A-G frozen GRU × weighted500 Radar residual fusion

> Execute inline with executing-plans and test-driven-development. The user's complete C/A/B specification and direct instruction authorize this bounded experimental spike; no production installation or encoder training.

**Goal:** Test complementary GRU-state information against both a Radar-logit-only capacity control and the existing frozen Radar baseline.

**Architecture:** Shared Linear128→64 on previous/current raw recurrent state, concat with original Radar logit (129dims), Linear129→64/GELU/Linear64→1; final layer exactlyzero; add residual to original logit. No explicitdelta/past6,Local,OR,attention,newtemporalnetwork,newgate.

**Tech stack:** Existing seer server PyTorch/NumPy; local interview pytest. No new dependencies. Diagnostic-only files/capsules; originalsrc/weights immutable; no Git checkout.

## Locked constraints and provenance

- C has0updates; A zeroGRU inputs/B realGRU inputs have identical6parameter tensors/initialization and each500actualAdamW steps. 16,641trainableparameters each. All originalmodules have0updates. FP32,seed42,lr.001,wd.01,clip1.
- Reuse exact500×16×64 fit schedule and common first6update exclusion from proposal_readout_matched_ablation_v2. No new sampler. h is raw128-dimensional recurrent state, without new normalization; previoush comes from precedingglobal update even across chunk boundaries. Do not pad fake history into computation: no-history positions bypass residual entirely.
- Raw zR comes from one continuous GPU replay of the pinned weighted500 Radar Head on pinned26cached Radar streams; no inverse sigmoid/clipping. Reset only atrecording start. Check sigmoid(zR) against saved existing500__radar scores (maxerror<=1e-7; record exactness), C reference metrics against priorrun, and byte-identical original stage tensors. Verify cached h endpoint/update timestamp alignment using prior26stream cache crosschecks/layerSHA, pinned encoder checkpoint/index/rawcacheSHA.
- Source weighted500 checkpoint /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v25_physical_proposal_loss_ab_v1/weighted/training/physical-step500.pt SHAde06a21a0a12816ec82e0fbc71e972c415916ac172bf009867fb67b9923bd89e. Source oldmodule state preserved asbyte-identical; no production checkpoint writes.
- Original Stage3A weightedGaussianBCE (sigma.36): weights3.341091295046995/.5879943564065809; eachchunk loss=sum(weightedcomponentloss)/sum(w_pos*y+w_neg*(1-y)), then equalmean of16chunks. This is the original V25Stage3._bce implementation. The prior diagnostic readout used valid-point normalization; it is not silently substituted for the user's explicitly requested originalloss. Same originalloss for A/B; no auxiliary objective.
- Zero-update phase BEFORE any actual-data optimizer construction: reconstructzR, preserve zero-update matrices/CSV, exact A/B residual-lastlayerzero and max|final-zR|<1e-6 for26streams, identicalinitial tensors; prefixcausality/recordreset/chunk continuity unitchecks; encoderdigest/oldHead byte checks. All first no-history updates residual0.
- First compute Radar×GRUstate hitoverlap on savedscores at eachpath's previousfit8threshold and atzero-threshold legalpeaks, separately fit/dev, subject×sensor, actiontransition class-ID pairs and failuretypes. CommoneligibleGTonly; excludedexplicit. Oracleunion is boundaryhit-set union, not attainable fusion performance. This read-only diagnostic cannot change hyperparameters, fit thresholds or pre-registered arms.
- Gradient norms beforeclip perprojection/inputMLP/output layer and individualweights atsteps1/2/5/100/500. Step1projection/inputMLP gradients expected0 due outputzero; forB require finite nonzero frontgradient bystep2; Aprojectionweight expectedalways0 (zeroinputs), not an error. Aprojectionbias may learn. Same nominalparamcount, activecapacity differs intentionally peruser control.
- Save checkpoint0/100/500 andfulloptimizerstate, full500step logs forA/B, schedules/sourceSHA. No devcheckpointselection. Fit thresholds lock before formal C/A/B dev evaluation.
- Trigger untouched: peaklag1=.18s,NMS.36s,originalneighbor-high; thresholdgrid0,.01,...,1,fitcaps3/5/8/10/15/20. Sixfit-selecteddev points plottedat ACTUALdevFP; no devsweep/modelselection. Devformal scoredonce aftertraining; all otherreadbacks reuse savedscores. The requested zero-update study necessarily reads previouslysaveddevpredictions beforetraining; it cannot inform modelchanges.
- Record C/A/B overall andall8subject×sensor AP/AUROC/stablemean,Recall/FP/Density/latency median&P95/Post-boundaryRecall/zeropeakcoverage. Preserve legacyfailureclassify ordering for below_threshold/neighbor_constraint/peak_shape/nms_suppressed/invalid_context/match_conflict/excluded/matched; no renamedfailuresemantics.
- Rescue/Harm perGT atfit8 plus allsixbudgets. Forrescue, save Czero acceptedpeakpresence andCzero/budgetfailure. Mutuallyexclusivegroups: old_zero_peak_shape,old_zero_no_legal_peak_other,old_has_peak_threshold_filtered,other. Firsttwo sum to “oldzero had no legalpeak”; peak_shape is a subset, not extra double-countedrescue.
- Save everyBzero-threshold legalpeak CSV:peakcenter/emissiontime,zR,delta,final,nearestGTdistance,nearGT±.72,subject/sensor,classprevious/current/next. Truepeak proxy uses peakCENTER distance; also save emission-timeGTflag and whether uniquelymatched. Offline futurelabels are metadata only, never modelinputs. No claim everynearGTpeak is atrue-positiveuniqueevent.
- Success atfit8 requires B-C Recall>=.05 and B-A>=.05; AP_B-AP_C>=-.01; actualFP_B-FP_C<=1/min; P95_B-P95_C<=.18s;>=6/8cells Recall improve andworstcelldelta>=-.05; atfit3/5 B-CRecall>=-.02; rescue>harm. Missingmetrics => notassessed/No-Go. C reference must reproduce56.108%Recall/7.6687FP within numericaltolerance. “Go” means nextdevelopment experiment only, never independenttest/formalGo/automaticdeployment.
- Artifacts completeJSON/CSV/NPZ/checkpoints/optimizer/logs/SHAmanifest, safechecksum/member verifieddownload, independentCPUparameter/predictionreadback and localfit-selection/event/guardrailrecompute. Data remain upstream-seen repeatedTRAIN18fit/8dev; no official/sealedholdouts.

## Task1 — pure analysis and residual core

- [ ] Write RED tests for zero-init equivalence,capacitycontrol isolation,history fallback/prefix/recordreset/chunkcontinuity,gradflow andoriginalweightedloss.
- [ ] Implement isolated core and hitmatrix/rescue/peakmetadata helpers; verifyGREEN andpreviousmatched-readout regressions.
- [ ] Run saved-prediction0update matrices without deciding model/threshold changes.

## Task2 — frozen interfaces and two arms

- [ ] Implement pinnedfrozenHead replay/sourcechecks, fullzeroacceptance, immutablecapsule andnativepreflight.
- [ ] Freshread-only review as required by executing-plans; fix materialfindings withRED/GREEN.
- [ ] Launchsingleisolatedworker onlyafter preflight; finishA/B500steps; fitlock thenformaldevonce; archiveallresults.

## Task3 — delivery

- [ ] IndependentCPUcheckpoint/optimizer/predictionreadback andallsourcefreezeSHA.
- [ ] VerifyarchiveSHA/members,downloadfull evidence,localmetric/selection/guardrailchecks.
- [ ] CompleteMarkdownreport+Recall–FPfigure,allmatrices/failures/rescue/peakstats,Go/No-Gowithlimitations. No subsequenttraining orinstallation.
