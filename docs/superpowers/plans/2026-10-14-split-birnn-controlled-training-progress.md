# Ledger: 2026-10-14-split-birnn-controlled-training.md

User V2 review read in full and authorizes pilot->formal under the disclosed approximate method protocol. Current SSH72072 alive; GPU35C/270MiB/0%, no other continuous_mmwave process. Existing audited data, model and folds are retained.

Ruling: use reconstructedCNN7283 and available data; do not block training on Hospital/CNN exact source. Treat all results as approximate-label method experiments.

Ruling: validation checkpoint selection is coverage-matched final weightedF1, separating output availability from classifier quality; no test-based settings or shifted-label selection.

Ruling: constantLR scheduler matches paper Adam.001 and is checkpointed explicitly; random snippet RNG and sampling counters must restore, not just Adam/model.

Pre-flight: cached sampler produces[B,L,128]/[B,L] with commonL, original-100 frames retained. Batched evaluation uses same schedule as V2 replay and existing shared head. Diagnostic labels are copies only. Source/data/fold hashes accompany every checkpoint.

New contracts RED8missing-module failures (sampler/resume/BN/fast-replay/diagnostics), then7pass1failure due test's exact floating .8 assertion vs .8000000000000002; numerical expectation corrected to approx, all8pass. Runner end-to-end REDmissing-module, then1pass. Total local related60passed3h5pyskips; nativecapsule36passed. No package/environment changes.

V3 source archive29602bytes SHA25695013e909b688d9c10468cea11ffde16f945524b858b72e16919832067da2246;21source/test hashes verified at /home/xibeidaxue/SEER/.audit-staging/split-birnn-controlled-20261014-v3. New pilot source/provenance differs from prior preparation; V1/V2 intact.

Implementation has loaded-once train/eval pools, recording-uniform sampler counters and RNG state, all-RNG/full-Adam/constant-scheduler checkpoints, cloned BN train/eval diagnostics, equivalent batched windows, original-label full/matched metrics and fixed sensitivity masks. Validation selection matched/final pooledF1 only. 300step fold0 pilot and150->300 exact recovery launched; outertest not evaluated. Fresh read-only review underway before formal run.

Pilot completed:300optimizer steps; resume150->300 exact model/Adam/scheduler/sampler/allRNG/step/extra. Trainloss first30mean5.935565 ->last30mean4.818937; valmatchedFinalF1 at100/200/300=.225810/.330177/.401490; valloss5.485661->5.147450. BN countersall300, finite variance/features, trainstd .9311/evalstd .8642 on fixed sample; groups positive atallsteps. Sampling1340736frames/300batches/103train recordings. No outertest evaluated.

Fresh review found no actionableCritical/Important/Minor; independently checked fast replay713frames L128H47maxlogitdiff7.45e-8 with exact schedule/predictions and actual153recording fold identities. Disclosed approximateCNN/labels remain acceptable method limits. Root verifying savedpilotNPZ metrics against original labels before formal launch. Additional baseline300step pilot verifies its separate loss/head/checkpoint path without changing formal configuration.

Pilot independent readback PASS24validation recordings161757frames, both Split and baseline savedlabel/timestamp equality, sixfull/matched confusion/F1 views reconstructed without production helpers. Baseline300steps loss first30mean2.89946->last30mean2.08116, valF1 .32834/.41718/.45703; its parameters/predictions checkpoint replay exact. Additional baseline pilot does not affect frozen fulltraining starting weights or selection.

G3 six checks computed from actual pilot/fold/log/BN/recovery/independent metrics and assertedallTrue. User-authorized formal driver started in new outputs/split_birnn_fivefold_20261014_v3 with immutable proof/source/data/fold hashes. Two models*five folds; maximum6400steps,validation128,batch8,minimum1280+patience15checks, identical rules/sampler seeds; actual stopping steps may differ and will be reported. No label shifting/collar/tail diagnostic alters training or checkpoint selection. Formal ownerPID recorded by tool.

G3 summary serialization initially encountered NumPybool_ after allsix assertions passed; regenerated with nativebool cast, persistedactual computedchecks. Formal source/data/proof gates passed and ownerPID16674 recorded. No model/protocol changes.

Fold0 Split step3072 valF1.71190, selectedstep2688 at this snapshot. GPU reached88C; applied scoped thermal controller only to verified ownerPID16674 (cmdline must contain this exact run root):stop at84C,resume at79C, poll2seconds, finallyresume if controller exits. Initial cooldown from88C. No global GPU power/fan setting or other process changed; numeric budget/seeds/sampling/scheduler unaffected, wallclock duration includes cooling. Controller/log in capsule evidence.

Thermal follow-up: after a 78->86C rise within two seconds, replaced only the owned controller with v2 stop80/resume75/poll0.5s, retaining owner cmdline verification. Observed lag/overshoot remains; no claim that the threshold prevents all excursions. Both owner16674 and controller8856 absent at final live check; GPU35C/0%/270MiB. Complete thermal logs retained.

2026-10-14 user reports training finished and asks for checks. Live progress statecomplete/completed10 verified: every model/fold6400steps, none early-stopped. Independently audited source/data/folds/checkpoint/NPZ hashes, subject-disjoint roles, each153record outertest once per model, original labels/timestamps, L512H171 endpoints/counts, all full/matched confusions/F1, shift/collar/tail/gains/delays/sampling. Audit PASS. Matched final foldmean±sampleSD Split.80059872±.053346 vs baseline.79303105±.054851; pooled .80273307/.79582015. Split forwardmean.582333, firstmean.754652. Split better3/5folds; no significance claim.

Fresh finished-training-check read all64000train rows and500validation rows: finite losses/positive group gradients, branch sum, LR.001, B8commonL100..1000, exact same sample sequences between models per fold. Reconstructed validation maximum and tie loss matches best selection. Last checkpoints have model/Adam/scheduler/sampler/allRNG/step/extra; Adam steps6400 and moments finite; all1000BN snapshots/clone feature probes finite/positive, finalBNcounts6400. Ten validation prediction directories independently verified. Reloaded all10best weights on CUDA, replayed306test recordings, logits exactly match saved three views with baseline NaNs retained; delays match timestamps and BN unchanged. No optimization/config/label/source mutation.

Post-hoc overlapping-window diagnostic: stable±10 command-boundary mask748653frames, 39451frames had>=1 correct->wrong transition,108had>=2 such transitions; among478031forward-correct stableframes17811had>=2wrong correction emissions,19650wrong final. Saved perframe counts and verified every event improve/harm against original event logs. Original full-tail3901frames correctioncovered1912(49.01%); matched-tail F1 Split.46142/baseline.49663, separating unavailabletail from label/noise concerns. Finalcorrection improves246591/harm43818, netaccuracy+.19627; first/finalmean delay8.8196/37.1704s is future-frame waiting, not physical onset or compute latency.

Archived complete formal/pilot outputs, source, data manifests, scripts and diagnostics in finished-training-artifacts.tar.gz (363846396bytes SHA2567508513ec60b06b9fb7dbdb259d9dea14391c84273d966e7741d5331f491b0c8). Downloaded and safely extracted to outputs/split-birnn-controlled-20261014/exported-server-evidence;1298files515750888bytes individually size/SHA256 verified. Four scientific figures PNG/PDF and docs/split-birnn-training-results-2026-10-14.md generated, inspected and README current status updated. Strict reproduction remainsfalse; no Hospital/clock/originalCNN claim and no test-based retraining.
