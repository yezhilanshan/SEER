# Execution ledger: Transition-aware Semantic Projection Pilot

Spec: `docs/stage3-transition-aware-projection-remedies-2026-10-02.md`.
User authorized implementation and training on 2026-10-03.

## Fixed scope

- Only existing SemanticProjection query/MLP trainable; disposable 131-parameter Logistic heads.
- Three independent outer folds; seeds1701/1702/1703;20 epochs maximum.
- Projection fit set000/001;dev set002;final probe fit000–002;calibration003.
- Reuse actual closed-loop TRAIN traces;no validation/test;no production promotion.
- Stage2 semantic rehearsal covers stream/commit/text;Candidate weights sum to1.

## Rulings

- Local checkout has no Git metadata. Add isolated new modules/tests/config;retain this ledger and file hashes instead of creating a Git worktree or commits.
- The user's instruction to train approves the supplied remedies and proposed budget/gates;do not re-request design approval.
- Use existing isolated server SEER environment;local pytest in interview;no dependency installation.
- First sandbox Python launch did not return;escalated interpreter verification succeeded (torch2.11.0+cpu,numpy1.26.4,pytest9.0.3).
- SSH via36373@100.83.206.93 to xibeidaxue@10.15.22.62 timed out during banner exchange. Asked user asynchronously to confirm connectivity;continue local work.

## Tasks

- [x] Core training contracts:TDD split isolation,weights,differentiable probe features,semantic retention,Projection-only optimization.
- [x] Source validation and role-gated rehearsal extraction.
- [x] Trainer,dev selection,checkpoint/provenance and final-probe integration.
- [x] Local targeted/regression tests and synthetic training/resume checks;full-suite limitations reported.
- [x] Server source/resource preflight,upload,new-directory real-data smoke.
- [x] Launch fixed-budget training and verify live process/log/smoke checkpoint; formal epochs and final evaluation continue in the detached pipeline.

Current state: original formal pilot stopped at epoch6; authorized bounded recovery completed normally. All three diagnostic conditions selected epoch0; no expansion or held evaluation. See recovery results below.

## Implementation and independent review

- Added four `parrad_projection_*.py` modules and four test files. Existing production modules are unchanged.
- Independent review found three issues: semantic cache bytes were not fully bound to resume provenance; undefined supporting endpoint metrics unnecessarily blocked the decision; interrupted cache generation left an unusable final directory.
- Added four failing reproductions and fixed all three. Reviewer rechecked the changes and reported no further important findings.
- Final deployment archive is `.audit-staging/transition-aware-projection-v2.tar.gz`, SHA256 `4f71d509eccaebe54eaa306743fd6f1987fb9e8d56ee3dfa4f6c9220977d9a46`. The earlier v1 archive must not be deployed.
- The launcher is `.audit-staging/run-transition-aware-projection-v1.sh`; it checks fresh output paths, runs and validates one real-data smoke epoch, then runs the fixed training budget and locked-checkpoint final evaluation.

## Validation and connectivity evidence

- Previous targeted verification: 19 passed. Latest full local regression in the specified `interview` environment: **373 passed, 2 skipped, 1 warning**, 128.26 seconds; log `.audit-staging/projection-local-full-tests-v2.log`.
- Local full regression excludes only `tests/test_parrad_convert.py`, whose import requires unavailable `h5py`. No dependencies were installed.
- During the earlier authenticated server session, verified the completed 24-recording source, original checkpoint/config, idle RTX2080Ti, available RAM/disk, and absence of competing training jobs. Those resource readings require refreshing before launch.
- The earlier v1 archive uploaded successfully but was not extracted. The latest v2 upload was interrupted; remote presence and SHA must be checked before retrying it.
- On 2026-10-03 around 10:10 +0800, renewed SSH attempts timed out at jump host `100.83.206.93:22`, before school-server authentication. Local Tailscale reported `starting` / `NoState`; its Windows service was running.
- The user supplied server authentication interactively. Credentials are not stored in project files or logs. Asked the user to restore the Tailscale/jump connection, not to supply the password again.
- Deployment, real-data smoke, and full training remain pending connectivity restoration. Do not report this experiment as running.

## Connectivity restored and server deployment (2026-10-03)

- User restored Tailscale. SSH login succeeded; latest v2 archive was absent remotely, then uploaded and SHA256 matched the value above.
- Extracted with `tar --keep-old-files`; exit 0. No pre-existing production files were overwritten.
- Server targeted tests: **19 passed in 4.44s**. Server full tests: **376 passed, 2 failed**, 35.15s. The failures are the pre-existing Gemma3 adapter prefix/sliding-window tests (`None_inv_freq`), outside this pilot's Qwen path. Logs are in remote `.audit-staging/projection-*-tests-v2.log`.
- Refreshed resources: RTX2080Ti 33C, 0% utilization, 270/11264MiB; 58GiB available RAM and 3.0TiB free disk; no competing pilot/training process.
- Started detached launcher PID **29779**. Real-data smoke is running first; formal training is conditional on the smoke's gradient, parameter-change and frozen-teacher acceptance checks.
- Remote launcher log: `/home/xibeidaxue/SEER/.audit-staging/transition-aware-projection-v1-launch.log`.
- Remote pipeline logs: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-projection-pilot-v1.logs/`.

## Live smoke acceptance and formal launch evidence

- Real-data smoke completed and passed execution acceptance: mean Projection gradient norm **2.8856593399513057**, maximum parameter change **0.006179438903927803**; frozen teacher and frozen head-only Projection remained unchanged.
- Smoke epoch1 dev score was **0.6390364736453708**, but semantic retention was ineligible. The selection lock correctly retained **epoch0**. Smoke acceptance confirms working optimization and isolation; it does not establish useful semantic improvement.
- Epoch0 cached-key reproduction maximum error was **2.2351741790771484e-07**, below the locked tolerance.
- Chain status records formal training start **2026-10-03T10:25:07+08:00**. Live launcher PID **29779**, formal runner PID **29891**.
- Formal manifest confirms `smoke=false`, epochs20, seeds1701/1702/1703, fixed loss weights and optimizer settings, and `validation_test_read=false`, `production_model_changed=false`.
- Last live check: formal runner alive, preparing subject02 fit rehearsal cache (65/434 windows). Full formal optimizer epochs and final held evaluation have not completed.
- Training checkpoints/output: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-projection-pilot-v1/`.
- After all nine training configurations lock their selected checkpoint, the same detached pipeline automatically runs the preregistered final evaluation; it does not promote weights to production.

## Progress inspection at 2026-10-03 11:11 +0800

- Pipeline stopped at **10:32:24 +0800** (`exit=1`, launcher line40). No training/evaluation process remains; GPU is idle (34C,0%,270MiB).
- Active configuration was **subject02 held / seed1701**, the first of nine configurations. Epochs1–5 completed; epoch6 stopped in `semantic_evaluation` -> `_action_probe` with `RuntimeError: action probe did not converge`.
- No configuration has a final selection lock (0/9); final evaluation has not started. No final scientific conclusion is available.
- Saved `last.pt` confirms epoch5, requested20, optimizer state and RNG state present. Epoch00–05 Projection checkpoints exist. The failed epoch6 update was not saved; any recovery should restart from the epoch5 optimizer/RNG checkpoint.
- Dev selection score: epoch0 **0.6258478965738886**; best so far epoch3 **0.6411790820727241**; epoch5 **0.6343442218150646**. All adapted epochs1–5 were semantically ineligible; higher dev AUROC alone does not meet the locked acceptance rule.
- Inspected action macro-F1 old -> epoch5: subject03 stream .61910 -> .62427, commit .57481 -> .58191; subject04 stream .70215 -> .70051, commit .67027 -> .68675. The reported semantic ineligibility must not be described simply as action-F1 collapse.
- This inspection reads logs/checkpoints only; it does not change the locked solver/gates, overwrite failed evidence, or automatically bypass the convergence requirement.

## Authorized bounded recovery implementation

- User authorized the recovery plan with “那请你开始吧”. Protocol: `docs/stage3-transition-aware-projection-recovery-2026-10-03.md` (fixed subject02/seed1701, rehearsal-only1 epoch, joint lr1e-5 3 epochs, conditional anchor10 joint3 epochs; all initialized at epoch0).
- Added failure diagnostics without altering action probe capacity, regularization, FP64 solver, 400-iteration budget or acceptance tolerances. Nonconverged dev action probe returns undefined and excludes the checkpoint; baseline failure remains fatal; required final semantic undefined produces inconclusive.
- Optimizer/RNG state is now atomically saved before dev evaluation, with pending-evaluation state. Resume re-evaluates the saved epoch without repeating optimizer updates.
- Added separate loss gradient norms/cosines for the first fit-only batch of each epoch.
- TDD: five original failure reproductions, two isolated replay/gating tests, and one review-found preflight failure test. Final targeted suite: **27 passed in22.01s**. Full local regression before the final preflight-only fix: **380 passed,2 skipped**,162.11s; same known missing-h5py conversion-file exclusion.
- Verified retained v1 archive's original pilot and runner against current code on two optimizer epochs: **model, all AdamW state, NumPy RNG and Torch RNG bit-exact**. Log `.audit-staging/projection-recovery-v1-parity.log`.
- Independent reviewer confirmed the preflight fix and retained-v1 comparison, then reported **复核通过，未发现新的重要问题**.
- Recovery archive `.audit-staging/projection-recovery-v1.tar.gz` SHA256 **880dd17b3d80236114c10d5e37903db99f919a5f1c2a5609c8e8cc4c9d1edf64**, uploaded and remote SHA matched.
- Server code is isolated at `/home/xibeidaxue/SEER/.audit-staging/projection-recovery-code-v1/`; original server `src` and v1 output remain untouched. Server tests are running before launch.

## Recovery deployment and launch

- Server targeted tests: **27 passed in8.78s**. The isolated full test initially lacked original `scripts`; copied original scripts/configs into the isolated checkout, then full regression produced **384 passed,2 known Gemma3 failures**,55.46s. No library installation or Gemma changes.
- Started detached recovery launcher PID **30888**, preserving original v1 code/output. New output: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-projection-recovery-v1/`.
- New logs: the same path with `.logs/`, `recovery.log` and `status.log`. The process will stop after bounded diagnostics; full pilot expansion remains conditional on retained semantics and independent dev improvement.

## Recovery completed

- Started **11:54:38**, completed normally **11:58:36 +0800**; status `diagnostics_complete`, no recovery process remains.
- Original epoch6 commit probe failure reproduced: finite inputs/objective,400 iterations exhausted, gradient max1.61456e-6 and objective change1.97603e-9. Dev undefined handling now continues correctly without lowering standards.
- Rehearsal-only1 epoch: best devAUROC .63677871, worst retrieval delta -21.43 percentage points; selected epoch0.
- Joint lr1e-5 3epochs: best devAUROC .62878757, worst retrieval delta -15.18 points, some semantic probe undefined; selected epoch0.
- Joint lr1e-5 anchor10 3epochs: best devAUROC .62931636, worst retrieval delta -5.36 points; selected epoch0.
- All conditions fail the original semantic tolerance. No full pilot expansion, no held/calibration numeric reading, no production promotion.
- First actual batch Projection gradient norms: transition0, semantic1.87229943, anchor3.96e-9. Rehearsal-only failure shows damage is not attributable solely to transition loss.
- Detailed local report: `docs/stage3-transition-aware-projection-recovery-results-2026-10-03.md`.
- Remote result archive SHA256: `015ef719f220da9a8e42baa8355d3fe1a810e219f3babf033f6a15980d3fd499`. Fetching raw JSON/logs into `.audit-staging/results-projection-recovery-v1/`.

## Authorized rehearsal fidelity/objective audit

- User authorized the next investigation after the explanation of semantic degradation. Audit is read-only: no optimizer steps, no production/experimental weight changes, only the existing subject02 fold's fit/dev inputs; calibration, held numerical inputs and validation/test remain sealed.
- Confirmed original Stage2 checkpoint epoch14 settings: batch_size4; auxiliary CE weight0.5; alignment1; consistency0.1; temperature0.07; label-run64 frames, minimum8, boundary-ignore1second. Current rehearsal uses balanced batch32 and omits auxiliary CE, while caption LM has no Projection dependency. These are objective/sampling differences, not yet a proven exclusive cause.
- Diagnostic script `.audit-staging/audit-projection-rehearsal-v1.py`, SHA256 `437fe858b65e03e8b691ba1611edfb244e3b1773c3a493b1684db94152bc77a0`; syntax checked and server SHA matched.
- Started audit PID31273 using the already isolated recovery code. It validates all cached row/window/action/template mappings and dependencies; recomputes42 text hiddens and stratified real event windows; tests mixed-length batch masks; measures eight batches each for batch4-uniform, batch32-uniform and batch32-balanced with/without the original frozen auxiliary classifier term. No optimization is performed.
- Audit output: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-rehearsal-audit-v1/`. Log `/home/xibeidaxue/SEER/.audit-staging/rehearsal-audit-v1.log`.

## Rehearsal audit initial results and paired follow-up

- Initial read-only audit completed normally. All648 cached windows correspond to the original clean-label window enumeration, correct metadata and action/text indices; every class has all3 templates represented. Config model/LLM differences are empty. All42 text hidden states reproduce exactly; stratified21fit+4dev event replays pass cached hidden equality.
- Cache retains434/898 possible fit windows and214/456 dev windows within the allowed role subset. Fit class-distribution total variation is .174294 versus full allowed fit windows; balanced sampling changes it further. No held raw inputs were opened.
- Original auxiliary CE weight.5 is missing from the adaptation semantic objective, but on eight batch32-balanced diagnostic draws, adding it changes current-vs-original gradient cosine only to mean .998493. This omission alone is not demonstrated as the primary cause.
- Mixed-length batch event keys differ by approximately7.55e-5–2.62e-4 from single-window keys; batch4-vs42 text keys differ by~9.8e-5. Full hidden differences are larger. Do not declare a padding implementation bug before separating compute precision from batch-shape effects.
- Follow-up script `.audit-staging/audit-projection-rehearsal-followup-v1.py`, SHA256 `c28b1972686deccba88bd4307400553838aefae0b976f6fc7558f06f08f619ed`; compiled and remote SHA matched. Started PID31436.
- Follow-up compares identical32samples as one contrastive pool versus8microbatches of4, isolates original epoch14 caption rotation on the same events, and compares single/mixed inference using native checkpoint dtypes versus full FP32. Still zero optimizer steps; no original weight files changed.

## Rehearsal investigation completed

- Paired follow-up completed normally. On exactly the same32 examples, batch32 vs average of8batch4 gradients have cosine mean **.682587**, range **.304163–.948004** across8 draws. This isolates a material contrastive-objective change caused by negative-pool size; it is not proof that this alone explains all dev degradation.
- Changing only static vs original local batch4 epoch14 template rotation gives gradient cosine mean **.983961**. Restoring original frozen auxiliary CE on batch32 gives mean **.998493**; neither is demonstrated as the exclusive cause.
- Actual native adapter dtype is **FP16**. Converting computation toFP32 reduces mixed/single stream,commit,text key differences to **1.04e-7,1.64e-7,2.27e-7**. Observed small batch discrepancies are consistent with precision effects, not a demonstrated padding-mask implementation error.
- Both scripts ran **0 optimizer steps**, with no checkpoint file changes; no held/calibration or validation/test numerical data accessed.
- Project understanding correction: matching the alignment formula/temperature does not match the original objective when contrastive pool, sample distribution, template rotation and auxiliary losses change. Current label-based rehearsal is renewed adaptation, not a guaranteed old-space preservation constraint.
- Report: `docs/stage3-transition-aware-rehearsal-audit-2026-10-03.md`. Original JSON evidence is being fetched to `.audit-staging/results-rehearsal-audit-v1/`.
- Proposed next research step: teacher relationship/retrieval distillation that is stationary atold weights, or a separately declared original-style rehearsal control. Neither was launched or used to alter the locked protocol during this investigation.

## Authorized teacher-relation small experiment

- User approved the concrete three-condition plan with “那请你先做小实验吧”. Preregistered protocol: `docs/stage3-transition-aware-relational-pilot-2026-10-03.md`.
- Added independent `parrad_projection_relational.py`, without modifying original pilot/runner/upstream. Fixed subject_02/seed1701; shared1epoch head-only warmup, then head-only/relation1/relation10 each3epochs, Projection lr1e-5/decay0. Full fixed434fit-event/42text bank each update; no original label alignment. Teacher relation MSE covers new/new and both cross-version pairings; consistency preserves teacher cosine rather than forcing equality.
- Seven meaningful tests cover zero gradient at teacher equality, common rotation compatibility, unequal query-chunk invariance, fit-only bank boundary, warmup freezing/task gradient, identical three-condition forks and pending-evaluation resume parity.
- Red: first five tests fail because the new module did not exist. Green: **34 relevant tests passed** locally10.21s and server9.94s. Local complete pytest collection blocked by existing missingh5py in `test_parrad_convert.py`; remainder **388 passed,2 skipped**55.20s. Server complete regression **391 passed,2 known Gemma3 None_inv_freq failures**40.18s, specifically `test_gemma3_adapter_matches_native_prefix_and_targets_only_qv` and `test_gemma3_adapter_honors_local_sliding_window`. No dependencies installed.
- Experiment archive SHA256 `57334720e5b1cb3ee650cfc6a596f4e362d538a3cb727bcb2fd10268a24f4c0d`; remote SHA matched. Isolated checkout `/home/xibeidaxue/SEER/.audit-staging/projection-relational-code-v1`; shared evaluator/pilot hashes match local and prior isolated recovery code.
- Read-only code reviewer dispatched by requesting-code-review skill; real training awaits review. New output reserved: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-relational-pilot-v1/`.
- Reviewer found two protocol safeguards to complete: reject resume before completed warmup rather than overwriting evidence; explicitly verify actual fit task gradient after warmup. Two new tests reproduced both gaps, then passed after the fixes. Cross-condition recommendation now breaks score ties in favor of smallerlambda and never promotes automatically. Final reviewer: passed, no remaining important issues.
- Final archive SHA256 `e3092febe3315b74c39f010a823d00e36f0899b88e3dba200ded1a58b01b78a0`, remote matched. Final related tests **36 passed** local35.14s/server10.13s. Final local available regression **390 passed,2 skipped**164.59s (h5py conversion collection limitation unchanged); final server complete regression **393 passed,2 known Gemma3 failures**42.16s. Original production files unchanged.
- Started detached launcher PID32239/worker32243 at **2026-10-03 13:19:10 +0800**. Real fit-bank preflight: relation0,consistency0,maxProjectiongradient0. Shared warmup check: Projection/teacher bit-exact unchanged, taskgradient norm**0.10872630029916763** on128positive-weight fit rows. No new optimizer steps were used for these checks.
- Completed normally **13:22:07 +0800** (177s). All three conditions selectedepoch0; no expansion. Bothjoint conditions show no negative dev retrieval delta in any adapted epoch, but no conversion dev improvement. relation1 bestcompleteAUROC .62461836 versus baseline .62584790; relation10 bestcomplete .61633344 (bestany .62481698 not eligible). The incomplete checkpoints have stream actionprobe undefined from solver nonconvergence, not demonstrated numerical semantic decline.
- Firstjoint Projection gradients: transition .14097221195697784, relation/consistency/anchor0; actual820optimizersteps including shared82stepwarmup and3conditions×3epochs×82steps. Teacher unchanged, head-onlyProjection unchanged, jointProjection actually changed.
- Final source/checkpoint integrity verified, launcher/worker exited. Raw result archive downloaded and SHA verified: `8923402e9e9d5c20cd2ffdb24d9925c2fde94148e5681d15533f02f4e89dfbc7`. OriginalJSON/logs and generatedsummary under`.audit-staging/results-relational-pilot-v1/`.
- Final report `docs/stage3-transition-aware-relational-results-2026-10-03.md`. Updated understanding: this bounded run preserves measured retrieval but does not improve independent conversionprobe. The next question is training objective versus selection metric alignment, not further unsystematic semantic-weight increases. No follow-up experiment launched.

## Authorized transition objective audit

- User authorized “请你开始检查吧”. Read-only Projection audit on saved warmup/head-only/relation1/relation10 checkpoints; no Projection optimizer steps or production changes. Standalone `.audit-staging/audit-transition-objective-v1.py`, syntax checked; SHA256 `cf57e5b8f31cd2de50128ec92dce6d5603cb36fc8f01bbc3e27d968d96c31d50`.
- Compares actual temporary-head scores on exactly the primary endpoint/age/subject aggregation; replays saved independentprobe states and verifies endpoint/target correspondence and probability/aggregate equality.
- Fits ephemeral diagnostic linear heads on fit labels only, fixed130-input capacity,400iteration budget/tolerances; ladder changes ages,endpoint subset,weights,age head partition and conditional class balance, holding original train normalization fixed. Saved independentprobe replay supplies the final per-age normalization comparison. Separate ridge0 continuation uses original full weightedBCE and unchanged head capacity to measure underoptimization; no diagnostic is promoted or used to select a Projection.
- Measures full-fit Projection task gradients by age using frozen saved warmup heads, then diagnostic-refitted shared heads. No Projection step is applied. Only fit/dev numerical rows admitted; held/calibration andvalidation/test remain sealed. Manifest/checkpoint/code/source hashes verified before/after; original experiment files untouched.
- Initial audit completed normally. Saved independentprobe probabilities reproduce with **max error0** for allthree conditions; primary aggregation exact. Actual temporary head-only primaryAUROC .54094, compared with original independent .62585.
- Converged fixed-normalization diagnostic ladder on oldProjection: allageCandidate .57505 -> age6/7Candidate .62352 -> sameprimary endpoints .62352 -> classbalanceduniform .63240 -> separateageheads withsameglobalbalance .62870 -> peragebalance .62867. Age selection is a large paired difference; endpoint eligibility changes no rows; independent perage normalization replay .62585. Diagnostic scores are not new model results.
- Fit4804/dev2072 rows,1299Candidates; primaryfit1414/dev640. Ages3–5 carry76.33% of finalCandidate mass, ages6/7 only23.67%. Final within-age weighted positive fractions12.26%,16.60%,19.65%,22.88%,24.85%, not50%; intermediate agebalance thenCandidate renormalization does not guarantee classbalanced finalweights.
- Initial warmup fullfit gradients all-vs-primary cosine .76360, early-vs-primary .57116 (not opposed). Ridge0 diagnostic continuation at400iterations did not satisfyconvergence, so its negativeearly/primary cosine is not used as a claim about converged training. Supplementary audit measures actual savedhead checkpoints and a genuinely converged ridge diagnostic; scriptSHA `68406c628230ead63baf05fdb043ab0ed398cce51e411ae0457c1e52effcee8d`.
- Both audits complete normally (initial13:52:14–13:54:15; followup14:01:35–14:02:47). Actualhead-only/joint1/joint10 final gradients allvsprimary cos.77464/.64931/.67197; earlyvsprimary.58666/.42839/.45891. Convergedridgehead givesearlyvsprimary-.37787, allvsprimary.53203; shows conditional ageobjective compromise, not continuously opposed gradients atactual training states.
- Savedhead-only fullweightedfitBCE .60112, constant fitprior17.35% BCE .46141; ridge0 continuation reducesBCE .39066 but fails400iteration convergence, so itsdevscore is diagnostic only. Ridge.001 sharedhead converges86–88iterations withfitobjective .4213–.4217. Warmup1epoch is not evidence of adequateheadfitting.
- No normalization overflow observed:std-floorcount0, inputp99abs3.97–4.05, max8.16–9.59. No endpoint eligibility discrepancy inprimaryages. Old/new frozen snapshotsreplay probabilitymaxerror0.
- Checkpoints/Projection unchanged; both Projectionoptimizer counts0; no held/calibration/validation/test numeric access. RawJSON/logs fetched and archiveSHA verified `029f33bb73a02377c5367b7052ba49bea3a509a2afc9133289951e45161621ee`.
- Report `docs/stage3-transition-objective-audit-2026-10-03.md`. Next recommendation: align primary conversionloss toage6/7, declare finalweightmass/classbalance explicitly, verify fit-only headpretraining beforeunlockingProjection. No correctiveProjection training launched inthisaudit.

## Authorized primary-age objective correction

- User authorized “请你开始”. Preregistered independent `docs/stage3-transition-aware-primary-pilot-2026-10-03.md`, implemented new `parrad_projection_primary.py`; original pilot/runner/relational and upstream unchanged.
- Fixed subject_02 seed1701; natural probe-eligible age6/7 transition rows; each age/class bucket total1/4 without Candidate renormalization. Frozen previous all-fit teacher normalization retained for isolation. FP64 shared3heads ridge.001 LBFGS400 with unchanged tolerance/convergence rules before adaptation and after every Projection step.
- Frozen control versus relation1/relation10, each at most20 full-primary-fit Projection updates lr1e-5, decay0, clip1; original semantic bank relation/consistency/anchor and original evaluation/retention gate unchanged. Milestones0/5/10/20. No automatic full-pilot expansion.
- Checkpoint phases after Projection update: head_fit_pending, evaluation_pending, ready; resuming completes pending work without repeating updates. Nonconverged head refit stops condition and locks baseline. Initial numerical convergence and actual nonzero fit task Projection gradient required.
- Tests red on missing module, green7 passed35.74s: bucket balance and cohort boundaries, real solver convergence/cap, global loss chunk invariance, frozen warmup/nonzero task gradient, two-phase interruption exact model/AdamW parity, missing warmup refusal, failed refit stopping. Related regression42 passed97.70s before final failure-test addition. Full available local regression running; read-only reviewer running.
- Code archive SHA256 `9533f4124a8dab03b936be6cbe0029fd453a9e9983285c9d57b23800c1147cae`, uploading to `.audit-staging/projection-primary-v1.tar.gz`. New isolated server checkout reserved `.audit-staging/projection-primary-code-v1`; real Projection training awaits completed review and server checks.
- Server v1 targeted43 passed16.13s; complete regression400 passed and two existing Gemma3 failures (`None_inv_freq`),49.66s. Local available regression396 passed2 skipped256.40s (collected before final seventh new test; that test passed separately).
- Reviewer found two P2 issues before real training: missing LBFGS prev_loss on valid zero-iteration convergence creates nonserializable infinity; nonfinite solver exception failed to stop-and-lock baseline per condition. Two new tests reproduced both failures. Fix preserves gradient convergence rule, records missing change asNone, distinguishes numerical head errors from resumable interruptions and routes numerical failure to baseline lock. Final new tests9 passed48.05s. Final broad regression and reviewer follow-up pending.
- Fixed archive SHA256 `068b25ffad7752b53312df81ed32fdc274164add6cc29d7ca2f37567df217a2e`; deploying only to the new isolated checkout. No real Projection steps applied yet.
- Follow-up reviewer passed both fixes, no remaining important issues. Final local regression399 passed2 skipped271.92s; final server402 passed and the same two known Gemma3 `None_inv_freq` failures50.54s, final server new tests9 passed8.46s. No dependencies changed. Uploaded archive SHA matched; primary module SHA256 `25e19ff3a531e0293a585c54f7071fab4d2320ed1ee5621065ffc788b5b0a158` matches local/server.
- Started detached launcher PID5641 in `.audit-staging/projection-primary-code-v1`, output `transition-aware-semantic-primary-pilot-v1`. Fixed budget unchanged. Original production runner/pilot hashes measured before launch `7a665f0d6868eb57e2de5f6c49045c8aaeb9cac395ad29575d2809446eaa97d5` / `2cb4d27263a3d8dbb1af5d27938fc58afae5b30a08196e9dc0e794779050b614`; these pristine production versions differ from the already-reviewed isolated diagnostic runner/pilot. Only isolated versions are used for this diagnostic.
- Completed normally20:30:54–20:33:30(+0800),156s; worker5644/launcher5641 exited. Actual primary weights1414rows/761Candidates; age6 negative588positive173,age7 negative478positive175, four masses.25 each. Shared warm heads102/104/110iterations, averageobjective.570698 vs balancedconstant.693147; P/teacher unchanged and actual fullfitPtaskgradient.7694166 beforeunlock.
- All40per-step head refits (120linear solves) converged under original numerical rules, no solver-budget changes. Actual P AdamW states each20, frozencontrol0, phasesready. Teacher fixed and jointPchanged. Warmup convergence used loss-change branch, not strict gradient <=1e-7 or exact optimum.
- Bothjoint conditions selectedstep20 and passed complete original semantic gate. OriginaldevAUROC.6258478966 -> relation1.6498085127 (+2.3961pp), relation10.6506791393 (+2.4831pp). Earlyrelation1steps5/10 andrelation10step5 ineligible only because400iteration streamactionprobe notconverged, requiredF1None. Retrieval accuracy worstdelta0 for bothselected; margin worstdelta-.00008932/-.00004214 andconsistencyincrease.00020602/.00017108, allwithin originalgate. No fullpilot/promotions/extra steps.
- Source andStage2checkpoint finalchecks passed; originalproductionrunner/pilot hashesunchanged. RawJSON/logarchive downloaded andverifiedSHA `f8534a58cb5523f3ac4dae70178fbbcc86cdfd0eff7534afe823fd5da3dfb579`, `.audit-staging/results-projection-primary-v1/`; standalone evidence summary rechecks all selected gate metrics andrefit convergence. Report `docs/stage3-transition-aware-primary-results-2026-10-03.md`.
- Updated understanding: combined primary-age/weight/head-solver/fullbatch corrections produce exploratory dev gain while retaining measured old/new retrieval. No isolated attribution to a single correction; onefold/one trainingseed, no independenttest evidence. Suggest preregistered seed/fold repeats before expanding; not launched.
- Final independent checkpoint readback verified selected SHA256 matches lock, selected P equals final P, every teacher tensor equals common frozen teacher, common warm P equals teacher, frozencontrol P unchanged and bothjoint P changed. All final phasesready and ps shows neitherlauncher5641 norworker5644 remains. Rawmanifest held/calibration/validation/test numeric-read flagsFalse; source/checkpoint/spec hashes match report. Execution complete.

## Authorized primary recipe replication

- User authorized “请你继续下一步”. Fixed newfoldheldsubject_03/seed3407; same originalStage2 start, coefficients1/10, each20fullbatchPstepsmax andmilestones0/5/10/20. Preregistered `docs/stage3-transition-aware-primary-replication-2026-10-03.md`. Fit/devsubject02/04, held03/set003/val/test numeric excluded for thisfold. No original artifacts overwritten.
- Ruling: current correctedrecipe is deterministic (fixedP,zeroheadinit,fullbatch,no dropout, fixed featureseeds). Changingtrainingseed doesnot provide independentstochastictrial; validated actual syntheticfullmodelstates+scores at1701/3407exactlyequal. Record equivalence, focus substantive stabilitycheck on newfold instead of adding randomness or counting identical trials. Folds overlap andStage2upstream sawTRAINsubjects; no independence/unseen-subjectclaim.
- Minimal primary.run parameterization exposesheld/seed/spec/cache_parent while retainingdefaults02/1701 and unchangedobjective/optimizer/evaluator. Actualsettings now record actualfold/seed correctly. Testsred on missingparameters,green12 tests61.68s; allrelated48 tests77.94s. Read-only reviewer running before realPtraining.
- Server authenticated; GPU0 RTX2080Ti idle270MiB/11264MiB. No activeProjectionworkers. Existingoriginalparent has onlysubject02 caches, so newfit/devcaches forheld03 must use originalfrozenprepare_rehearsal cap4/template/window settings. Newisolatedcode `.audit-staging/projection-primary-replication-code-v1`, newoutput `transition-aware-semantic-primary-replication-v1`; codearchiveSHA `7a2129b29213674a5f33fc2ddee599f03b52ee8c4a99a0c41be4f2e818c8d1c9`. No realreplicationPsteps yet.
- Reviewer found P2 resume held/seed mismatch could overwrite completed manifest before warm identity rejection. Added two actual-entry tests (held/seed), watched failures, fixed pre-write comparison of previous provenance AND actual settings; prior manifest bytes remain unchanged. Newreplication tests5passed4.58s, includingactualseed-equivalence optimization. Cache acceptance additionallychecks cap4/14classes/42templates/source/config hashes. Final reviewer follow-up andbroad regression pending.
- Network interrupted before successfuldeployment: initialpassword authentication eventuallysucceeded andinventoryreadabove, butSCP timedout thenSSHreset. Final Tailscaleping100.83.206.93 threetimes timeout andBatchModesshjumphost ConnectTimeout15 timeout. No newfoldworkerlaunched, noStage2oroldexperiment changed. Asked user toconfirmjumpcomputeronline/notasleep/Tailscaleconnected; continuingunaffectedlocalverification. Deploymentmustrestartwithhashverifiedv2archive, freshisolateddirectory andservertests.
- Follow-up review passed recovery/cap/hash fixes, no remainingimportantissues. Added direct seed-equality assertions for actualAdamWstates aswell as allmodeltensors/scores; final5replicationtestspassed4.27s. Finalavailablelocalregression404passed2skipped80.23s; h5pycollectionlimitation unchanged, no installs. LatestshortTailscaleping stilltimeout/no reply. No realnewfoldtrainingclaim; ready-to-deploy finalv3archive supersedesv1/v2. Progressreport `docs/stage3-transition-aware-primary-replication-progress-2026-10-03.md`; usernetworkquestionpending.
- Finalv3archive SHA256 `fb49017f73469eca56365340b44a3354d5083b808d548391de6743d9ffda4506`, primarymodule SHA256 `bd1a0a9dbf76ae520c7dffd12952a79458aa694c2718185c1a156fa6d5fb802f`. Networkrestore required beforedeployment/tests/cacheprep/realnewfoldrun. No backgroundworker orautomation created.
- User supplied exactSSH command andrequestedretry; successfullyauthenticated target viajump. GPU0 freshsnapshotidle270MiB/11264MiB, noProjectionworkers, replicationoutputabsent. SCP finalv3finishedexit0; remotearchive/module hashes match lockedlocalvalues. Originalproductionrunner/pilot/Stage2checkpoint hashes unchanged. Newisolatedcheckoutprepared; servertargetedtests running beforecacheprep/realtraining.
- Finalservertargeted50passed18.14s. Startednewfoldpipeline22:31:57(+0800),launcher7014/cacheworker7017,cuda0. Frozenfitcache446windows, observed289/446done, GPU1511MiB; nooriginalartifacts overwritten. Newoutput`transition-aware-semantic-primary-replication-v1`, trainingroot`subject_03-seed-3407`. Cachedeps/rolechecks followedbylocked20stepconditions; noalgorithm/budgetchange.
- Subsequentmonitorconnection stoppedreturningoutput thenreset; a fresh exactSSH retrytimedoutconnectingjump100.83.206.93. Backendpipelinealreadydetached; do NOT relaunch or infer completion/failure withoutreadingserverstatus. Lastverifiedphase frozenfitcache289/446, no verifiednewProjectionupdates/results yet. Needrestoreconnectivityandinspectstatus/caches/trainmanifest/ps first.
- FreshTailscalestatus reportsjumppeer explicitoffline,lastseen1mago; shortpingno reply. Askeduserto keepjumpmachine/Tailscaleonline forresultreadback. Progressreportupdatedtomark actualbackgroundlaunch andunverifiedlaterstate; do not treattrainingascomplete. Pipeline7014launched22:31:57, lastcacheworker7017. No duplicateworker/automaticfollowupcreated.
- Userrequestedchecktrainingresults. FreshSSHauthenticated; directstatuslogcompleted22:38:58 afterstarted22:31:57 (421s),cachefinished22:35:49. Bothconditions20stepscompleted, newfoldbaseline.5629059617571361 -> relation1.5671820566263357 (+.4276pp),relation10.5676813881601614 (+.4775pp); bothselected20. All0/5/10/20semanticmetricscompleteandgatespassed. Fixed20stepgain agreeswithselectedgain; relation10step5slightlybelowbaseline.
- Independentserverreadbackassertionspassed: commonwarmP==teacher,finalteachersunchanged,allfinalphasesready,actualAdamWsteps20/20,all40refitsconverged3heads,selectedSHAverifiedandoriginalretentiongaterechecked. Newcachesfit446/dev216 roles+cap4/14classes+artifactsverified; source/code/spec/Stage2hashesverified. Productionrunner/pilothashes unchanged. Savedserver`result-readback-checks-v1.json`.
- AttemptedJSON/logarchiveexcludingpt/npz; SSHstoppedreturningoutputandSCPjumpconnecttimedout. Noarchivedigestreceived,noverifiedlocalrawdownload. Do NOT run summaryscriptuntilrawarchivecompleted/hashverified. Savedclearlylabelledlive-stdouttranscription `.audit-staging/projection-replication-live-readback-v1.json`, report `docs/stage3-transition-aware-primary-replication-results-2026-10-03.md`. Exactretrievaldeltasnotyetavailablelocally; reportgatepassonly,noclaimallretrievalunchanged. No retraining/checkpointreplacement/newfoldlaunchduringcheck.
- Understanding: bothcoefficientsimproveinbothobservedfolds,butnewfoldgain~.48ppversusfirst~2.48pp; gainmagnitudeisfold-dependent. Overlappingfolds/deterministicseed/upstreamTRAINexposureprecludeindependenttrial/unseen-subjectclaims. Suggestremainingfold04afterrawresultfetch,budgetunchanged; notlaunched.
