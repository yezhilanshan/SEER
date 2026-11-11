# Frozen Proposal readout matched A/B and dynamics ablation

> Use executing-plans and test-driven-development for this approved diagnostic experiment. User explicitly approved execution and supplied the experimental design. No original model training or architecture installation is authorized.

**Goal:** Determine whether frozen GRU/Fused states or exported Radar Tokens are better Proposal inputs, and whether causal differences add value beyond state content.

**Architecture:** Nine diagnostic-only readouts consume the already verified seven-layer diagnostic cache; original Encoder/EventFormer/gate/P/Evidence/Proposal weights remain untouched. Three readout positions times three feature ablations, with identical trainable shapes and initialization.

**Tech stack:** Existing seer PyTorch/NumPy server environment; interview conda environment for local pytest. No dependency installation. Isolated audit capsule; no Git checkout is present.

## Locked design

- Sources: md_gru_transition_signal_diagnostic_v2, exactly18fit/8TRAIN-dev recordings, its saved schedule/records/predictions/extraction checks and pinned encoder index/checkpoint. Verify all old archived member SHA, raw/cache source SHA and zero time gaps before reusing frozen features. Record every reused layer file SHA before/after. No holdout/official-test access.
- Inputs: token_mean256 (A), fused128 (primary B), recurrent128 (secondary B). Four-token flattening excluded. All are available at current two-frame update endpoint, as in the previous diagnostic.
- Native current features standardized with fit-only mean/scale on common valid observations. Lossless dimension alignment: pad native vector to256, apply the same seeded orthogonal256x256 rotation, and multiply by sqrt(256/native_dim) to match mean energy. No PCA/truncation or labels in alignment. Save rotation/normalizers and verify isometry/invertibility. Different intrinsic source ranks remain a limitation, but no input coordinates are discarded and no dead projection columns are deliberately added.
- Every trainable readout has the exact same architecture: shared Linear(256,128) on each of four history blocks, then concatenate four128-dimensional outputs, GELU MLP(512,64,1). All9arms have identical initial tensors, nominal parameter count and optimizer. The shared projection is trainable; alignment is fixed.
- Past-only history: current, previous, current-previous, current-mean(previous6). Standardize native current before history; no chunk-local history resets. State uses blocks0/1; Delta uses2/3; Combined usesall4. Mask excluded blocks AFTER projection so projection bias cannot leak into excluded inputs. Ablations intentionally change active input capacity, while retaining nominal trainable shapes/initialization.
- Reuse exactly the previous immutable500x16chunksx64updates fit schedule for all9arms; common first6update exclusion and original valid masks. Weighted Gaussian BCE only (sigma0.36s), positive3.341091295046995/negative0.5879943564065809 from original fit mass weights;16chunks each averaged over its valid points, then equal chunk mean. AdamW500actual steps/arm, lr.001,wd.01,clip1,seed42. No auxiliary peak/rank loss; no hyperparameter search or dev checkpoint selection. 4500total new readout updates,0original updates.
- Checkpoints0/500, actual per-parameter optimizer steps and initial/final tensor digests. All training finite. Save full500step logs/chunk schedule and fit/dev raw scores for every arm.
- Fixed online trigger: peak lookahead1update(.18s),NMS.36s,neighbor-high condition; no trigger changes. Fixed threshold grid0,.01,...,1.0; select onfit only atFP caps3/5/8/10/15/20 perminute with the original deterministic tie-break. Evaluate saveddev scores once for each locked setting; plot these six fit-selected points against ACTUAL devFP. Do not optimize ondev or imply exact devFP matching.
- Report AP/AUROC,positive prevalence,stable scores,Recall,actualFP,Density,latency median/P95,post-boundary recall,all8subject/sensor cells. Re-evaluate frozen existing500 Local/Radar/OR under identical valid context and six fit caps. These oldheads are reference paths, not matched-architecture arms.
- Threshold0 legal-peak coverage is a diagnostic ceiling of this fixed trigger family, not an information-theoretic encoder ceiling. Plateau below80% cannot by itself prove encoder information absence; consider peak conditions and annotation timing. GT is command-transition timing, not verified physical motion onset.
- Primary position contrast: fused__combined vs token_mean__combined; secondary recurrent__combined vs token_mean__combined. Do not choose whichever B wins ondev as a single pre-registered result. State/Delta/Combined reported separately for all positions.
- Lock “clear improvement” atfitcap8: Recall >= +5percentage points, AP drop <=.01absolute, P95latency worsening <=.18s, and >=5of8subject/sensor cells have strictly positive Recall delta. Also flag actualdevFP increase >1/min as calibration-confounded, so nominal passes alone cannot establish equal-FP superiority. Missing comparison fields => not assessed, not pass. Single seed/reusedTRAIN-dev => diagnostic support, no significance or formalGo.
- Dynamic contribution assessed by Combined minus State atsameposition; Delta-only included. Strong claim requires same aboveguardrails; do not call state correlations causal transition dynamics.
- Keep original source/output immutable. Full provenance/archiveSHA/member checks and independent checkpoint/prediction/fit-selection readback required. A failed preflight must launch no training worker; do not duplicate running workers.

## Task1 — matched core

- [ ] Write failing tests for lossless alignment, exact parameter/init equality, feature masks, prefix causality and chunk-equal weighted loss; run RED.
- [ ] Implement .audit-staging/proposal_readout_core.py; run GREEN with prior diagnostic regressions.

## Task2 — capsule and execution

- [ ] Implement isolated worker, source checks, schedule reuse, six-budget causal evaluation and evidence archive; test actual500steps and calibration/source guards.
- [ ] Build immutable capsule and preflight local/server tests; launch only after successful preflight in new output directory.
- [ ] Complete9arms with no original updates; preserve all failures/logs.

## Task3 — independent readback and report

- [ ] Fresh read-only review of whole implementation; resolve material findings before final claims.
- [ ] Independently reconstruct checkpoints/scores and fit selections; verify full archiveSHA/member paths before local extraction.
- [ ] Report all9arms, primary/secondary contrasts, allgroups,guardrails and six-point Recall–FP curves; recommend next branch without automatic encoder training.
