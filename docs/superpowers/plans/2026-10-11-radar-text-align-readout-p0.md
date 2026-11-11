# P0 frozen Align encoder readout comparison

Spec: user's P0 request; only Align H1/H4 versus existing B3, plus existing Stage1 H1 reference. No full2x2 or Qwen/encoder training.

Ruling: reproduce H1/H4 original_optimizer recipes exactly: same original Stage1 linear weights as initialization, AdamW lr.001/wd.01/clip1/batch4096/seed17,20epochs; H1 raw/unweighted, H4 train-only standardization/inverse-frequency weights with function-preserving initialization. This controls initialization for Stage1 versus Align. These are newly fitted heads, not reused Stage1 fitted heads. If initialization were changed to random, encoder attribution would be confounded.

Ruling: training and checkpoint-selection features freshly extracted in the same nonoverlap12s protocol as original H1/H4 (train subjects02/03/04, selection05, every valid original frame once). Checkpoint chosen validation frame Macro-F1, ties earliest. Evaluation independently freshly encoded on identical caption sliding12s/stride6s windows; never slice nonoverlap features. Strict full12s700val/482test primary, original702/484 supplemental. Reusing training protocol preserves encoder comparison; it does not estimate classification's best possible recipe.

Ruling: reuse sealed existing B3 predictions only after every window identity/GT/input RD+MD+timestamp SHA and Align checkpoint SHA match. No Qwen loaded or updated. Existing Stage1 H1 outputs likewise reused from matched prior run. Saves generation work without changing paired samples.

Ruling: each classifier gets same9candidate validation-only smoothing search, primary individual selection; common-threshold sensitivity across Stage1H1/AlignH1/AlignH4. Freeze JSON before test extraction. Do not choose best method on test. Label durations remain command intervals, not actual execution times.

No Git metadata: new immutable server source capsule and output directory; no branch operations. Original input/cache/model SHA before/after. Features remain server, export heads/logits/paired predictions/provenance/curves/14class recall/Count MAE.

Tasks:
- [x] 1. RED/GREEN P0 recipes, ID/GT/input identity gate, count metrics; implement utilities.
- [x] 2. Implement fresh Align extraction/frozen-head fitting/sliding pairing, native tests and sealed server run.
- [x] 3. Finish result collection, independent sequence/count metric audit, report/14class CSV/plots.
- [x] 4. Fresh read-only review, final hashes and docs links.

Review focus: no encoder gradients; only train statistics; same original initialization/frames/optimizer as Stage1 H1/H4; holdout subjects; checkpoint/smoothing validation only before test; B3 per-window input identity plus model identity; signed over/under count decomposition, not forced GT count; overlapping dual-sensor windows and observed test subject limitations.

Ledger: preflight interfaces checked against head_ablation.fit_head, sliding_baselines.score_windows/choose_thresholds, prior strong-baselines exports. P0 uses new files and imports unchanged prior helpers. No retained memory history; current workspace artifacts are authoritative.

Task1: observed4 missing-module failures then22 related tests passed1warning locally; initial native66passed1warning. v1 source129files SHA2402cac1d93389376a58ccdd001c067463c9ad224b92dd732cfef99bff868db5. v1 startup failed before extraction/training: evaluation holdout check wrongly applied totrain extraction. Added regression RED(1failed4passed), explicit split/train identity gate retains val/test leakage protection. Preserve v1 failed output; run fixed v2 capsule/output. No original model or data changed.

Task2 fixed v2: local23passed1warning, server67passed1warning. Source129files SHA338cf014dba904a41995e27760f9a4406319fc9243d8848d7bacf221c9998f6d,384732bytes. WorkerPID24360 started12:59:41+08:00. Train/valfresh nonoverlap extracted and equality to prior frozenAlign cache verified. H1selectedepoch20 valMacroF1.4974912879; H4selectedepoch8 val.5074172861. Fresh sliding validation inference active. Post-inference report/collector/independent audit live outside immutable training/inference capsule.

Task3 complete13:04:02+08:00: fresh train160328/val53454/test40088 valid nonoverlap frames and702/484 identical sliding windows; strict700/482. All3 classifiers independently/common choose median0/min0.6s, locked13:02:28 before test. FrozenAlign parameters/buffers bitwise unchanged, no gradients;57 original inputs unchangedSHA;129source verified;40newfeatures remain server. Result archive25,666,068bytes SHA53611f80c30d8f13ab2b92fb20e9c35cb03f306a964d99a1c6b31ce297218913;27rawfiles verified including v1 failed status/log and new H1/H4 best/last heads.

Report docs/radar-text-align-readout-p0-2026-10-11.md +560row14class CSV +PNG/SVG rendered. Independent audit passed27rawSHA,10674rebuiltclassificationsequences,11860per-window main-score rows,40aggregate summaries,27validation candidates; CountMAE/signedbias/over-under decomposition equal, previousB3 text/inputSHA andStage1H1 logits unchanged. Final read-only review dispatched per executing-plans, pending; final derived SHA seal after review.

Main results validation-selected AlignH1: strict482 sequence91/482=18.88%, B3 119/482=24.69% gap5.81pp; original484 93/484=19.21% versus121/484=25% gap5.79pp. AlignH4 strict90/482=18.67%. Stage1H1 strict17.84%→AlignH1 18.88%; nonoverlaptestMacroF1 49.28%→51.74%. AlignH1 Recall65.91%, LCS52.25%, CountMAE1.09751, over253/under57; B3 46.06%,45.37%,.33610,over13/under148. Same encoder removes that weight confound but does not isolateLLM causal contribution.

Task4 complete: fresh read-only review found no issue invalidating this run. Two reproduction/reporting findings handled in one final pass: figure title now distinguishes Stage1 reference; generic entrypoint now explicitly checks actual Align SHA against previousB3 provenance across pilot paths. Identity regression RED1failed5passed→GREEN24relatedpassed1warning locally. This preflight-only hardening is post-run source, separate from immutable v2 inference capsule; no retraining or numerical outputs changed. Explicit postrun identity check01837724ac4641a84292dd4ce31ab86498f44f4b3020c82ede481323c7975e9d matches previousB3 and27rawhashes still pass. All current entrypoint/report/audit utilities tracked in derived manifest; original v2 source/results retained. No deferred minors.

Final: Ruling: reviewer declined isolatedLLM/module causality, independent-event significance/cross-subject generalization, and human factuality beyond phrase metrics — these are outside P0 and explicitly limited in report; making those claims would require additional controlled experiments/subjects/manual review. This work makes none of those claims.
