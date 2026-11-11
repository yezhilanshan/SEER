# RD-only X-Fi Transfer Implementation Plan

> For agentic workers: use superpowers:executing-plans inline, TDD and one final fresh read-only reviewer. User approved the existing-flow design in the preceding reply; no repeat approval. No Git metadata: versioned source capsules/results replace worktrees/commits, retain audit ledger.

**Goal:** Implement a safe single-channel X-Fi ResNet18 RD encoder and measure frozen-feature transfer before any LLM training.

**Architecture:** Three frozen frame backbones: R0 existing Stage1 RD-CNN128, R1 randomly initialized single-channel ResNet18-512, R2 X-Fi ResNet18-512 adapted by summing17input-channel weights. Per-window signed-log/z-score same as existing RD path; each backbone produces per-frame features. Train-only coordinate statistics; fresh input projection to256, two-layer BiGRU128/direction, frame14head and ordered8-bin token heads for presence14/count3/ordered3x15PAD. Qwen/oldAlign/projector untouched.

**Spec:** docs/xfi-mmwave-rd-transfer-assessment-2026-10-11.md and preceding approved in-chat design. This is the first frozen-spatial-backbone readout experiment, not an end-to-end comparison or a claim of recovering original17channel semantics.

**Tech Stack:** Existing torch/numpy environments; local interview, server seer. No installations or torchvision dependency.

## Global Constraints

- Same pilot data/split subjects02/03/04 train,05val,06test; original12s/stride6/min8/max3 action windows2187/702/484. Primary700val482test strict12, original supplemental.
- RD only during inference and feature extraction; do not read or concatenate MD values in model input. Input audits may hash paired old MD for identity, but noMD conditioning. Normalize each input window RD only, validframes only. NoGT boundaries used to encode features.
- R0 uses originalStage1 RD CNN weights trained jointly withMD, now onlyRD is input; report this inherited training difference. R1/R2 architecture/initialization recipe and all downstream seeds/budgets identical; onlypretrainedCNNweights andBNstatistics differ. This pair measures frozen transfer under given preprocessing. R0 comparison cannot isolate capacity or original training modality.
- R1random17channelResNet initialized same ResNet recipe then sum-adapt firstconv, matching R2 channel-collapse procedure; no pretend17temporalframe reuse. BatchNorm runningstats frozen/eval, no softmax/fc in backbone.
- New temporal/readout weights seeded17 eacharm; same20epochs AdamWlr.001/wd.01,clip1,batch32,dropout.1. Loss frameCE inversefrequencytrain + presenceBCE posweighttrain + countCE inversefreqtrain + ordered3slotCE includingPAD14, equal task weights. Frame loss validframes. Encoder checkpoint selected valframeMacroF1→valorderedexact→countacc→earliest; also separately report epoch chosen byorderedexact→frameF1→countacc (same training run) so frame selection does not hide language-task difference. Nativefeature widths differ and report parameters.
- Existing P1 fixed HMM candidate grid probability0/.3/.5s × minduration.6/1/2s × priorweight0/.25/.5/1, trainfullrecord prior only; val exact→LCS→Recall selects. Count-constrained segment decoder uses separatecount output trained with same network, min.6/1/2 val selected. Counts predicted, noGTcount input. No newOracle needed.
- Lock allthree model epochs+thresholds onval before testfeature extraction. Select formalarm byvalorderedexact of frame-selected checkpoints→frameMacroF1; reportallarms regardless. Test alreadyobserved prior rounds, overlapping windows/dual sensors correlated, nofreshgeneralization/significance claim.
- Verify originalweightSHA03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56 before/after. Preserve oldweights/results. Featurecaches stayserver. Safeweights_only deserialization only allowlistedlocallycontrolledclasses/torchlayers. NoQwen load/update.

## Tasks / completion contract

- [x] 1. tests/test_rd_only_xfi.py RED thenGREEN: safecheckpointadapt/exactstate/load/frozenBN+noSoftmax, summedconv17repeatidentity, frameCNNwithbatch1, padding invariantBiGRU/pooling, realgradientonlyreadout, invalidlength/nonfinite/shapevalidation, repeatedorderedtargetsexistingP1 tests. Newmodule src/continuous_mmwave/radar_text/rd_transfer.py. Run new+existingradar tests.
- [x] 2. scripts/run_rd_only_xfi_transfer.py: immutableinputs/sourcepreflight, three frozen featurecaches ontrain/val, sharedreadout trainingwithtwo epochselectors, val-onlyHMM/count grids, lock then freshpairedtest features, perframeconfusion/perclass/beds and ordered/count/windowmetrics plusmodelcheckpoint/provenance. NativeGPUtests/smallsynthetic training smoke before realrun. Logstatus/retainfailedversions.
- [x] 3. Archive weights/predictions/allcandidatehistories/source/tests/logs; independentmetrics/selection/hash audit, perclass/matchedpairs, report/figures/docs. Freshfinalreadonlyreview; sealartifacts. If no validation benefit, stopbefore Qwen/fine-tuning and state evidence. If benefit, report this first bounded frozenfeature stage and require a separatelycontrolledfinetune/alignprotocol before claiming end-to-end gains.

## Review Focus

- Pretrained model full pickle customclasses safelyloaded and exact learnedBN buffers; originalfirstconv17sum vsmean, no accidentalSoftmax/classifier.
- RD-only normalization/padding/sequence lengths, BN stays eval throughtraining; cachedframe independent spatialCNN but perwindow normalization prohibitsrecordwisecache shortcuts.
- R0 inheritedtrainingMD confound disclosed; R1vsR2same architecture/readoutseed/budget withtrainonlystatistics.
- Val-onlycheckpointandpostprocessselection, strictwindowmask/frame overlap semantics, predicted count only, repeatedorderedlabels/PAD decoding.
- Hashes/immutablebasefiles, oldobservedtest correlation and no language improvement claimed withoutQwen evaluation.

## Ledger

Preflight: Task1 featurewidth128/512 consumed by Task2 train-onlynormalizer and projection256; nativewidthdiffparameters disclosed. Task2 outputs pairedIDs/logits/ordered/count andval-onlyselection consumed by Task3 independentaudit. Reuse P1decoder only, notP1metrics blindly.

Ruling: proceed with the explicitly recommended frozen-backbone-first stage — original approved assessment calls for this before fine-tuning; R0 reuses Stage1RDbranch as a trained reference — cost: not a fullycontrolledsmallCNNvsResNetend-to-endcapacity experiment. ResNetpair remains controlled for pretraining.

Task1 complete: observed six initial missing-module failures, then implementation tests; added oldTorch compatibility RED AttributeError then GREEN. Local new7tests pass. Server Torch2.4.1/cu121 full radar suite25passed/1skipped (original full-object file local only). GPU smoke two updates each128/512, allheads finite gradients, passed. Safe tensor export exactly matches all original state tensors; original SHA unchanged. Nounsafe pickle fallback.

Task2 started: immutable source139files capsule SHA a8aeef0d028d490b1b0681bccf3768e7737f089e7aaf17134e780cac87597164. Export SHA6ae61614f65e17a965423d5f1f9f51011d1ca67a78d6c70ea74042d7b20f2098. Server native tests/smoke preserved separately. Launch versionradar_text_rd_only_xfi_20261011_v1, retain cacheonserver; source capsule is historical prelaunchplan snapshot.

Task2 complete: server16:06:03–16:20:03, allthree20epochs. Test frameMacroF1 R0/R1/R2=.620137/.622476/.625214; ordered exact156/139/152 of482; predicted-count exact158/151/162. Validation selectors locked before freshtestextraction. Task3 archive/cache+independent audit inprogress; these are preaudit numbers, no modelchange from test observations.

Ruling: no Git repository exists; finishing-branch merge/push menu is inapplicable. Keep isolated historical source capsule and new versioned outputs, no destructive cleanup or publication. Local expanded pytest session stalled during existing tiny-LLM tests and was interrupted; authoritative full radar regression is server25passed/1skipped; local new7tests and CPU smoke passed. No claim that the interrupted local run passed.

Task3 complete: raw31file archive SHAe56e19206ad1ea52520c73e941c78f31e96230254ccd75b0a5ae5842f9be3c9f,152138350bytes. Independent audit48sequence summaries/18confusions plusvalidationselector/grid/protocol checks passed. Server cacheaudit27rawRDwindows/384savedreadouts/9cachesHA/6trainstatistics passed; nativecache staysserver. FormalvalidationarmR0; reportdocs/rd-only-xfi-transfer-2026-10-11.md explicitly states feasiblecandidate, no stable superiority/directreplacement and noQwenimprovement. Figures visually checked, CSV andR1/R2pairedwindows archived. README/guide/assessment nowlink results.

Final fresh read-only review review_rd_only_xfi_final: noCritical/Important issues, checked31filemanifest/currentcorematchessealedsource/originalSHA. Minor ledger: selection.json phrasevalidationstrict12only describeswindow/postprocess scope; framecheckpoint selection actuallyall702windows, alreadyexplicitinreport. Futureversion shouldsplitthose scopes. Packaging/archivingwrite sidecarmanifest beforeexistencecheck: no rerun occurred and archiveverified, futureversion shouldfailbeforeanysidecarwrite. Do not mutate historicalworker/results tofixlabelsaftermeasurement.
