# P1 Action Evidence Retention & Count Calibration

User spec: P1-A existing P0 AlignH1/B3 sequence error decomposition; P1-B frozen H/Z/P probes for presence/count/ordered actions; P1-C probability smoothing, duration, train transition prior, independent count head, separate oracle diagnostic. No Encoder/Projector/Qwen updates or full2x2.

Protocol: same original sliding12s/stride6s corpus2187train702val484test; strict full12 subset700val482test primary, all original supplemental. train02/03/04,val05,test06; old observed test and correlated sensor/windows explicitly limited. No new LLM load.

Probe specification fixed before results: H full native BiGRU sequence padded to training max length, Z8x256, P8x896; flatten each with train-only coordinate mean/std. Two probes per representation: rank32 factorized linear (no activation) and rank32 GELU nonlinear, same output62 dimensions (14presence,3count,3x15 ordered slots with PAD14). Same20epochs AdamW .001/.01,batch64,seed17; shared multitask BCE+countCE+orderedCE, presence posweight/count inverse frequency fromtrain only. Model selection val full12 ordered exact, then presence macroF1, count accuracy, earliest. Different native input dimensions imply different stem parameters; report parameter counts and do not infer irreversible information removal or compare unlimited representations from a single probe. PAD terminates ordered decoding; repeated labels retained.

P1-C: independent H readout Count MLP32 on mean-of8 temporal bins (not GT boundaries), traincountCE20epochs, val count accuracy/MAE selection. Reuse frozen P0 AlignH1 classification head; newtrainH to derive logits ifneeded. Probability smoothing0/.3/.5sec, minimum predicted duration .6/1/2sec. HMM frame Viterbi with transition weight0/.25/.5/1, train recording frame transition counts with Laplace1, no test-derived priors. Select probability/HMM candidates on val sequenceexact→LCS→Recall. Count-constrained segmental Viterbi uses selected HMM probability/prior config, min segment .6/1/2 selected val; predict1/2/3 from independent count head. OracleGTcount uses same locked decoder, is diagnostic and not a mathematical upper bound or formal performance. No force-GT-count in deployable methods. Selection locked before test feature extraction.

P1-A: unit-cost edit distance ref→prediction, deterministic tie priority match/sub/delete/insert, multipleoptimal-path ambiguity flagged. Count correctness, MAE, conditional exact and paired groups. All29/57 exclusive windows receive symbolic/text audit tables; sampled qualitative review does not claim manual radar/time alignment.

Execution: use executing-plans inline/TDD/final read-only reviewer. NoGit: isolated new source capsule/output; old weights/results/cache hashes unchanged. New feature caches retained server; export probes/predictions/candidates/artifacts. No installs.

Tasks:
- [x] 1. RED/GREEN edit alignment, ordered repeated targets, probability smoothing, count-constrained Viterbi and train-only priors/probe decoding.
- [x] 2. P1-A report/export; frozen train/val H/Z/P extraction,6probes+independentcount training and validation-only calibration.
- [x] 3. Lock selection, extract test, evaluate all paths/probes, collect and independently audit.
- [x] 4. Final report/86window review/plots/CSV, fresh read-only review, immutable manifests and docs.

Review focus: exact H/Z/P nodes and896width; nativeHlengthpadding/trainingstatistics; probe capacity confound; selection onlyvalbeforetest; prior trainrecordings only; independent count head, oracle separated; segment count/duration/traceback noGTleak; tie-dependent edit attribution; all86windows reviewed at symbolic level only.

Ledger: existing P0 source/results verified in workspace; reuse helpers unchanged. Memory registry has no retained task history.

Execution ledger: immutable source133files SHA712593ddd719e1dc9a2841c2062fdadcda45f1a75a9d237f1ee3794451e23e79; server run radar_text_p1_20261011_v1 finished13:57:28+08, selection13:55:52 beforetestextraction. Original66inputs unchanged, Hmax134/Pwidth896, all2187/702/484windows, Encoder+Projector bitwise frozen/no gradients/Qwen not loaded. Sevenreadouts checkpoints exported; fullnativefeatures remainserver. Raw29file archive SHAa07a6276db40ad6f4fca9f87657a59fbf2d546bf0dcd2926d63fe7308fb23be0.

Checks: local30/server74tests pass; independent56summarymetrics+probe/candidate selections+86pairs+archive hashes pass. Nativecache reconstruction H→Z exact, P CPU/GPU max6.68e-6 allclose, trainstatistics exact,8302probeoutputs+1186decodedwindows match. Extra cache-audit helper first failed because repeated NPZ member access allocated whole arrays for each slice; fixed materialize-once, separatev2 auditpassed, rawv1 experiment unchanged and initialhelper preservedserver.

Prespecified strict482results: B1 91/B3 119/probability99/HMM123/predictedcount131/Oracle164; Oracleexcludedselection. H/Z/P nonlinearordered102/122/102; no pooling-collapse claim. Afterprimaryresult countimbalance noticed, added explicitlyposthoc diagnostic fixedtrainmajority3 with alreadylockedcountdecoder; no newtuning. Val/testexact135/116 vs learnedcount169/131; countaccmajoritytest80.91 vshead78.63. No replacementofprimarymethod ornewheldoutclaim.

Final review: fresh read-only review_p1_evidence_count found no actual defects/blockers, reran6newtests successfully, verified29rawfileSHA and coresealedsource. Declined physical radar labels/timegroundtruth, independentremoteextraction, unseen-subjectgeneralization/significance: accepted as explicit evidence limits; root separatelyran remote node audit, no unsupported claims. Report+86symbolic/textrecords+168classcoveragerows+24probe+24calibrationCSV+2plots complete; README/guide updated. Finalartifactmanifest seals current code/docs/derivedproducts separately from immutable rawarchives.
