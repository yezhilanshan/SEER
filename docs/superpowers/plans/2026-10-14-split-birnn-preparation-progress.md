# Ledger: 2026-10-14-split-birnn-preparation.md

User review authorizes implementation preparation, not formal training. Original spec plus the pasted review are the binding brief.

Ruling: Execute inline; use package/capsule isolation because no Git checkout exists. No shared existing model/data mutations or dependencies are needed.

Ruling: User specifically requests official README/source verification in the review. Read the public official download metadata under that new authorization; never access private accounts or transmit server credentials to third parties.

Ruling: Fixed 0.09-second clocks are an explicit approximation. An audit can verify arithmetic/coverage, but cannot establish physical synchronization or true movement onset from command CSV alone.

Pre-flight: data arrays `[T,128]`, 9-class labels and -100 feed model/probe; model cache `[B,T,64]` feeds stream corrector; timestamps are frame arrival timestamps, not physical action labels.

2026-10-14: review read in full. Local interview Torch2.11.0 CPU/pytest9.0.3/NumPy1.26.4 confirmed; h5py unavailable locally, use native server for real H5 tests. Old elevated SSH tool session cannot be reused under new policy; fresh SSH session72072 authenticated, GPU35C/270MiB/0%.

Task1/data RED: 7 missing-module failures observed; GREEN 7 passed. Source mapping, per-frame normalization, nonzero origin/ignored prefix, off-range commands, transitions and grouped folds exercised.

Task2/model/loss RED: 8 missing-module failures; GREEN 15 tests including data. Cached and joint backward agree, every GRU receives backward-only loss gradient, whole/frame/irregular chunks agree in eval; exact .8/.025 smoothing and literal global Dice verified. CNN7283 vs inferred8301 mismatch remains unresolved; GRU4896/6432 and shared head153 closed. No dummy parameter fitting.

Task3/stream RED: 5 missing-module failures; GREEN20 including prior groups. Independent L5/H2 window reference verifies first/final/count/delay, no hidden flush and reset. Standard BiGRU uncovered frames unavailable, not manufactured causal predictions; fixed-denominator metric counts abstentions.

Task4/probe sampler RED: 1 missing-module failure; GREEN21passed1skip(h5py). Relevant original data/split regression total48passed1skip. Entire local pytest collection fails tests/test_parrad_convert.py because interview lacks h5py; no dependency installed. Ruff unavailable locally. Real HDF5 test delegated to native runtime, not declared locally passing.

Sourcev1 archive16704bytes SHA2566c7996b290cae39ffad097cb8a1e6eef9eaad5439c52046fbe28ef00decbfec0 uploaded to new capsule, hash and safe archive members verified; native22passed. Capsule path /home/xibeidaxue/SEER/.audit-staging/split-birnn-preparation-20261014-v1. Full existing project native regressions not claimed.

Immutable audit completed in canonical/split_birnn_preparation_20261014_v1:23subjects77sessions153sensor recordings1033155frames6939CSV rows. Source/saved MD hashes and shapes equal; labels valid under approximate origin, no zero-transition recordings and no commands past recording ends. Ignoredframes0 is induced by first-command-origin assumption, not evidence of exact alignment. Candidate folds retain subject grouping. Hospital/actual radar clocks/physical onset remain unknown. Server lacks matplotlib; export8train-source01/02 diagnostic NPZ windows for local rendering rather than install.

Official public page HTML obtained; both public listing endpoints return401, browser surface initialization times out. README not obtained; no private authentication attempted. User review's request is source-verification authorization, not authority to bypass authentication.

One fresh whole-change reviewer dispatched per executing-plans/requesting-code-review instructions; read-only, no nested agents. Native throughput preparation probe launched with owned PID, no formal training/checkpoint.

Fresh review:3Important/2Minor, noCritical. All5 proved RED natively against immutableV1; local3failed2skipped. Single fix pass adds origin-aware diagnostic times; native precision/source_md_dtype; rejects fully unlabeled probe before BN/Adam mutation; separates negative CSV coverage from positive final-label extrapolation; rejects final H5 with noMD. Local GREEN51passed3skipped including original data/split tests. General all-ignored differentiable loss retained.

Independent data audit PASS153records1033155frames:all actual source MD exactly equal saved values, CSV/NPZ hashes, unchanged H5 stats, independent interval-fill labels, grouped folds and every record outer-test once. Total valid transitions13247; longest last-command extrapolation20.226724s. Source dtype separately confirmed float32*153; none of review fixes require rewriting existingV1 data.

V1 CUDA probe:batch8 at100/512/1000 Split31.32/43.39/60.07ms, baseline26.41/34.77/55.06ms, peak allocations .47/2.32/4.52GiB. Real source01 two512frame samples passed one Adam step, finite gradients, no checkpoint. V1 metadata random_weights_only was imprecise after that real step; V2 explicitly records random initialization/no checkpoint and the single-step state for latency. Prefer V2 timings after fix.

Task1: software/data arithmetic preparation complete; Hospital/original-clock evidence unresolved, not certified. Task2: computing-contract preparation complete, originalCNN closure unresolved. Task3: complete on reconstructed model and independent window reference. Task4: preparation checks complete, formal-budget/protocol freezing not complete, formal training remains disabled.

V2 fresh capsule /home/xibeidaxue/SEER/.audit-staging/split-birnn-preparation-20261014-v2:15source-file hashes verified;17981byte archive SHA2562f3067781137e015fdf8e75da76449c96a923730cb669b063eb415978892e7cd. Native27passed. CUDA real MD64frames whole/frame/chunk max difference5.96e-8; cached/joint backward0; corrected-only gradients reach CNN/bothGRU groups.

V2 throughput:Split100/512/1000frames32.372/41.956/60.183ms, baseline26.172/34.933/55.670ms; peak allocatedmax4.520GiB.42synthetic+1real probe steps per version (V1+V2=84synthetic+2real); no checkpoint, no formal fitting or model F1 evaluation.

Export691835bytes SHA2563af0e5d5ad6bb60cbd3b4d8e3d62fcb8f8ca547f9b0dc6fd94c6dedcd854250c. Initial local assertion had one duplicated trailing hash character from terminal wrap; re-read server sha256sum and local Get-FileHash, corrected expected literal, then all44export hashes verified. No data/archive edits needed. Eight plots rendered locally and two representative windows visually inspected. Full MD NPZ remain server-side.

Budget candidate6400steps/128step validation/patience15after1280minimum steps proposed in new config, not frozen or executable approval. Final report docs/split-birnn-preparation-results-2026-10-14.md and README updated. Preparation results do not imply originalCNN/Hospital/numerical reproduction or permission to start formal training.
