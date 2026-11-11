# Frozen Short-Horizon Sequential Audit — Implementation and Run Record

2026-10-02. This is frozen train-only diagnostic replay and Logistic Regression, **not a new Stage2/3 training run**. No production model/controller or threshold edits; validation/test unopened.

## Implemented

- Continuous diagnostic ages1–13 with the old sparse profile retained as default.
- Candidate/cohort-local past-only histories, shared endpoints for1/3/8 comparisons; no padding, no natural/forced substitution.
- F/E/z/F+E/F+z; fold-only RMS source scales; fixed128 random projection plus2 geometry inputs,131 Logistic coefficients.
- Three probe-held train subjects, fixed seeds/settings, natural early endpoints4/6/7, actual terminal endpoints, later8-update cohorts separately reported.
- Geometry/age-only controls, session/sensor/age strata, same-recording conditional AUROC, paired deltas and explicit undefined/nonconverged outcomes.
- Automatic read-only smoke overlap gate before the full chain: earlier sparse rows and tensors, actual Candidate decisions and reference identities must match exactly.

## Verification

Local `interview` targeted suite: **68 passed,1 skipped**,28.86s. Remote targeted suite: **69 passed**,7.94s. Local skip is optional cached-Qwen/peft dependency. Complete local pytest was attempted but collection failed at `tests/test_parrad_convert.py` because this environment has no h5py; no installation performed.

Remote full suite: **323 passed,2 failed**,14.45s,1 existing FutureWarning. Failures are the pre-existing `test_gemma3_adapter_matches_native_prefix_and_targets_only_qv` and `test_gemma3_adapter_honors_local_sliding_window`, both `Gemma3RotaryEmbedding.None_inv_freq`; the user previously excluded Gemma repair, and no model code was changed. All diagnostic tests passed within that run. A test-only cross-platform correction selected fitted NPZ via its recorded artifact path rather than relying on directory glob order; transferred file SHA `97f0dcdf029fd712a5e4f5fce2db53e985e387abe95e4e19ec59ca94380eeab4`.

Independent read-only review found an empty-training-fold defect in fixed geometry controls. Regression reproduced it, guard added, tests passed; review confirmed no remaining Critical/Important issues in the core modules and independently executed the corrected edge case.

Deferred review minors: training predictions lack explicit training endpoint IDs (held-out IDs and all selection rules saved); projection cache may repeat matrix generation across blocks. These do not change probe capacity or held-out split boundaries; benchmark preparation before optimizing.

## Deployment and source integrity

SSH through `36373@100.83.206.93` to `xibeidaxue@10.15.22.62`; no password stored in files. Project `/home/xibeidaxue/SEER`; Python `.conda/envs/seer/bin/python`. Initial resource check: RTX2080Ti34°C,0% utilization;59.7GB RAM available,3.0TiB volume free.

Archive `.audit-staging/short-horizon-sequential-v1.tar.gz` SHA-256 `c403cdf5792742f0f1e9bf58d820324a77b13f23b7e1cfd060f8245e64761529`; transferred via compatible legacy SCP after default transport failed. Existing extractor/test backed up inside `.audit-backups/short-horizon-sequential-20261002`; no other project folders altered.

Confirmed remote originals:

- Stage2 `best.pt`: `939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a`.
- Stage3 `best_verify.pt`: `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`.
- Selected threshold artifact: `5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef`.
- Production controller: `b3c37594f78b85c2342fb5de17ff2662f48ed5cfd1adcbfdbce8f68dc8153b44`.
- Event Former: `22d666a8b23f9670fe14140bced3bd7cdd28690bea1fe7e7a1b99c3c34f83ca5`.
- Existing config: `7d2b6b33fcaf3da0c316a1e085c8fe5cbb91e25733835dffcf6491aefe9ee80a`.

## Run contract/status

Script: `.audit-staging/run-short-horizon-sequential-v1.sh`. It runs one-record smoke, verifies overlap against `frozen-layer-decodability-v1`, then full24-record extraction, then sequence probes, stopping on any failure. Delay0.02s; thermal pause82°C/resume below78°C. New output root:

`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/frozen-short-horizon-sequential-v1`

Smoke sibling `frozen-short-horizon-sequential-smoke-v1`; logs sibling `frozen-short-horizon-sequential-v1.logs`.

**Started 2026-10-02 01:23:52 CST:** parent PID8408, smoke extractor PID8417. At parent elapsed54s the log had reached500/3344 updates,33 Proposals, GPU48°C and1491MiB allocated; no traceback or trace mismatch. This is startup progress, **not completed smoke/full diagnostics**. The script advances only after smoke completes and strict overlap verification succeeds. PID and status files: `pipeline.pid`, `chain-status.log`; log `pipeline.log`; smoke gate writes `smoke-acceptance.json`.

Crude replay-only estimate from startup snapshot: full80,170 updates on the order of2–3h, plus smoke/compression and CPU probes. Do not present this as end-to-end ETA: natural/forced workload varies by recording and CPU projection preparation has not been timed on full-dimensional data yet.

**Smoke accepted / full replay started 01:29:20 CST:** one recording took307.87s for3344 natural updates,5516 checked forwards,3506 retained rows. All1222 earlier sparse rows matched exactly (max_absolute_error0); E/z controller errors0, decisions/references unchanged. Full24-record extraction now running under parent8408. Smoke timing extrapolates to roughly2.05h replay for80,170 updates before per-record workload variation; retain a2–3h replay estimate and leave CPU-probe ETA unspecified until preparation is observed.

## Implementation rulings

- No Git repository: retain backups/hashes and durable docs ledger; skip Git-dependent scripts/cleanup. The scratch ledger directory was unavailable, so use docs without changing ACLs. Cost: no commit-based rollback.
- Float32 block GEMM equivalence tolerance2e-6, controller/reference parity1e-5 unchanged. Cost: sub-micro feature rounding accepted.
- Cache unscaled source projection contributions and geometry moments; fold-only scaling reconstructed algebraically and dense-equivalence tested. Cost: float32 summation rounding, no additional learned parameters.
- Add a strict automatic smoke→full gate. Cost: benign numerical drift aborts instead of proceeding without parity evidence.
- Use the fitted artifact recorded in results rather than directory enumeration order. Cost: none; test now checks exactly131 coefficients across filesystems.

Two deferred minors remain as described above. No implementation decisions authorize later representation or Verify training.
