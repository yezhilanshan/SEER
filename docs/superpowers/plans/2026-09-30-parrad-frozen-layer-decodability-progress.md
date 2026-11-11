# Execution ledger — 2026-09-30-parrad-frozen-layer-decodability.md

Spec: `../specs/2026-09-30-parrad-frozen-layer-decodability-design.md`.

- Task 1: trace callback implemented on actual forward. Uncached RED→GREEN; cached test requires server peft. Local regression 21 passed/2 skipped before later changes.
- Task 2: reference binder and strict clean-update labels implemented. E/z identity tolerance 1e-5 and no assumed previous index. Tests cover ignored frames/gaps, mixed update, multiple boundaries and stale reference state ID.
- Task 3: train-only natural/forced recording replay implemented; actual Active object ownership plus E/z are asserted. Output folders refuse overwrite. Natural/forced traces share weights and inputs; no controller code changes. Overlapping forced Candidates are tested.
- Task 4: fixed random projection and CPU LBFGS implemented; all primary layers have 130 features/131 parameters. Raw sensitivity restricted to age 13 to bound RAM/CPU, labeled secondary. Current-only proposal probe is also secondary.
- Tasks 1–4 latest local targeted verification: 16 passed, 1 skipped (optional cached Qwen dependency).
- Task 5: remote cached-path tests and original one-recording smoke passed; revised smoke and full extraction/evaluation remain pending as of the entry below.

Ruling: no git exists in this workspace, so generic skill commit/worktree scripts are inapplicable. Preserve this ledger and source snapshots for reproducibility.

Ruling: source code is instrumented through a temporary diagnostic instance wrapper; production controller source, weights, and decisions are unchanged. Wrapper restoration is tested.

Ruling: CPU LBFGS and fixed lambda replace a sklearn dependency; settings were declared before any held-out scores. No dependency installation is necessary.

Local whole-suite run before Tasks 3/4: full collection blocked by absent h5py; excluding test_parrad_convert.py yielded 261 passed, 2 skipped. Server full suite pending.

## Review fixes and remote verification — 2026-10-01

The requested fresh code review found three important diagnostic gaps, all addressed before the formal run:

- Retain complete current/reference `alpha`, exact `E_prev`, and `E_hat-E_prev` tensors, not just gate summary statistics. NPZ round-trip tests verify the recurrent interpolation.
- Snapshot Candidate and Shadow immediately before the diagnostic runtime clears them; verify both discarded and promoted branches at confirm/reject. The production controller source remains unchanged. Mismatch-injection tests cover both terminal decisions and wrapper restoration.
- Validate the exact unique subject_02/03/04 × set000..003 × TI_xWR14xx/TI_xWR68xx cohort using train sidecars before loading recording arrays. Smoke loads only its selected train recording. Probe evaluation rechecks the complete matrix. Tests refuse missing, duplicate and unexpected cohort entries before array loading.

Latest local targeted verification: **24 passed, 1 skipped** (cached Qwen requires server peft). Remote targeted verification including cached Qwen and Stage P3 training regression: **44 passed**. Remote full suite: **279 passed, 2 failed**, the two previously known `test_gemma3_adapter` failures (`matches_native_prefix_and_targets_only_qv`, `honors_local_sliding_window`), outside this Qwen diagnostic. Log: `.diagnostic-backups/frozen-layer-audit-v1/full-pytest-v2.log`.

Original smoke v1: `subject_02__set000__TI_xWR14xx`, 3344 natural updates, 181 proposals, 1222 retained rows, 5516 natural/forced checked steps, max E error **0.0**, max z error **0.0**, 216.78 seconds. It predates the gate/terminal/cohort fixes and is retained, not reused for the formal run.

Revised code schema: 2. Revised smoke uses a new `frozen-layer-decodability-smoke-v2` directory. Deployment archives v1/v2 and original Event Former/runner snapshots are retained. Remote diffs confirmed only the approved trace tap and optional pre-load recording filter in shared runtime files. Working SSH route: ProxyJump via `36373@100.83.206.93`; no credentials are recorded here.

Revised smoke v2 completed and was accepted by a read-only artifact checker: 3344 natural updates, 181 proposals, 1222 retained rows, 5516 checked steps, max E/z error **0.0/0.0**, 218.29 seconds. Every layer, full gate, previous-state and gate-delta array is finite and row-aligned, gate values in [0,1], all layer dimensions correct; artifact and source checkpoint/threshold hashes match. Natural rows, Candidate metadata and decisions have the same hashes as original smoke v1; the added gate arrays changed only the tensor artifact. Revised smoke directory is retained.

Formal launch script chains extraction then probes with shell `set -euo pipefail`, one CPU thread, 0.02 s per-update pause and GPU temperature pause at 82°C until below 78°C. No validation/test data, weights, thresholds, or formal controller policy are modified.

Formal chain started 2026-10-01, parent PID **2895**. Output root suffix: `frozen-layer-decodability-v1`, sibling `.log` and `.pid`. Full probe metrics and interpretation are still pending; launch is not completion of Task 5.

## Full audit and interpretation completed — 2026-10-01

The formal chain finished at 20:53:51 CST. Extraction status complete (24/24 records), probe status complete (all six layers), source manifest SHA matched, train-only and test disallowed. No log exceptions or E/z mismatches. Retained 27,985 rows from 4,244 Candidates.

Task 5 numeric analysis is now recorded in `docs/stage3-frozen-layer-decodability-2026-10-01.md`. It compares subject/seed-paired layer differences, sensor/session strata, age/task/cohort eligibility, geometry baselines and convergence. Main finding: no gate information cliff; F→H4 and E→z have repeatable probe-performance declines; late E/z retains decodable signal but fixed cosine is weak and natural Candidates have no age-13 survivors. All 756 fitted primary probes converged; 108 natural age-13 primary entries are undefined due to zero samples. The 28 unconverged fits are exclusively high-dimensional secondary probes and are not used to infer information absence.

Next short-sequence/vector-readout diagnostic is a recommendation only. No follow-up extraction, training, controller change, threshold change, validation/test evaluation or network deployment was performed during this analysis. Source metrics copied locally have matching SHA-256 values and remain under `.audit-staging/results-frozen-v1`.
