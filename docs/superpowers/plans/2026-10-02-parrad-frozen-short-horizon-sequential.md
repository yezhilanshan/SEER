# Frozen Short-Horizon Sequential Decodability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether 3-update histories and Radar/Event combinations improve frozen transition decodability in the unchanged natural controller's decision window, with capacity-matched train-only subject-disjoint probes.

**Architecture:** Extend the existing diagnostic extractor with a continuous retention profile, leaving its old defaults and the formal controller unchanged. Separate endpoint selection, streaming history features, and evaluation into diagnostic modules. Reuse the existing linear-probe solver; source scaling and standardization are learned only from the two probe-training subjects.

**Tech Stack:** Existing Python/PyTorch/NumPy/pytest environments; PARrad Stage P2 cache and frozen Stage P3 controller; SSH jump host for remote replay.

**Spec:** `docs/superpowers/specs/2026-10-01-parrad-frozen-short-horizon-sequential-design.md`, approved 2026-10-02.

**Execution method:** Preserve the user's previously selected Native method. This plan must be reviewed before implementation. This checkout has no `.git`; do not initialize Git or create a worktree just to execute this diagnostic. Save patches, file hashes and verification evidence instead of commit steps.

## Global Constraints

- Train only: subject_02/03/04 × set000–003 × TI_xWR14xx/TI_xWR68xx, exactly 24 recordings. No validation/test arrays or evaluations.
- No Radar/Event Former/Semantic Projection/head training, threshold change, formal controller change, old-output overwrite, or automatic probe promotion.
- Candidate age seconds = `(age_updates-1)*0.18`; continuous retention ages 1–13; trace/controller/reference E/z error <=1e-5.
- Primary natural endpoints age 4/6/7 (0.54/0.90/1.08 s), histories 1/3. Natural age 8 and terminal age >=8 compare 1/3/8 separately. Forced endpoints age 4/6/7/8/13 remain separate.
- Terminal comparisons capture the Candidate before confirm/reject cleanup; exclude recording-end truncations. All variants share endpoint IDs and targets.
- Sources F/E/z/F+E/F+z; history order oldest first; mixed-source order F then E or z; fixed sign projection128 plus cosine/L2 =130 inputs/131 logistic parameters.
- Seeds `[3407, 7211, 12893]`; CPU float64 LBFGS strong_wolfe; lambda1e-3; max_iter200; tolerance_grad1e-7; tolerance_change1e-9; threshold0.5; train-fold-only scales, standardization and inverse-frequency class weights.
- GPU replay uses the isolated SEER environment; delay0.02 s; thermal pause at82°C, resume below78°C; CPU fits use one thread. No dependency installation or unrelated-job termination.
- Formal output: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/frozen-short-horizon-sequential-v1`; separate smoke output; refuse existing destinations.

## Review Focus

- Missing/duplicate ages or changed frozen reference must never be hidden by padding or mixing Candidate histories; Task2 tests must reject malformed traces and exclude incomplete histories explicitly.
- A terminal Candidate can disappear during cleanup; Task1 tests must retain the pre-cleanup terminal trace and preserve decisions exactly.
- Held-out extremes or raw E magnitude must not affect fitted scales/weights/coefficients; Task3 tests must vary only held-out values and verify training artifacts are unchanged.
- Mixed-age terminal classification can succeed using lifetime rather than representation; Task4 must report age-only controls and age-stratified metrics on identical rows.
- Empty/single-class strata, nonconvergence and truncated artifacts must not produce invented AUROC or a successful run status; Tasks2/4 test each condition.

## File map

- Modify `src/continuous_mmwave/parrad_stage3_frozen_layers.py`: add a diagnostic continuous profile and its manifest contract; keep default four-age behavior.
- Create `src/continuous_mmwave/parrad_stage3_sequence_data.py`: strict source checks and Candidate endpoint/history indexing, without fitting or controller operations.
- Create `src/continuous_mmwave/parrad_stage3_sequence_features.py`: streaming fold-only scales, projected relational histories and geometry.
- Create `src/continuous_mmwave/parrad_stage3_sequence_probe.py`: fixed experiment matrix, solver orchestration, artifacts and paired summaries.
- Extend `tests/test_parrad_stage3_frozen_layers.py`; create matching `tests/test_parrad_stage3_sequence_data.py`, `tests/test_parrad_stage3_sequence_features.py`, `tests/test_parrad_stage3_sequence_probe.py`.
- Create `docs/stage3-frozen-short-horizon-sequential-2026-10-02.md`: deployment commands, hashes, smoke evidence, run status and eventual results.
- Do not modify `models/event_former.py`, `parrad_controller.py`, training modules or production YAML for this extension.

### Task 1: Continuous frozen trace profile

**Interfaces:** Add `extraction_ages(profile: str) -> tuple[int, ...]` with `sparse` returning existing `(1,4,7,13)` and `continuous` returning `tuple(range(1,14))`. Define `SEQUENCE_SETTINGS` in this diagnostic extractor with the timing, endpoint, history, source-order and scaling contract from the spec; retain existing `PROBE_SETTINGS` without changes. Add CLI `--trace-profile {sparse,continuous}`, default sparse. Consume existing `extract_recording(..., ages=...)`. Continuous manifests declare `audit_kind="frozen_short_horizon_sequential"`, `trace_profile`, exact ages, both settings dictionaries and hashes of the three sequential modules as well as existing source hashes. Preserve sparse extraction/probe compatibility. The Task2 loader and Task4 evaluator compare their contract to these same declared settings.

- [ ] Write `test_extraction_profiles_keep_sparse_default_and_continuous_ages`: assert exact tuples and reject unknown profiles.
- [ ] Write `test_continuous_replay_retains_terminal_step_without_changing_controller`: use the existing tiny replay fixture; compare sparse versus continuous Candidate IDs/terminal ages/decisions, exact overlapping layer arrays and integrity. Assert a confirm/reject terminal row appears before cleanup and no later natural row for that Candidate exists. Exercise terminal reject too.
- [ ] Run targeted tests in the local `interview` environment; confirm the new tests fail for the missing profile behavior.
- [ ] Implement the profile and manifest fields without changing the replay operations or natural decision logic. Record spec/probe settings hashes and preserve the existing train-only/fingerprint/output guards.
- [ ] Run frozen-layer tests and legacy linear-probe tests; require pass or documented existing optional-dependency skips. Save evidence; no GPU extraction yet.

### Task 2: Checked endpoint and contiguous-history indexing

**Interfaces:** In `parrad_stage3_sequence_data.py`, define immutable `SequenceEndpoint` with `recording_key`, `candidate_id`, `cohort`, `endpoint_row`, `history_rows` (max requested history, oldest first) and `metadata` (endpoint row metadata plus terminal information). Define `load_sequence_source(source: Path) -> dict` to validate complete continuous manifest, settings/code/artifact/source hashes, exact train cohort and recording metadata before feature loading. Define `select_endpoints(rows: Sequence[dict], candidates: Sequence[dict], *, cohort: str, endpoint_kind: str, age: int | None, history_lengths: tuple[int,...], task: str) -> tuple[list[SequenceEndpoint], dict[str,int]]`.

`endpoint_kind` is fixed_age or terminal; terminal accepts natural only. For fixed_age, `age` is the exact endpoint age; for terminal, `age` is the minimum terminal age (3 or8). `task` is update or proposal. All source/history variants use one endpoint list per cohort/endpoint/task comparison, selected with the largest history. Require unique Candidate/cohort/age, consecutive current updates, consistent reference state and proposal update. Structural corruption aborts; ordinary missing history or excluded target increments an exclusion count. Boolean target False is valid. Never read future feature rows.

- [ ] Write `test_history_selection_is_past_only_candidate_local_and_cohort_local`: include interleaved natural/forced Candidates; age7 with max history3 must select exactly ages5/6/7 from the same branch, never later rows.
- [ ] Write `test_shared_endpoint_rows_and_no_eight_update_padding`: age7 accepts histories1/3 and excludes histories1/3/8; age8 supports all three with identical endpoint IDs/targets. A missing age causes an explicit incomplete-history exclusion.
- [ ] Write `test_duplicate_age_and_changed_reference_abort`: duplicate age/current update and mismatched frozen reference state ID must raise ValueError.
- [ ] Write `test_terminal_selection_keeps_final_trace_and_excludes_recording_end`: use terminal metadata; require matching confirm/reject row and minimum age. Missing recorded terminal trace is corruption, not a padded row.
- [ ] Write `test_sequence_source_rejects_invalid_split_hashes_and_partial_outputs_before_arrays`: instrument the array loader so it must not be called for validation/test, incomplete manifests, changed settings/code/source/artifact hashes or an invalid24-record cohort.
- [ ] Run new tests, observe missing implementation failures, then implement the strict indexing and loader.
- [ ] Rerun new and legacy split/reference tests; require all assertions pass.

### Task 3: Streaming capacity-matched relational history features

**Interfaces:** In `parrad_stage3_sequence_features.py`, define `iter_history_blocks(source: Path, endpoints: Sequence[SequenceEndpoint], sources: tuple[str,...], history: int, *, block_size: int=64)` yielding `(endpoint_indices, current_by_source, reference_by_source)` with each source shaped `[N,history,dimension]`. Read only required F/E/z arrays from one recording at a time, slice the last `history` rows from `history_rows`, and preserve endpoint order.

Define `fit_source_scales(blocks) -> dict[str,float]`, using the exact per-source RMS relational norm in the spec and equal endpoint/history-position weights; it receives probe-training endpoints only. Define `project_history_block(current: dict[str,np.ndarray], reference: dict[str,np.ndarray], *, sources: tuple[str,...], scales: dict[str,float], seed: int) -> np.ndarray` returning `[N,130]` with scaled interleaved oldest-first history, sign projection128 and full-history cosine/L2. Reject invalid/unknown sources, mismatched shapes, nonpositive scales and nonfinite values. Reuse the existing projection and geometry helpers; do not retain a projection matrix for every experiment indefinitely.

- [ ] Write `test_projected_history_capacity_for_all_sources_and_lengths`: all five source sets and histories1/3/8 yield130 columns; same seed/order is deterministic, different seed changes projected coordinates, geometry remains unchanged.
- [ ] Write `test_history_order_and_single_update_geometry`: compare a small hand-constructed ordered history to an explicit dense projection; k1 single-source geometry equals legacy cosine with L2 divided by source scale. Reversing time order must change projected features.
- [ ] Write `test_fold_only_rms_scale_ignores_holdout_extremes`: train relational squared norms4 and16 imply scale sqrt10; changing only held-out values must leave scales and subsequently fitted training coefficients unchanged.
- [ ] Write `test_hybrid_source_rescaling_preserves_features`: multiplying one source's current/reference values and its training-derived scale by100 must preserve hybrid features within numeric tolerance. Zero residual scale uses floor1e-6.
- [ ] Write `test_blockwise_features_match_dense_and_reject_nonfinite`: multiple recording blocks agree with dense endpoint ordering; NaN/Inf or inconsistent reference shapes abort.
- [ ] Run failing tests, then implement blockwise scales and features. Estimate scales in a first pass, project in a second pass; retain compact130-column matrices for fitting, not the full cohort high-dimensional histories.
- [ ] Run feature and legacy solver tests, including fit130→131 parameters and train-only standardization/class weights. Record peak-memory expectations; no new dependencies.

### Task 4: Fixed experiment matrix, predictions and paired reporting

**Interfaces:** In `parrad_stage3_sequence_probe.py`, define `experiment_grid() -> tuple[dict,...]`, fixing natural ages4/6/7 with histories1/3, natural age8 with1/3/8, natural terminal min_age3 with1/3 and terminal min_age8 with1/3/8; forced ages4/6/7 with1/3 and8/13 with1/3/8. Tasks update/proposal, five source sets, three folds and three seeds. Define `evaluate(source: Path, output: Path) -> dict`, with CLI `--source-dir` and `--output-dir`.

Use the Task2 shared endpoint list, subject fold masks and Task3 train-only scales. Fit using existing `fit_probe`/`binary_metrics` and unchanged settings. Save every fit's endpoint identities, targets/scores, scales, mean/std, class weights, coefficients, convergence and train/held metrics. Geometry-only fits and fixed cosine/L2 controls share the same rows; terminal age-only fits use age seconds as their single input and are separately named. Save a settings/grid manifest and source manifest hash.

Outputs: `sequence-results.jsonl`, `sequence-summary.json`, `endpoint-counts.json`, per-fit NPZ artifacts. Write summary only after the expected grid is accounted for as fitted/undefined; any failed fit keeps run incomplete. Nonconvergence is a flagged result, not silent success. Single-class training cannot fit; mark undefined without altering the solver. Single-class held-out metrics retain predictions with AUROC/BA undefined.

Aggregate three held subjects with macro metrics, then seed median/range; never choose the best seed. Compute paired 3-minus1 and hybrid-minusconstituent deltas only after asserting identical held-out row identities/labels. Add session/sensor/terminal-age strata and same-recording conditional AUROC weighted by valid positive-negative pair counts; show support counts and undefined strata. Natural decision histogram and sample counts precede performance tables. No pooled independent-update confidence claims.

- [ ] Write `test_grid_respects_eight_update_timing_and_separates_terminal_cohorts`: assert no8-update age<8 and all declared comparisons are represented with separate cohort labels.
- [ ] Write `test_small_synthetic_sequence_evaluation_is_subject_disjoint`: use a reduced explicitly injected test grid and synthetic train-only source; verify no held labels enter fitting and primary parameter_count131 for every source/history.
- [ ] Write `test_pair_deltas_require_identical_endpoint_ids_and_targets`: shuffled/missing endpoints must not be silently compared.
- [ ] Write `test_terminal_age_baseline_and_single_class_strata_are_explicit`: retain age-only control, stratified support and undefined metrics; do not call undefined AUROC zero or0.5.
- [ ] Write `test_failed_or_nonconverged_fits_are_not_hidden_and_outputs_are_immutable`: inject solver failure/nonconvergence; failure leaves incomplete status, nonconvergence is enumerated, existing output remains byte-identical and is refused.
- [ ] Run failing tests, implement evaluator/reporting, then run all new tests and existing frozen/probe/controller tests. Use a reduced synthetic grid only for tests, never as a default production flag that could silently change the declared full grid.

### Task 5: Remote verification, smoke and formal diagnostic chain

**Files:** Create `docs/stage3-frozen-short-horizon-sequential-2026-10-02.md` with actual commands/evidence and pending status. Deployment changes only the modified/new diagnostic modules and targeted tests inside remote SEER; preserve original files with an explicitly named backup before replacing any existing module.

- [ ] Run the local targeted suite with `$env:PYTHONPATH='src'` and `C:/User/anaconda/envs/interview/python.exe -m pytest tests/test_parrad_stage3_frozen_layers.py tests/test_parrad_stage3_linear_probe.py tests/test_parrad_stage3_sequence_data.py tests/test_parrad_stage3_sequence_features.py tests/test_parrad_stage3_sequence_probe.py -q`. Record passes/skips and reasons; do not install optional dependencies locally.
- [ ] Connect using `ssh -J 36373@100.83.206.93 xibeidaxue@10.15.22.62`; use the supplied credentials interactively without embedding passwords in scripts/logs. Check GPU load/temp, RAM, disk space, existing jobs and remote source hashes. If busy, do not stop unrelated jobs.
- [ ] Transfer the reviewed patch/tests, retain existing-module backups, verify transferred SHA-256 and run the targeted tests using `/home/xibeidaxue/SEER/.conda/envs/seer/bin/python` with `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Do not modify the environment.
- [ ] Read source config/checkpoint/threshold locations from the approved existing audit manifest, not a guessed newer checkpoint. Verify unchanged Stage2 SHA939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a, Stage3 SHAc359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6 and threshold SHA5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef. A mismatch requires explanation, not auto-replacement.
- [ ] Run one-recording continuous smoke for `subject_02__set000__TI_xWR14xx` under a fresh sibling directory `frozen-short-horizon-sequential-smoke-v1`; use the existing config `configs/parrad-md-gru-stage3-sequence-v2.yaml`, Stage3 `best_verify.pt`, and `timeout-reject-ablation-shards-v1/merged/validation_thresholds.json`. Verify E/z errors<=1e-5, no nonfinite traces, exact overlapping-age Candidate IDs/decisions/arrays against the prior audit, histories and terminal rows. Smoke is not eligible for a24-record subject-disjoint evaluation.
- [ ] Measure smoke time, RAM and shard size; estimate the24-record replay from update counts with thermal/disk uncertainty. Check destination absence and volume headroom again. Launch a background shell chain that runs full extraction `python -m continuous_mmwave.parrad_stage3_frozen_layers ... --trace-profile continuous --device cuda:0 --step-delay-s 0.02` then, **only after successful complete extraction**, `python -m continuous_mmwave.parrad_stage3_sequence_probe --source-dir <new-root> --output-dir <new-root>/probes`. Record parent/child PID, stdout/stderr log and shell exit status. No scheduler changes to other jobs.
- [ ] Inspect initial progress and GPU temperature, then report real launch status/PIDs/output path and evidence-based rough ETA. Do not claim the diagnostic is completed merely because the background process started. Stop this implementation turn once the healthy background run is handed off; recurring monitoring requires the product mechanism if requested.
- [ ] When results are later available, verify complete manifests/grid/source hashes before analysis. Present natural counts first, early-window paired gains and cross-subject/seed stability, then hybrids and late/forced caveats. Recommend the next study; do not train or deploy a new model automatically.

## Plan review and execution gate

This plan covers the approved scientific contract with five independently testable tasks. Native execution is retained from the user's earlier choice. Approved for implementation by “开始写代码呀” on 2026-10-02. Tasks1–4 implemented and tested; Task5 deployment/tests passed and background chain PID8408 issued, startup verification in progress. The companion progress ledger carries RED/GREEN evidence and deferred review minors. No experiment-result completion is implied by a launch.
