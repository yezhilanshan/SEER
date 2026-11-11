# Frozen Radar Readout Pilot — approved Experiment 1

User approved the bounded in-chat design with “可以，开始吧”; resumed implementation with “继续” after the permission-review service usage limit prevented the first test command. No architectural/production modification is authorized.

## Scope and fixed protocol

Reuse the complete `frozen-short-horizon-sequential-v1` extraction, exact 24 TRAIN recordings: subject_02/03/04 × set000..003 × both sensors. No Radar/Event Former/Verify/LoRA/checkpoint/controller/threshold training or editing. Fit only offline diagnostic readouts on CPU; keep validation/test closed. Old 0.679 terminal AUROC is contextual, not a fair reference under the new fitting/calibration protocol.

- Inputs are actual current Radar F and the controller-bound frozen Active reference F. Causal natural histories only, three past/current updates, age at least 3. No forced survivors, padding, cross-Candidate histories or future features.
- Label: current clean GT run differs from frozen Active reference GT run; inherit source mixed/ignored/gap/multiple-boundary exclusion. Proposal-level labels are not used.
- Use all eligible natural online update rows for fitting; base weight 1/(number of eligible rows for that Candidate), then fit-only weighted inverse class-frequency balance. Candidate base contributions are equal before class balancing.
- Three probe-held subjects. For each fold, the other two subjects' set000..002 are fitting, their set003 is solely calibration (both sensors together); held subject all four sessions is evaluation. The frozen upstream has seen the train subjects already, so this is probe-only subject holdout.
- Fit-only Candidate-weighted source RMS residual scale and feature mean/std. Separate calibration negatives choose thresholds; no held cutoff selection, no model selection or early stopping using calibration/held labels.
- Fit a projected130-input/131-parameter baseline and age-only control on the exact same new split and weighted online-row training protocol as the raw heads. This is a refit, not a literal reproduction of old terminal-only fitting.
- Main heads: raw `[ΔF, age]` Logistic (1026 coefficients for F dimension1024); raw `[F_t,F_a,age]` Logistic (2050 coefficients); `[F_t,F_a,ΔF,age]` MLP32 (98,401 coefficients); same MLP plus stability (98,433 coefficients). Different model capacities are explicitly recorded; gains are not evidence of head-only causality.
- Stability is mean(cos(F[t−2],F[t−1]),cos(F[t−1],F[t])); only current-history Radar inputs. Both MLP variants otherwise share architecture, seeds, fitting rows and optimization budget.
- Logistic: CPU float64 LBFGS strong_wolfe, ridge lambda1e-3, max400 iterations, grad tolerance1e-7/change1e-9. Record nonconvergence explicitly, never silently retry/tune.
- MLP: one32-unit GELU hidden layer, CPU float32, AdamW lr1e-3/weight_decay1e-2, batch256, fixed60 epochs. No validation-driven early stopping. Mark fixed-budget completion, not optimization convergence.
- Seeds3407/7211/12893 for projected baseline and each MLP. Raw/age Logistic is deterministic and fitted once per fold, not artificially triplicated: total36 fits,144 scored group records.
- Calibration target FPR5% and10% on all eligible calibration online updates, one threshold per fit and target shared across all evaluation groups. Higher empirical quantile plus nextafter upward rejects ties. Target calibration FPR is not an enforced held-out FPR.
- Common held evaluation endpoints: natural ages4/6/7 (0.54/0.90/1.08s), and actual confirm/reject endpoint with minimum age3. Each comparison's IDs/labels are fingerprinted. Report AUROC/AP/BA@.5, actual FPR/recall/precision at calibration thresholds, recording/session/sensor/age strata. Terminal timing and survivor selection still reflect old controller behavior.
- Save fitting/calibration/held IDs, targets, scores, transforms, class/base weights, model state, losses and exact artifact hashes. Source/codes revalidated before completion. Existing output is never overwritten; exceptions record failed status.

## Implementation

New files only:

- `src/continuous_mmwave/parrad_stage3_radar_readout.py`: split, weights, features, calibration, tiny-head fitting and reproducible prediction.
- `src/continuous_mmwave/parrad_stage3_radar_readout_runner.py`: checked immutable inputs, online row assembly, full matrix, NPZ/JSON outputs and source integrity.
- `tests/test_parrad_stage3_radar_readout.py`: real synthetic24-record pipeline and safety/unit tests.
- `.audit-staging/run-radar-readout-pilot-v1.sh`: CPU-only low-priority background launcher, separate logs/PID/status, fail closed, no automatic retry.

Controller SHA unchanged: `b3c37594f78b85c2342fb5de17ff2662f48ed5cfd1adcbfdbce8f68dc8153b44`; Event Former SHA unchanged: `22d666a8b23f9670fe14140bced3bd7cdd28690bea1fe7e7a1b99c3c34f83ca5`.

## Verification/status

TDD RED:8 failures, all missing new module,4.30s. GREEN:8 passed,4.74s. Related local regression53 passed,23.76s. Complete local pytest attempted: collection fails only because interview lacks h5py at `tests/test_parrad_convert.py`; no dependency installed. Independent read-only code review pending. School server resource check10:37CST:~59.8GB RAM available, GPU34°C/0%; destination/new module paths absent.

Remote target: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/radar-readout-pilot-v1`; log sibling `radar-readout-pilot-v1.logs`. Not launched at this status snapshot. Do not interpret this document as results or production-promotion authorization.

### Deployment verification

Failure context regression: RED KeyError active_fit → GREEN. Final related local54 passed21.99s; remote54 passed6.65s. Complete remote pytest:332 passed,2 pre-existing Gemma failures,1 existing amp FutureWarning,14.54s. Failures are `test_gemma3_adapter_matches_native_prefix_and_targets_only_qv` and `test_gemma3_adapter_honors_local_sliding_window`, both `Gemma3RotaryEmbedding.None_inv_freq`; excluded from repair by user. Local full-suite collection limitation remains missing h5py; no installation performed.

Independent read-only review found no Critical/Important issues; Minor missing failed-fit identity was fixed with RED/GREEN regression and re-reviewed. Runtime Torch/NumPy/device and active-fit context now saved. New package SHA `800a639e345bad8c2224d857989b4dc1c9d0fd523042ca31baad6fb8bbd9f95f` matched after transfer; archive contains only this task's five new files. Existing paths were checked absent before extraction. Bash syntax and CLI help verified. No Git repository exists, so no Git init/commit/worktree performed.

New module hashes: readout `1c8e9e20ceac85062a0d57600a4637feb78c3b55cc280f0a284a9849307e98c1`; runner `90cb59cdff7e92d04bc19a2f7a60621e3963b8dc93ad57bbc2734e1b1d9972d1`; test `f073411c5a9dbc1cedf7b5c63abff5bee94638a55c8ca4001ac4a13995c79e01`. Later local edits to this human-readable run document do not modify deployed executable code.

### Live launch

Launched2026-10-02 10:51:39CST, shell PID15703, CPU-only Python PID15707. Fresh remote controller/Event Former hashes still match originals. At parent elapsed59s, source verification and preparation had succeeded and7/36 fits were complete, including all first-fold Logistic fits converged and the first60-epoch MLP fixed budget complete. Example fit timings raw ΔF Logistic3.24s, full relation Logistic6.97s, MLP7.19s. This is startup verification, not a completed-result claim. No Traceback or failure at that snapshot. Logs/PID/status are under `radar-readout-pilot-v1.logs`; output manifest records the active fit and completed fit matrix.

At Python elapsed2m16s: status=fitting,21/36 fits complete, active fit022 stability_mlp/seed3407/held subject_03. Dataset14,357 valid online updates from3,728 Candidates; source manifest SHA remains `5d9439e5b4c8cb533bc0de067faf7d2626307ec4a98c9a50e90e1b23638af440`. First fold fitting/calibration/held row counts6932/2221/5204; second7560/2361/4436. CPU approximately one core, RSS1,892,092KiB; GPU33°C,0% utilization. Healthy startup handoff, no completed-result or readout-improvement claim.

### Completed results checked 2026-10-02 19:23–19:27CST

Completed10:56:08CST, chain wall time4m29s including initial source verification.36 fits/144 group records complete; all Logistic fits converged, MLPs completed fixed60-epoch budgets. Processes15703/15707 exited normally, chain status complete. Fresh read-only verification checked all36 fit NPZ hashes, summary/results/endpoints hashes, readout code hashes, and `load_sequence_source` checked original source artifacts/code/checkpoints again; source manifest unchanged. Train-only and test_evaluation_allowed=false remain intact.

Local downloaded metrics under `.audit-staging/results-radar-readout-v1` match manifest SHA: summary `21e329d7aa28bc0c4430975a2cd2009a03136de8e993e4758efc40693ac99abf`; results `8fcc2a8c11aebd7e3dd13752d498cc14625365d3af6bca4b5aed6cd29b449c09`.

Actual natural terminal endpoints, minimum age3; subject macro then seed median. Working-point columns use thresholds chosen solely on all-online calibration negatives at target FPR5%, not held-out-selected cutoffs:

| Head | AUROC | AP | Actual held FPR | Held recall |
| --- | ---: | ---: | ---: | ---: |
| Age control | 0.5840 | 0.2449 | 4.29% | 2.27% |
| Projected baseline | 0.6736 | 0.3426 | 9.30% | 20.17% |
| Raw ΔF+age Logistic | 0.6687 | 0.3318 | 12.24% | 25.70% |
| Ft/Fa/age Logistic | 0.6787 | 0.3488 | 13.55% | 29.59% |
| Full relation MLP | 0.6283 | 0.3034 | 8.47% | 17.99% |
| MLP+stability | 0.6298 | 0.3102 | 8.11% | 17.15% |

No terminal head reaches0.75; best raw Logistic gains only approximately0.005 AUROC over the refitted same-protocol baseline. MLP/stability do not solve the problem under this fixed budget/protocol. Higher raw Logistic recall is accompanied by higher actual FPR, not a demonstrated low-FPR win. These FPRs are diagnostic update-level semantic labels, not the production controller's old proposal-level false_confirm_rate. Neither a head-only causal conclusion nor absence of all Radar transition information follows from this pilot. No additional training, controller promotion, or validation/test evaluation was started during the status check.
