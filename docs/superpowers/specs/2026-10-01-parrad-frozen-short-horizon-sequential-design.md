# Frozen Short-Horizon Sequential Decodability Audit

Date: 2026-10-01. Status: approved by the user's “开始” on 2026-10-02; implementation plan pending review.

## Purpose and scope

Implement the user's attached follow-up to the completed frozen-layer audit. Determine whether continuous short histories and Radar/Event combinations improve transition readout at the ages at which the **unchanged natural controller actually decides**. This is a diagnostic, not Stage P2/P3 training or a new deployed Verify head.

Freeze Radar Encoder, Event Former, Semantic Projection, all controller heads, Fork initialization, gate, thresholds and controller decisions. Only fixed-capacity logistic probes are fitted. Read exclusively the 24 train recordings: subjects 02/03/04, sessions set000–003, both radar sensors. Validation and test remain unopened. Subject disjointness applies to probe fitting, not the upstream frozen models, which have already seen these train subjects.

## Chosen approach and alternatives

Use the existing verified frozen-layer replay with continuous Candidate trace retention, then a separate sequential evaluator. Reuse its actual-state reference binding, terminal-step capture, forced-horizon separation and train-only validation guards. Do not reconstruct missing updates by interpolation from the old four-age traces.

The alternatives are unsuitable for this question: padding an incomplete eight-update history changes the meaning of history length; extending the natural controller lifetime changes its operational distribution. Neither is included. A forced continuation is permitted only as a separately named diagnostic cohort, never as natural data.

## Timing contract

Update spacing is 0.18 s. Candidate age 1 is the Proposal update, at elapsed age 0; age a is `(a-1)*0.18` seconds. A history of k Candidate updates needs age at least k. Eight updates first exist at age 8, **1.26 s**, outside the proposed primary 0.5–1.1 s decision window.

Predeclare endpoint groups:

| Group | Endpoints | History comparison | Interpretation |
| --- | --- | --- | --- |
| Primary fixed-age natural | age 4/6/7: 0.54/0.90/1.08 s | 1 vs 3 | Early evidence available before/on actual natural decisions |
| Secondary actual-decision natural | Every confirm/reject terminal update | 1 vs 3, on terminal age >=3 | Readability immediately before the unchanged decision |
| Extended natural survivors | age 8: 1.26 s | 1 vs 3 vs 8 | Later, survivor-selected subset; not an early-window claim |
| Extended actual-decision natural | Confirm/reject terminal age >=8 | 1 vs 3 vs 8 | Same late-terminal subset for every history |
| Forced diagnostic | age 4/6/7/8/13 | 1 vs 3; also 8 where age >=8 | Fixed-lifetime mechanistic context, not deployable natural evidence |

All history lengths in a comparison use the **same Candidate IDs, endpoint updates, endpoint labels and recording rows**. At a fixed endpoint, assemble history from that same Candidate's consecutive, past/present updates, oldest first. Never cross Candidate IDs, borrow pre-Fork Active updates, pad, peek ahead, or substitute forced updates after a natural decision. Keep recording-end truncations distinct from terminal decisions and exclude them from the decision cohort.

For fixed-age temporal comparisons, endpoint eligibility is shared across histories. Report counts and exclusions first. Age-to-age improvements have changing labels/cohorts and are not pure maturation effects. Terminal-age distributions also carry lifetime/selection information; save an age-only diagnostic baseline and stratify terminal metrics by age. A good mixed-age terminal AUROC alone is insufficient.

## Continuous collection and reference alignment

Retain Candidate traces at each age 1 through 13, using the same controller replay as the previous audit. Natural collection stops with the actual Candidate lifetime; diagnostic forced continuation reaches at most age 13 or recording end. Retain the existing actual computational trace `F/H4/R/E_hat/E/z` and gate auxiliaries for integrity and comparison, but only F/E/z enter this experiment's probes. This avoids a second approximation of the runtime or a controller refactor.

Record Candidate ID, Proposal update, actual frozen Active reference update and state ID, cohort, age, current update, sensor/session/subject, terminal age/decision and exclusions. Use the controller's actual frozen Active trace as reference at **every** history position. Compare captured E/z and reference E/z against controller tensors at every applicable natural/forced step. Abort on max absolute error >1e-5, nonfinite tensors, source fingerprint mismatch, changed checkpoint or changed thresholds. Capture the terminal Candidate representation before confirm/reject clears or promotes it.

Store recording-level compressed shards rather than one large in-memory dataset. Preserve prior outputs. The earlier four-age run occupied about 3.3 GiB; continuous full traces will be larger, so preflight memory/disk and measure one-recording smoke output before the full run. The volume currently has ample space, but the smoke measurement, not a speculative multiplier, determines the run estimate.

## Labels

Primary: the existing clean update-level same/different GT-run target relative to the actual frozen Active reference update. Require clean reference/current frame labels, no -100 or mixed updates in the relevant interval, no ignored gap, and at most one intervening GT-run boundary. GT labels are retrospective targets only; no labels, timestamps of future boundaries, cached oracle keys, oracle maturity or run IDs become features.

Secondary: the existing strict Proposal-level boundary label, with ±0.72 s proximity and no ambiguous/multiple boundaries over the 2.16 s horizon. Keep the same label definition at every endpoint for comparability to the previous audit. Its future exclusion rule makes it an auxiliary retrospective task, not an online label available at inference. Never combine the two label meanings or duplicate update rows as independent Candidate examples within an endpoint fit.

## Feature and capacity contract

Five source sets: F, E, z, F+E, F+z. Three histories: 1, 3, 8 where feasible. Flatten each source in the existing tensor/token order. For a source s and history position j, use relational vector `x_s,j - x_s,ref`.

Mixed-source scale must not be set by raw state magnitude or unequal dimensions. For each fit, estimate one label-free scale per source using **only the two probe-training subjects** on the fit's eligible Candidate histories:

`scale_s = max(sqrt(mean_i(mean_j(||x_s,i,j - x_s,ref,i||^2))), 1e-6)`.

Each Candidate contributes equally and each history position contributes equally. Divide both current and reference source vectors by this scale. Concatenate positions oldest-to-newest; within a position use the source-set order F then E or z. Save the scales. Apply the identical training-derived scales to the held-out subject. No per-held-subject fitting or label-based scale selection.

Project the concatenated relational history to 128 coordinates with a fixed random sign projection, then append two geometry scalars: cosine distance and L2 between the corresponding full scaled current history and repeated scaled reference history. The single-source k=1 case recovers the previous relational definition up to a common source scale. Every primary probe has **130 inputs and 131 fitted logistic coefficients**, including bias. Equal learned capacity does not imply equal random-projection information retention; weak projected performance cannot prove information absent.

Predeclare seeds `[3407, 7211, 12893]` for all sources, histories, folds and endpoints. Reuse the previous solver exactly: float64 CPU LBFGS, strong_wolfe, lambda 1e-3, max_iter 200, gradient tolerance 1e-7, change tolerance 1e-9, classification threshold 0.5. Standardization and inverse-frequency class weights use probe-training subjects only. Do not tune or reverse directions on a held-out subject. Mark nonconverged fits explicitly; do not quietly retry with changed hyperparameters.

No nonlinear sequence head, raw-dimensional sensitivity fits or extra learned temporal parameters in this first experiment. Fixed geometric scores and geometry-only fitted probes provide controls on the same eligible rows; terminal-age-only control is diagnostic and separately labeled, not counted as a representation source.

## Evaluation and interpretation

Three leave-one-train-subject-out folds, holding out all sessions and both sensors. Report AUROC, AP, prevalence baseline, balanced accuracy at 0.5, positive/negative counts, convergence and per-fit predictions. Save train and held-out metrics, feature transforms, coefficients, seeds and row identities for reproducibility. Undefined single-class metrics stay undefined.

Primary comparisons are paired differences on identical held-out rows: 3 minus 1 update; F+E versus F and E; F+z versus F and z. Report subject-macro means and seed median/full range, plus per-subject and per-session/sensor diagnostics. Same-recording conditional AUROC, with pair support counts, is a descriptive check against session shortcuts. Do not pool correlated updates for an independent-sample confidence claim or choose a winning seed.

Show natural counts and decision ages before feature rankings. Do not claim a useful natural eight-update result if the eligible survivor subset is small or single-class. The previous run suggests only about 144 Candidates reached age 8; verify this again rather than assuming the new replay matches. An age-13 forced result cannot establish that the current controller has useful evidence before confirmation.

There is no automatic promotion based on an AUROC threshold. Consistent early E/z sequence gains would motivate an online-distribution evidence-reader study; stronger F or hybrid results would motivate a Radar residual study; weak projected results require inspecting compression, convergence, counts and task ambiguity before proposing representation retraining. All are later user decisions.

## Safety, outputs and verification

New output root: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/frozen-short-horizon-sequential-v1`. A separately named smoke directory precedes the formal extraction. Refuse pre-existing output destinations; no overwrite/delete/restart of earlier experiments. Record source checkpoint/config/threshold/code hashes, cache fingerprint, exact cohort matrix, split guard, timing contract and probe settings in the manifest. Recheck hashes on extraction completion and before probes.

Use the existing isolated SEER environment and GPU 0 for frozen replay; CPU for logistic fitting. Keep the existing 0.02 s step delay and thermal pause at 82°C/resume below 78°C. Do not stop unrelated jobs or install dependencies into other environments. This work does not change any deployed model or controller.

Before the full run, tests must cover consecutive history assembly, endpoint equality, no padding/future/cross-Candidate leakage, natural versus forced separation, terminal-step capture, clean-label exclusions, 130/131 capacity, fold-only scale/standardization/class weights, source immutability and invalid split rejection. Run one-recording replay smoke, assert E/z parity and verify overlapping ages against the previous audit. Only then run the 24-recording collection and probe chain.

## Written-spec review

The user approved this concrete contract on 2026-10-02, including: **1/3 updates are the early-window comparison; 8 updates are a separate >=1.26 s comparison, never padded into the early window.** The next artifact is the implementation plan. No implementation or new GPU run has started under this contract yet.
