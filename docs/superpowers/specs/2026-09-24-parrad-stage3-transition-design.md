# PARrad Stage P3 Continuous Transition Design

**Status:** Pending user review  
**Date:** 2026-09-24  
**Workspace:** `D:\lab\SEER`  
**Source design:** `方案书v2.3.md`, sections 6-16, 23, 30-33  
**Variant:** PARrad MD-only validation variant; this does not claim the full RD + Point Cloud system

## 1. Purpose

Stage P3 must turn the accepted Stage P2 representation into an online event-boundary mechanism without retraining or weakening the accepted radar-language model. It must answer two separate questions:

1. Has the physical radar stream changed enough to justify creating a Candidate?
2. Has the Candidate become semantically different, stable, and mature enough to replace the Active event?

The implementation must validate `Proposal -> Candidate -> Semantic Verify -> Fork-and-Shadow` on complete, subject-disjoint PARrad recordings. Hierarchical memory is out of scope until this stage passes its go/no-go gates.

## 2. Accepted Inputs and Frozen Contract

The fixed Stage P2 checkpoint is:

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage2_event_joint_formal_v1/best.pt
SHA-256: 939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a
```

The Stage P3 runtime consumes the existing configuration-driven interface:

```text
frame_period_s          = 0.09
frames_per_update       = 2
update_period_s         = 0.18
radar_tokens            = [B, 4, 256]
event_tokens            = [B, 8, 896]
semantic_key            = [B, 128]
```

Stage P3 must never hard-code the original proposal's 20 fps or four input frames per update. Time constants, tolerance windows, maturity, and latency are expressed in seconds and converted through `frame_period_s * frames_per_update`.

The following Stage P2 components remain frozen in every Stage P3 training phase:

- causal MD-GRU Radar Encoder;
- Event Former and base Event Tokens;
- Semantic Projection;
- Qwen backbone and LoRA;
- disposable Stage P2 auxiliary classifier.

Only Stage P3 heads are trainable. The accepted Stage P2 checkpoint must remain byte-identical and is referenced, never overwritten.

## 3. Scope and Non-Goals

Stage P3 includes:

- Local Change Head;
- Radar Transition Head and physical prototype state;
- noisy-OR proposal logic;
- Semantic Difference Head;
- parameter-free Candidate Stability;
- monotonic Verify Head;
- Active/Candidate/Frozen Reference/Shadow state management;
- sealed-event `FinalizationJob` emission;
- validation-only threshold calibration;
- full-recording closed-loop evaluation and latency profiling.

Stage P3 does not include:

- Hierarchical Memory Compressor or retrieval;
- Idle/Presence Controller, because PARrad `-100` labels are not presence ground truth;
- free-form caption evaluation;
- Stage P5 joint fine-tuning;
- new RD, point-cloud, or multimodal backbones;
- test-set threshold selection;
- further Stage P2 training.

## 4. Architecture

```text
continuous update
      |
      +--> frozen Radar Encoder --> F_t [4,256]
                                     |
                       +-------------+-------------+
                       |                           |
                 Local Change              Radar Transition
                       +-------------+-------------+
                                     |
                              noisy-OR proposal
                                     |
                                 fork at t_p
              +----------------------+----------------------+
              |                      |                      |
      Frozen Active Ref       Candidate=P_base       Shadow=Active snapshot
              |                      |                      |
              +---- semantic diff --+-- stability ---------+
                                     |
                              monotonic Verify
                              /              \
                         Confirm            Reject
                            |                 |
                   Candidate -> Active  Shadow -> Active
                   emit sealed event    no replay
```

The Radar Encoder has one continuous state and runs exactly once per update. Only Event Former state and event-local physical statistics fork. Frozen Active Reference is retained for comparison and is not forwarded after the proposal.

## 5. Physical Proposal

### 5.1 Local Change Head

The current MD-GRU `motion_projection` has not been independently supervised, so the first Stage P3 implementation uses accepted Radar Tokens rather than treating `motion_state` as a trusted signal.

For consecutive updates:

```text
X_t = LN(F_t)
D_t = [X_t - X_{t-1}, abs(X_t - X_{t-1}), X_t * X_{t-1}]
Linear(768,128) -> GELU
attention pool over four tokens
Linear(128,64) -> GELU -> Dropout(0.1) -> Linear(64,1)
```

The output logit represents a high-recall local-change score. The first update of a recording has no predecessor and is ignored for this loss.

### 5.2 Radar Transition Head

The head projects the masked mean of `F_t` into a separate 128-dimensional prototype space and L2-normalizes it:

```text
masked_mean(F_t) -> Linear(256,128) -> GELU -> Linear(128,128) -> normalize
```

It maintains:

- `R_fast`: exponential moving average;
- `R_slow`: event-local Welford mean and variance;
- event age in seconds.

The EMA coefficient is always computed as:

```text
alpha = exp(-update_period_s / tau_fast_s)
tau_fast_s = 2.0
```

The transition feature is:

```text
[R_fast, R_slow, abs((R_fast-R_slow)/sqrt(var+eps)),
 R_fast*R_slow, 1-cos(R_fast,R_slow), normalized_age]
```

and is mapped through `MLP -> 128 -> 32 -> 1`.

During supervised head training, physical state resets at ground-truth event boundaries. During validation/test it resets only through controller decisions.

### 5.3 Proposal Rule

```text
p_proposal = 1 - (1-s_local) * (1-s_radar)
```

`tau_proposal` and persistence `{1,2}` are selected on validation recordings. Once Candidate exists, neither physical score participates in Confirm.

## 6. Semantic Verify

### 6.1 Semantic Difference Head

The head consumes the frozen Active reference key `z_A`, Candidate key `z_C`, and maturity:

```text
[z_A, z_C, abs(z_A-z_C), z_A*z_C, 1-cos(z_A,z_C), maturity]
-> Linear(514,256) -> GELU -> Dropout(0.1)
-> Linear(256,64) -> GELU -> Linear(64,1)
```

Training pairs are balanced across:

| Pair | Target |
|---|---|
| mature(A), nascent(A) | same |
| mature(A), mature(A') with the same class | same |
| mature(A), nascent(B) with a different class | different |
| mature(A), mature(B) with a different class | different |

Same-class pairs prefer different sessions and subjects. Pair generation is deterministic for a fixed seed and never crosses split boundaries.

### 6.2 Candidate Stability

For the last three Candidate keys `z_t`, `z_(t-1)`, and `z_(t-2)`:

```text
c1 = cosine(z_t, z_(t-1))
c2 = cosine(z_t, z_(t-2))
v  = mean squared distance from their centroid
stability = clip(((c1+c2)/2) * exp(-v/tau_stability), 0, 1)
```

Before three Candidate updates are available, stability is zero. Stability has no parameters and no independent loss.

### 6.3 Monotonic Verify Head

```text
p_verify = sigmoid(
    softplus(w_diff)      * p_diff
  + softplus(w_stability) * stability
  + softplus(w_maturity)  * maturity
  + bias
)
```

All three influences are therefore non-negative. Confirm requires `p_verify > tau_confirm` for two consecutive updates. Reject requires Candidate age to reach 1.8 seconds and `p_diff < tau_reject` for two consecutive updates.

## 7. Continuous Data and Feature Cache

Training and evaluation start from complete recordings. Label runs are used only to construct supervision; they never define validation/test chunk resets.

The cache builder reconstructs the accepted module through the existing Stage P2 builder: it first loads the Stage P1 Radar Encoder export named by the config, then overlays the trainable-only Stage P2 checkpoint. It preserves the Radar Encoder state across the whole recording and writes one compressed NPZ plus one JSON metadata record per recording. Required arrays are:

```text
radar_tokens   [updates,4,256] float16
radar_mask     [updates,4]     bool
frame_labels   [updates,2]     int64
timestamps     [updates,2]     float64
```

Oracle event-key trajectories used for head training reset Event Former state only at training label-run boundaries, advance once per cached update, and record normalized semantic keys at every maturity step. They are stored as float16 keys, not full Event Tokens. Validation and test controller execution does not use these oracle resets. Each cache record includes the Stage P2 checkpoint SHA-256, configuration hash, manifest hash, subject/session/sensor identity, and array shapes. A fingerprint mismatch is an error; stale caches are not silently reused.

Before cache creation the audit reports per split, subject, sensor, class, event-run count, transition count, ignored gaps, and duration distribution. A split with no valid transition, or a training class with no valid semantic run, blocks Stage P3 training. The two sensor views of one session remain in the same split.

Boundary time is the timestamp of the first valid frame in the new label run. For update time `t`, the physical target is:

```text
y(t) = exp(-(t - nearest_boundary)^2 / (2 * sigma_boundary_s^2))
sigma_boundary_s = 0.36
```

Updates containing only `-100` labels are ignored. `-100` is never converted to Idle.

## 8. Training Schedule

### 8.1 P3-A Physical Proposal

Train Local Change and Radar Transition with weighted BCE on Gaussian soft targets:

```text
L_A = lambda_local * L_local + lambda_radar * L_radar
```

Training batches balance boundary-near updates and ordinary within-event updates. Class weights are computed from the training split only.

### 8.2 P3-B Semantic Difference

Train only Semantic Difference on balanced maturity pairs:

```text
L_B = L_semantic
```

Metrics include same/different AUROC, balanced accuracy, maturity buckets, sensor groups, and subject groups.

### 8.3 P3-C Verify Calibration

Generate deterministic Candidate trajectories from:

- real boundaries;
- proposals shifted before and after a real boundary;
- within-event false proposals;
- short-lived physical spikes.

Train only the monotonic Verify Head:

```text
L_C = weighted_BCE(verify)
```

False Confirm examples receive the positive over-segmentation cost through BCE sample weights; no fifth loss is introduced.

### 8.4 Threshold Selection

The validation grid is fixed before test execution:

```text
tau_proposal:       [0.30, 0.40, 0.50, 0.60, 0.70]
proposal_persistence:[1, 2]
tau_confirm:        [0.50, 0.60, 0.70, 0.80]
tau_reject:         [0.20, 0.30, 0.40]
confirm_persistence: 2
reject_low_persistence: 2
reject_timeout_s:    1.8
maturity_seconds:    1.2
```

Among configurations with median confirmation latency at most 1.2 seconds, select the highest Boundary F1 at `+/-0.72 s`; ties prefer lower over-segmentation, then higher Boundary F1 at `+/-0.36 s`. The chosen thresholds are frozen before the test split is run once.

## 9. Fork-and-Shadow Controller

The controller owns explicit immutable snapshots and branch-local mutable state:

- `ActiveEventState`: streaming Event Tokens, physical prototype state, event start, and radar-token buffer;
- `FrozenActiveReference`: proposal-time Event Tokens and semantic key;
- `CandidateBranch`: starts from `P_base`, owns its key history, prototype state, and buffer;
- `ShadowBranch`: starts from the Active snapshot and consumes the same post-proposal updates;
- `ControllerOutput`: state, scores, decision, timings, and optional finalization job;
- `FinalizationJob`: sealed radar-token buffer, start/end time, provisional Event Tokens, and metadata.

On Proposal:

1. Freeze Active Event Tokens and key.
2. Freeze the old-event buffer boundary at `t_p`.
3. Start Candidate from `P_base` using post-`t_p` updates.
4. Start Shadow from the Active snapshot using the same updates.
5. Keep one shared prompt cache and clone it through the existing adapter contract.

On Reject, Shadow becomes Active in O(1) with its Event Tokens, prototype statistics, and appended buffer. No historical radar update is replayed.

On Confirm, Candidate becomes Active, Shadow is discarded, and a `FinalizationJob` for the old event ending at `t_p` is emitted immediately. Existing Commit Refiner consumption may run outside the critical path, but memory insertion and full caption scheduling are not part of this stage.

## 10. Evaluation

### 10.1 Baselines

1. Proposal-only physical Change Detector.
2. Proposal + Candidate without semantic Confirm, using a fixed delay.
3. Full Proposal + Semantic Verify + Fork-and-Shadow.

All baselines use the same frozen Radar Encoder, recordings, split, update period, and one-to-one boundary matching.

### 10.2 Metrics

Boundary metrics:

- Boundary precision/recall/F1 at `+/-0.36 s` and `+/-0.72 s`;
- boundary mean absolute timing error;
- over-segmentation and under-segmentation rates;
- proposal-to-confirm ratio and false-proposal rate;
- proposal and confirmation latency.

Semantic metrics:

- same/different AUROC and balanced accuracy;
- event-level activity accuracy using the disposable Stage P2 probe only for evaluation;
- semantic purity and event-sequence edit score.

Streaming metrics:

- normal-path and Candidate+Shadow P50/P95 latency;
- Candidate duty cycle;
- Event Former pass count per update;
- Online Control Commit latency;
- Commit Refiner latency measured separately;
- peak GPU memory and state-norm distributions.

Metrics are reported overall and by subject, sensor, recording, class, and event duration bucket.

### 10.3 Go/No-Go

Stage P3 passes only when all of the following hold:

1. Full Active/Candidate improves Boundary F1 over Proposal-only, or lowers over-segmentation at non-inferior boundary recall.
2. Median confirmation latency is at most 1.2 seconds.
3. Semantic Difference outperforms the same/different majority baseline on validation and subject-disjoint test.
4. Reject restores a state numerically equivalent to an uninterrupted Shadow path.
5. Complete recordings contain no NaN/Inf and no sustained monotonic/exponential state-norm growth.
6. The normal and Candidate paths meet the hard 180 ms update deadline at P95 on the RTX 2080 Ti.

The paper targets remain stricter engineering goals: normal P95 below 70 ms and Candidate P95 below 140 ms. Missing these targets while meeting 180 ms triggers optimization work; exceeding 180 ms is a no-go.

Hierarchical Memory work is not authorized unless Stage P3 passes.

## 11. Error Handling and Invariants

The implementation raises explicit errors for:

- non-monotonic timestamps;
- frame/update shapes inconsistent with configuration;
- all-masked updates;
- split or cache-fingerprint mismatch;
- non-finite cached features, scores, states, or losses;
- state batch/device/dtype mismatch;
- Candidate operations while no Candidate exists;
- Confirm/Reject transitions from invalid controller states;
- a second proposal while Candidate is already active;
- a finalization buffer with reversed or empty time bounds.

Required invariants include:

- the Radar Encoder executes once per update;
- Frozen Active Reference never changes after proposal;
- Candidate and Shadow see identical post-proposal Radar Tokens;
- physical scores cannot directly Confirm a Candidate;
- Verify weights remain non-negative;
- Reject performs no replay;
- test labels never influence online controller decisions.

## 12. Code Boundaries

Create:

```text
src/continuous_mmwave/models/transition.py
src/continuous_mmwave/stages/parrad_stage3.py
src/continuous_mmwave/parrad_stage3_data.py
src/continuous_mmwave/parrad_controller.py
src/continuous_mmwave/parrad_stage3_runner.py
src/continuous_mmwave/parrad_stage3_diagnostics.py
configs/parrad-md-gru-stage3-smoke.yaml
configs/parrad-md-gru-stage3.yaml
tests/test_parrad_stage3.py
tests/test_parrad_controller.py
tests/test_parrad_stage3_training.py
```

Modify narrowly:

```text
src/continuous_mmwave/parrad_config.py
src/continuous_mmwave/train.py
src/continuous_mmwave/models/__init__.py
src/continuous_mmwave/stages/__init__.py
```

`train.py` only dispatches Stage P3; the runner and evaluation logic remain in focused files. Existing Stage 1/2 APIs and outputs stay compatible.

## 13. Verification Strategy

All implementation follows test-first development.

Unit tests cover:

- Local/Radar head shape, masking, finite output, and gradients;
- time-derived EMA coefficient and Welford updates;
- Gaussian boundary labels in seconds;
- semantic-pair balance and split isolation;
- parameter-free stability edge cases;
- monotonic Verify behavior and non-negative weights;
- proposal persistence and threshold handling;
- exact fork snapshot behavior;
- Reject equivalence against uninterrupted Active execution;
- Confirm boundary ownership and finalization payload;
- one Radar Encoder invocation per update;
- invalid state transitions and non-finite inputs.

Integration tests cover:

- deterministic Stage P3 cache creation from a tiny continuous recording;
- loading the accepted Stage P2 trainable-only checkpoint;
- one optimizer step for P3-A, P3-B, and P3-C;
- checkpoint save/resume and cache fingerprint rejection;
- full tiny-recording controller execution;
- validation-only threshold calibration and test immutability.

Server verification proceeds in this order:

1. dataset transition audit;
2. frozen-feature cache smoke;
3. one-step CUDA tests for P3-A/B/C;
4. controller correctness smoke with forced Proposal/Confirm/Reject;
5. short training pilot;
6. full validation threshold sweep;
7. one final subject-disjoint test and latency report.

The workspace has no Git metadata. Progress, RED/GREEN evidence, hashes, commands, and server artifacts are recorded in a dedicated SDD ledger instead of commit IDs.

## 14. Implementation Order

1. Audit transition supervision and benchmark the frozen normal/double Event Former paths.
2. Implement time-based targets, cache format, and fingerprint validation.
3. Implement and train physical Proposal heads.
4. Implement and train Semantic Difference and parameter-free Stability.
5. Implement and train monotonic Verify.
6. Implement Fork-and-Shadow controller and finalization-job contract.
7. Add full-recording baselines, threshold calibration, metrics, and diagnostics.
8. Run CUDA smoke, short pilot, validation calibration, and final test in separate output directories.
