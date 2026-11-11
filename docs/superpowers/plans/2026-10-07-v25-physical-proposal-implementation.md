# Stage 3A: Physical Proposal Learning

Authority: 计划/方案书v2.5.md §§8,26,Stage3A,37 and user's explicit correction/authorization. This supersedes the earlier boundary-head terminology and Candidate classifier proposal. Scope is exactly 1000 actual AdamW updates of Local Change and Radar Transition, seed42, lr .001, weight_decay .01, clipping1, FP32. Evidence500, Semantic Difference, gate, encoder, Event Former and P remain frozen. No Verify Head or L_verify. No automatic next phase.

## Locked data and supervision

Reuse preparation-v2 encoder cache and its locked index SHA; fit/development split unchanged, sealed holdout/official test never read. Gaussian sigma .36 seconds, max over known adjacent-frame GT class changes. Align scores and labels to the second/final frame timestamp of each two-frame radar update. Both frames and the preceding update must be known; include updates straddling a known class change. Unknown gaps never create boundaries. Radar inputs remain causal; future GT only supplies offline labels.

Use 16 subject/sensor balanced continuous chunks of at most64 updates per AdamW step, with per-record detached prototype state and32-update TBPTT. Preserve actual chronological history across chunks, reset at recording end. Fit-only Gaussian mass determines separate positive/negative BCE component weights: a=N/(2 sum y), b=N/(2 sum(1-y)); loss=sum[-a*y*log sigmoid(z)-b*(1-y)*log sigmoid(-z)]/sum[a*y+b*(1-y)]. This balances global fit component masses without cross terms from scalar sample weighting. Each head loss coefficient1. Log record/chunk/valid-update draws. Native zero-step preflight rejects any fit chunk with no eligible loss before launch. Source-head handoff is explicit from SHA-locked Evidence-step500; preserve its500 Evidence count and all nonphysical tensors. Validate prior code digest against the prior isolated source tree and archive manifest; record old/new code digests rather than silently bypassing bundle checks.

## Locked evaluation

Standalone Proposal stream, no GT prototype resets, no Candidate/Confirm or final segmentation claim. Primary operating point: fusion 1-(1-local)*(1-radar), threshold .6, persistence2, one proposal on the second qualifying update per contiguous high-score excursion (rearm only below threshold or when annotation eligibility breaks). Also report each head and thresholds .3/.4/.5/.6/.7/.8/.9 without choosing a dev winner. This is a standalone excursion detector, not calibrated Candidate controller output.

Main matching tolerance ±.72s. Proposal emission time is update completion time. Greedy chronological one-to-one matching to earliest unmatched eligible GT boundary in tolerance; duplicate proposals count as false. Eligible boundaries require a score-eligible update within tolerance. Report total/excluded GT boundaries separately; count only known scored update durations (actual timestamp deltas), never span unknown gaps. Main metrics: Proposal Recall; False Proposals/minute; signed latency (median/p95, misses excluded and counted); Proposal Density (proposals/minute, plus fraction of eligible updates above threshold). Also report post-boundary-only recall in [0,.72] and early-proposal fraction so negative latency cannot be mistaken for causal detection. Missing denominators give null, never zero. Report all, subject, sensor, subject×sensor. Boundary F1/MAE is not the acceptance gate.

Evaluate checkpoints0/100/500/1000, save raw timestamp/GT/valid/Local/Radar scores and predictions. The final1000 checkpoint is preselected; no dev early stopping, threshold optimization or checkpoint cherry-picking. Spec defines no numerical Proposal pass threshold; this round reports the recall/false-rate tradeoff and infrastructure checks, never invents formal Go. Frozen frontend retrieval is preserved by verified identity; automatic-event retrieval still awaits later phases.

## Tasks

1. Tests first for causal timestamp alignment, mixed-frame boundary inclusion, unknown-gap exclusion, one-to-one duplicates/latency/duration/null cases, fit-only weights and weighted streaming BCE.
2. Implement isolated Proposal sampler/evaluator/native worker, explicit locked Evidence handoff, checkpoint metrics, source/optimizer/frozen-state readback and failure evidence.
3. Run targeted and v25 regression tests, one fresh whole-source review, package with source SHA, native preflight, exclusive detached launch. Record PID, status, observed steps; do not claim1000 complete while running.

Later order only: Semantic Difference oracle + stable closed-loop hard pairs → episode-level calibration of H+/H-/rho/tau_s/tau_d/lambda_e (and other declared controller parameters) → Confirm-after Boundary Localization → automatic events + Memory + QA.
