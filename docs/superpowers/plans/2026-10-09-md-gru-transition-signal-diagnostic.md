# MD-GRU transition signal diagnostic implementation plan

> **For agentic workers:** use executing-plans for native implementation, with tests before diagnostic code. The user approved the in-chat diagnostic design and execution. No further training authorization is inferred for encoder updates.

**Goal:** Locate where action-transition information weakens between MD input, GRU recurrence, residual fusion and exported Radar Tokens.

**Architecture:** A diagnostic-only exporter observes the existing frozen MD-GRU, reproduces cached Radar Tokens, then analyzes causal changes and fixed-budget probes. All work is isolated from production modules.

**Tech Stack:** Existing Python, PyTorch, NumPy, scikit-learn; interview environment for local tests, existing seer environment for server execution.

**Spec:** User-approved preceding chat design; this plan locks routine experimental choices before extraction or dev results.

## Global constraints

- Existing index SHA205d96c8993c1570c13b0cdb0a708faa7025d094d1b974b5a8ecc5420bbbdc60; exactly18fit/8TRAINdev recordings; no sealed holdouts or official-test arrays. Upstream-seen TRAIN development, not independent test.
- Original encoder state digest a2c85175d7c95986682ca55dec886042fbe57ecfedeaa41d6155dbdfe2cb38e7; Stage1 export SHA42065f59311cb41e6c23c7734b66392028d634d7c11264bfa3325cb2be7b7289. Require equality before inference, never substitute another checkpoint silently.
- Encoder, EventFormer, gates, P, Evidence and existing Proposal Heads receive zero updates. New diagnostic probes are not installed into any production bundle.
- Seven observations: raw MD, MD after learned input LayerNorm, projected MD, raw GRU recurrence, residual+output-LayerNorm feature, mean exported Radar Token, flattened four Radar Tokens.
- Every observation is available causally. Continuous two-frame updates, last-frame sample at current update; recording-only reset, no GT reset. Discard same odd tail as cache, verify labels/timestamps and raw source SHA.
- No-training scores: adjacent cosine distance, relative L2 change, relative deviation from previous6update mean. Report raw distributions, AUROC/AP, fit-calibrated event metrics and boundary-relative offsets.
- Probe features: current, previous, signed adjacent delta, current minus previous6update mean. Causal history resets per recording; first6updates invalid for probes and matched comparisons. Fit-only component standardization, no recording time/duration/subject/sensor labels as probe inputs.
- Two probe families per representation: native-dimensional linear and MLP(hidden64,GELU,scalar). Same seed42;500actual AdamW steps each; lr0.001,wd0.01,gradientclip1;16continuous64update chunks per step, identical fit chunk sampling across14arms; unweighted Gaussian BCE(sigma0.36s). No validation checkpoint selection, no hyperparameter search; report parameter-count differences explicitly. Native dimensionality avoids pretending PCA truncation proves information loss.
- Score labels: positive within0.36s, stable negative farther than0.72s, ignore intervening band; also Gaussian-target fit. AP must report positive prevalence. The absence of a distance peak or probe success is not proof of absence/presence of all information.
- Fixed event trigger: existing causal peak family,lookahead1update,NMS0.36s,neighbor-high rule unchanged. Threshold0.01..0.99 plus1.0 selected on fit separately atFP/min caps3/5/8; dev never tunes thresholds. Report actualFP,Recall,Density,latency; caps refer tofit,notexact dev matchedFP.
- All14probe results and no-training paths reported bysubject/sensor; compare against same-trigger existing500stepHead scores when available; no winning-arm-only report. Event comparisons use common valid context.
- Preflight plan/code/sourceSHA before run; checkpoint/encoder hashes before and after; archive original predictions, probe states, logs, normalization, selections and evidence withSHA, safe member path check before extraction.

## Review focus

- Hooks must observe raw GRU recurrent sequence, not confuse final carried state with LayerNorm(projected+recurrent).
- Prefix perturbation and chunk equivalence; diagnostics cannot consume future MD.
- Unknown labels/time gaps must not create GT boundaries or valid history across gaps.
- Empty positive classes or no proposals reported as missing/zero as mathematically appropriate, not fabricated passing metrics.
- No large-RD load or accidental holdout reads; only pinned26record arrays and pinned existing checkpoint.

### Task1: Causal capture and feature tests

- [ ] Add tests for exact exported Token reproduction, raw recurrent capture, prefix causality, past-only features and masks.
- [ ] Implement isolated `.audit-staging/md_gru_signal_core.py`; no production edits.
- [ ] Run targeted interview tests; record RED/GREEN.

### Task2: Extraction, probes and analysis

- [ ] Implement `.audit-staging/run_md_gru_signal.py`: verify pinned sources, export26streams, train14diagnostic probes, calibrate onfit, evaluate once ondev, archive.
- [ ] Test known synthetic recoverable transitions, fit-only normalization and common trigger integration; recordactual probe counts.
- [ ] Upload isolated code/plan/manifest, run preflight then execute without duplicate workers.

### Task3: Readback and recommendation

- [ ] VerifyarchiveSHA/members, independently recompute saved predictions and fit selections.
- [ ] Report all layer/family scores, actualFP and pergroup differences, frozenencoder checks and limits.
- [ ] Decide which next branch evidence supports; do not start encoder retraining automatically.

No Git checkout is available; commits/worktrees cannot be used. Isolation uses uniquely named audit directories and immutable source manifests.
