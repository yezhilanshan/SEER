# Stage3A zero-update supervision diagnostic Implementation Plan

> For agentic workers: use superpowers:executing-plans inline with TDD and one fresh final review. Preserve all evidence; this checkout has no Git repository.

**Goal:** Explain fixed500 A/B Proposal misses, actual chunk supervision exposure and causal token/MD-proxy information, then recommend exactly one next experimental variable where evidence permits.

**Architecture:** Pure NumPy diagnostics consume verified saved predictions/selected settings; a CPU-only server exporter reads the locked26 encoder records and computes fixed causal differences. Fit fixes signal baselines before dev summaries. No optimizer, backward, model training or new trigger search.

**Spec:** User-approved five-step proposal immediately preceding “请你开始吧”; original loss-ablation plan and verified results remain authoritative source constraints.

## Global Constraints

- New AdamW updates0, model forwards0. Encoder/EventFormer/gate/P/Evidence and both500 checkpoints untouched. Exporter torch.load only, no model construction.
- Reuse original encoder index SHA205d96c8993c1570c13b0cdb0a708faa7025d094d1b974b5a8ecc5420bbbdc60;18fit/8dev. No holdout/official data access.
- Reuse all18 previously locked loss/path/cap settings SHA63636f5dff81f555566feeba8450830feae4160d248e96415eebb24bec9650ef. Weighted500 primary, unweighted500 control; all three paths and3/5/8 reported.
- Event tolerance remains±.72s at actual emission time; report peak center/lookahead separately. Offline visualization window±2.16s does not change scoring.
- Mutually exclusive primary failure stages: below_threshold, neighbor_constraint, peak_shape, invalid_context, nms_suppressed, match_conflict; matched/excluded separate. Wider-window nearby emissions are secondary tags, not proof of annotation errors.
- Fixed causal features: token mean cosine distance over common valid slots; relative flattened token L2; normalized MD level difference; cached frame-absolute-difference and frame-cosine-difference proxies. Preserve missing masks/NaN. They are radar statistics, not speed/pose ground truth.
- Signal reference is sensor-specific fit-stable(>.72s)Q95, locked before dev. No fit parameter fitting besides descriptive quantiles. AUROC/AP main hard labels≤.36 vs>.72, optional Gaussian ranking. Pairwise common support compares feature with eachHead; weak simple differences never prove representation information absent.
- EachGT alignment window returns strongest valid signal above fitQ95 or NA; compare fixed stable control centers≥2.16s fromGT. Exclude overlapping GT windows from isolated timing claims, report overlap flags/all-boundary sensitivity. No new label shifts or widened recall claim.
- Replay actual500×16 draws. Report valid count N, Gaussian mass M, per-chunk A denominator aM+b(N−M), B denominatorN, per-update1/Z, actual repeats and boundary coverage. Half/full chunks both explicit; frozen eval-loss proxies are not training gradients.
- Exclusive directories, source/SHA manifests, safe member checks before extraction, native and independent local audits. Raw evidence retained; no automatic next training or production installation.

## Review Focus

- Decision time vs peak center and finite-lookahead tail must not create false match/miss categories.
- Invalid slots/zero-norm tokens and disabled MD attributes must stay missing, never fabricated zero evidence.
- Adjacent GT windows and old greedy matches can make nearby peaks non-identifying; timing observations are not physical GT.
- Variable chunk lengths/valid masks and repeated draws alter exposure; bucket balance must not be mistaken for per-boundary balance.
- Dev must not choose sensor quantiles/features/thresholds; compare paths on identical support and retain all failures.

## Task1: causal failure trace and exposure

- [ ] Write failing tests for delayed peak matching, neighbor rejection, plateau, invalid context, NMS and greedy-match conflict; trace emissions must exactly equal old emit.
- [ ] Implement .audit-staging/proposal_supervision_diagnostic.py core functions trace_peak(time,score,valid,config), classify(time,score,valid,boundaries,config), chunk_mass(target,valid,start,end,a,b), token_features(tokens,mask), alignment(time,values,valid,boundaries,threshold).
- [ ] Test in interview; preserve scalar replay and no-input mutation tests.

## Task2: verified CPU feature export

- [ ] Test masked/zero-norm token differences and chunk denominator/repeats semantics before implementation.
- [ ] Export .audit-staging/export_proposal_supervision_features.py: SHA-check index/caches, write NPZ(time,labels,boundaries,transition_from/to,target,valid,5features+5masks), per-record source metadata and archive manifest. No torch model/optimizer code.
- [ ] Lock source package/spec before execution; native tests, exclusive output. Download all exported feature arrays/metadata/logs and check SHA/member paths. Failure and network interruption do not authorize duplicate exporters.

## Task3: fit-lock, complete analysis and recommendation

- [ ] Lock per-sensor featureQ95 onfit-stable; then readdev summaries. Produce all boundary-ledger rows, all group/category tables, all500draw chunk rows, per-boundary repeated exposure, paired common-support ranking and isolated alignment/stable controls.
- [ ] Verify every classification partitions original misses and emits exact old events. Independently verify original draws/labels/timestamps and feature validity. Scientific plots show representative cases selected onfit rules, plus worst group sensitivity without selecting success only.
- [ ] One fresh read-only review of whole diagnostic source, fix consequential findings with RED→GREEN; report code/data limitations without infinite auditing.
- [ ] Save docs/stage3a-proposal-supervision-diagnostic-results-2026-10-08.md, rawCSV/JSON/NPZ, source SHA, and one evidence-based next500-step A/B proposal; do not launch it.
