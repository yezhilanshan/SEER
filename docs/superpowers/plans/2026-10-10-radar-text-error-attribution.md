# Radar-to-Text Error Attribution Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; implement and test inline, then request one independent read-only review.

**Goal:** Deliver the user's Stage 1 systematic error-attribution report, reusing existing stage1/align weights and freezing Qwen.

**Architecture:** Separate timestamp/statistical utilities, frozen extraction/probes, and caption controls. New isolated server capsule and output directory; original weights/data are read-only. All artifacts retain source/checkpoint hashes, split roles and diagnostic definitions.

**Tech Stack:** Existing PyTorch/NumPy/Transformers/Matplotlib; no environment changes.

**Spec:** User's 3.1–3.5 checklist in this conversation (2026-10-10).

## Global Constraints

- Train subjects02/03/04; validation05; already observed test06 only descriptive, never threshold selection.
- No original model updates. Only linear heads and 8x896 fixed prefix are optimized.
- Frame evaluation uses nonoverlapping timestamp-based 12s chunks, includes final short chunk, all valid labels, once per original frame. BiGRU context is per chunk, not entire recording.
- Steady mask excludes +/-0.3s from full-recording label-transition midpoints, including transitions to unknown. Unknown frames are never scored.
- Extract stage1 CNN/GRU and align CNN/GRU, 256D each; fixed seed17, train-only standardization, identical training indices and balanced CE, linear256→14, AdamW lr1e-2, 20epochs, batch4096. Select each head by validation macro-F1, keep full histories. Bed classes1/2/3/4/5/7/8; short=GT run<2s, frame macro-recall over classes present in short subset, report support.
- Smoothing 3x3 candidates median seconds0/.3/.5, minimum duration0/.3/.6; timestamp neighborhoods, fixed class-ID median rule, drop short predicted segments then merge; no GT action-count constraint. Choose by validation sequence_exact, then order_lcs, then recall, then simplest candidate.
- Language-only baseline: original chat prompt and loss/decode, frozen Qwen, 8 learnable constant prefixes, no radar or label input; same2187 captions, 3epochs,batch2,accum4,lr1e-3,seed17; choose validation loss. Greedy identical inputs imply one constant generated caption.
- Controls: validation evenly spaced64 and descriptive test64; same-sensor/duration (within one frame period) distinct-sequence mismatch; synchronized full-frame RD/MD permutation with seed17. Fixed target loss, free generation and own-radar references separate. Search real same-multiset/different-order pairs; deduplicate synchronized sensor views by subject/session/frame range, report scarcity and overlap, never invent.
- GT instances record full-record run and window-visible duration, relative interval, left/right/both truncation, subject/sensor/environment. Class-occurrence absence uses multiset allocation; repeat-instance identity marked ambiguous. No claim of reliable temporal substitution alignment; deterministic40-window manual-review queue.

## Review Focus

- Irregular timestamps/unknown gaps: reject invalid spacing, never bridge unknown in templates.
- Boundary runs extending across crop edges: use full recording labels for duration and truncation.
- Multiple occurrences of same action: multiset coverage with explicit identity ambiguity.
- Mismatch controls: same sensor/duration, different actions, don't confuse original and substitute targets.
- Test leakage: smoothing/probe/prefix selection strictly validation, distinguish validation diagnostics from independent evidence.

## Tasks

- [x] 1. Add failing tests for statistics, timestamp masks, smoothing, instance attribution and matched controls; implement `radar_text/diagnostics.py`, verify in interview environment.
- [x] 2. Add frozen feature hook tests and constant-prefix gradients tests; implement `radar_text/diagnostic_runner.py` and `radar_text/diagnostic_controls.py` without altering original model interfaces.
- [x] 3. Package source/tests and fixed settings, hash verify server capsule, run native regression tests and launch detached single worker with status/logs.
- [x] 4. Complete full-record classification + probes + smoothing + fixed prefix + mismatch/permutation/real reverse-pair diagnostics; preserve raw arrays/JSON/plots and failures.
- [x] 5. Download/hash verify results, inspect confusion and examples, review code/results independently, produce bottleneck report with limitations and an explicitly text-only sample audit. Human radar/time-alignment review remains pending, with 40-window queues per split.

## Ledger

2026-10-10: User supplied explicit diagnostic design and deliverables; execute it directly. This workspace has no Git metadata; use isolated versioned source capsules instead of inventing commits/worktrees. Existing pilot source/weights remain unchanged.

2026-10-11: Red tests observed for missing diagnostic utilities/models, then 9 new local tests passed in interview. Original + diagnostic server tests53 passed,1 warning. Source capsule119 files verified, SHA970f9a741b7386d5c608c3daedc0f85f71874ce1d35e81f3fab3cad0734053ad. Single detached launcherPID14394 started00:08:02+08:00, output `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_diagnostics_20261011_v1`. Original pilot hashes locked in status.

Ruling: Metadata environment_id is only `parrad`, and label_policy is `command_active_until_next_command` — preserve these values, mark environment unknown beyond generic dataset ID, interpret duration as labeled-command interval rather than physical action execution time. No fabricated environment mapping or timestamps.

Ruling: Server lacks matplotlib — generate plots locally using existing interview matplotlib; do not install dependencies. All original model interfaces and weights unchanged; frozen features tapped by temporary GRU pre-hook.

00:09:50 extract/probes/smoothing/misses complete; fixed prefix running. Validation all/steady macroF1 .294969/.302635; stage1CNN/GRU .188039/.474702, alignCNN/GRU .199803/.505464. Smoothing selected .3s median,0s minimum by validation, test sequence_exact stays54/484. Short labeled runs only40validation/50test frames, too small for stable conclusions. Independent review has found no startup-blocking issue so far.

00:22:10 all6 stages completed. Collected49 original artifacts and independently verified hashes and six14-class confusion matrices from frame arrays. Generated13PNG/13SVG plots plus per-class CSV locally. Final related local regressions48passed,1skipped; original server capsule53passed. No original model/data edits or dependency installation.

Post-run audit: terminal-frame-only differences between synchronized views inflated validation physical window keys352vs351. Added a failing regression and fixed local real_order_pairs deduplication; all52/8 eligible pairs and evaluated predictions are unchanged. The sealed server source and raw archive remain intact; derived correction is explicit in real-order-dedup-audit.json.

Final independent read-only review found no blocking issues and verified counts/interpretation. Added its caveat that out-of-vocabulary claims (table/stairs) escape14-class extra-action counts. Reviewed16 outputs as assistant text audits only, never human physical alignment. Final report docs/radar-text-error-attribution-2026-10-11.md records evidence, confounding and pending human review.
