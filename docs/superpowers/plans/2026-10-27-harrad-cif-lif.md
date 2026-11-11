# HARrad CIF/LIF Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; execute in this session with TDD and one independent final review.

**Goal:** Deliver an additive HARrad event-gating package with training, evaluation and bounded streaming inference.

**Architecture:** Freeze and adapt the existing HARrad CNN/GRU without editing it. Train new CIF/LIF gates on reconstructed multi-instance sequences; evaluate natural recordings separately with partial-label metrics.

**Tech Stack:** Existing PyTorch, NumPy, standard library; tests in C:/User/anaconda/envs/interview/python.exe.

**Spec:** ../specs/2026-10-26-harrad-cif-lif-design.md (approved by user on 2026-10-27).

## Global Constraints

- Add new package, tests, configuration and documentation; preserve existing model and runner source files.
- HARrad train subject1–6, validation subject7; test subject8–9 signals are not accessed.
- Nominal 15 FPS; unknown intervals are not background; preserve repeated instance labels.
- Never concatenate cached recurrent hidden states to synthesize a sequence; replay the original frozen GRU on concatenated CNN features.
- No LLM, APK changes, dependency installation or real-data performance claims from synthetic tests.
- Workspace is not a Git repository; use source hashes to prove existing model files remain unchanged, no worktree or commits possible.

## Review Focus

- Missing/changed source arrays and checkpoint provenance must fail before training.
- Adjacent same-class instances and insufficient CTC paths must not disappear.
- Chunk boundaries, EOF and long recordings must not lose frames or grow inference state.
- Natural-stream evaluation must distinguish unknown and mixed predictions from fully known false positives.
- Reloading best and stage transitions must preserve architecture and split provenance.

### Task 1: Integration-independent gate mechanics

Files: new event_gating/config.py, cif.py, lif.py; tests/test_event_gating_core.py.

Interfaces: cif_integrate(features, alpha, target_count=None) returns weighted tokens and residual mass; LIFController.step(drive, dt, state) returns membrane, threshold, hard/soft spike and next state.

- [x] Write analytical CIF split/conservation, target scaling, leaky integral, zero-drive, reset and gradient tests.
- [x] Run tests and observe missing-feature failures.
- [x] Implement mechanics and strict input validation; rerun tests to green.

### Task 2: Model and bounded stream

Files: new event_gating/model.py, streaming.py; tests/test_event_gating_model.py.

Interfaces: EventGate(config, mode); model.training_loss(features, targets, teacher=None); EventStream.push(features), finish() yield event dictionaries.

- [x] Write tests for causality, chunk equivalence, padding/length handling, repeated-target CTC, zero hard-spike gradients, EOF coverage, bounded memory and retroactive frame ownership.
- [x] Observe failures, implement shared gate step/state and CIF/LIF streaming, verify independently computed event means and boundaries.

### Task 3: HARrad adapter and partial-label evaluation

Files: new event_gating/data.py, metrics.py; tests/test_event_gating_data.py, test_event_gating_metrics.py.

Interfaces: prepare(config, output, device); load_prepared(path); sequence_metrics(predicted, target); score_recording(events, instances, frames).

- [x] Test role isolation, hashed assets, invalid intervals, GRU replay, same-label identities and stable sample IDs.
- [x] Test matching with unknown/mixed intervals, cardinality/IoU ties, count and sequence errors; observe RED.
- [x] Implement immutable preparation and independent stream/constructed datasets; validate all new tests.

### Task 4: Train, checkpoints, CLI and documentation

Files: new event_gating/runner.py, __init__.py, __main__.py, configs/harrad-cif-lif-v1.json, docs/harrad-cif-lif.md; tests/test_event_gating_runner.py.

Interfaces: prepare/train/evaluate/infer/smoke subcommands; explicit stages cif/lif-distill/lif/frame-baseline; immutable run dirs, best.pt and per-record predictions.

- [x] Write end-to-end CPU tests for optimizer updates, stage provenance, readback, missing real data, and CLI smoke.
- [x] Implement train/evaluate/checkpoints and source-locked configuration; keep test assets separate from real HARrad assets.
- [x] Run new suite, existing HARrad regressions, independent final review, full pytest command and CLI smoke (full-suite limitations documented).
- [x] Fix important review findings with regressions, verify old-source hashes, document results and remaining real-data limitations.
