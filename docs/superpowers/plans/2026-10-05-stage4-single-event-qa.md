# Stage4 Single Event QA Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; tests first and one whole-change review.

**Goal:** Run the authorized single-event controls and one fixed500update independent QA-adapter pilot.
**Architecture:** Frozen original frontend/P/LLM and cached L0; new residual896/256/896 QA-only adapter. Same-question opposite-event supervision plus closed-set activity identification. No memory compressor updates.
**Tech Stack:** Existing PyTorch/Qwen, yaml, checked Stage4 cache, pytest in interview locally.
**Spec:** docs/stage4-single-event-qa-2026-10-05.md; design already approved explicitly by human in chat.

## Global Constraints

Source cache8da699...77fb and protected compressor3ff452...8e507; TRAINsubject02/sensor14xx,set000/001fit,set002dev. Exactly500actual AdamW steps, seed42, lr.001, clip1, scale256; checkpoint100/250/500, reportfinal. No original model, retrieval key or threshold change. No credentials in files. NoGit metadata; isolated server code/hash provenance replaces worktree/commit operations.

## Tasks

- [x] 1. Write failing tests in tests/test_single_event_qa.py for same-query/opposite-source pairs, no cohort leakage, confusion negatives, identity adapter/input preservation, gradients through frozen FullLM, class/paired metrics. Observe failure.
- [x] 2. Implement models/qa_adapter.py and parrad_single_event_qa.py; config configs/parrad-stage4-single-event-qa-v1.yaml. Verify targeted and available regression tests. One finite whole-change review.
- [x] 3. Upload isolated code, server tests, source validation, baseline native/text/no-memory, train500steps with balanced pair+identify CE, evaluatefinal including swapped memory; no duplicate or extra worker.
- [x] 4. Independent checkpoints/readback/source digests, download compact raw predictions/logs with archiveSHA/path checks, recount metrics, Chinese results report and project ledger.

## Review Focus

Counterfactual pairs really share question and differ in label; split provenance is enforced, class absence is explicit; adapter only trainable, mask and causal labels match generation; baseline text is oracle-only; losses/updates/checkpoints finite, no skipped steps counted; all prediction evidence preserved, no dev best-checkpoint selection.
