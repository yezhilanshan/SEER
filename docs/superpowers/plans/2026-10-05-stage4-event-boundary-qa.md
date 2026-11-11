# Event Boundary QA Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, TDD and one finite final review.

**Goal:** Run authorized matched A/B500steps each to test explicit event markers and position supervision.
**Architecture:** New task/runner modules reuse frozen source/chat/QA-adapter; immutable marker embeddings only in generation context. Shared deterministic sampler and sourceSHA, independent optimizer/initialmodel per arm.
**Tech Stack:** Existing PyTorch/Qwen/yaml; interview pytest locally, seer Python remotely.
**Spec:** docs/stage4-event-boundary-qa-2026-10-05.md and approved in-chat plan.

## Global Constraints

TRAIN295events only, original cacheSHA8da699...0c77fb; initQA SHA7e511c2...bd637 cumulative1000. Each arm500updates200/100/200 bylength, foursequence100/positions25each. lr.0003 seed42 scale256. Frozen upstream/P/LLM/compressor; no dev selection or new budget; credentials only actual password prompt. NoGit metadata: isolated code/SHA artifacts replace Git helpers; retain ledger as evidence.

## Review Focus

Numbering must not enter retrieval or the QA transform; position answers must keep repeats; single renderer exact unchanged; reset initial model and optimizer before B; no dev examples in batch; code/source/checkpoint/sampling SHA shared and read back.

- [x] Task1: RED tests for position targets/metrics, exact role/class coverage, marked source order/single parity and QA-only real gradients; implement parrad_event_boundary_tasks.py and parrad_event_boundary_qa.py, config; targeted/full regression and one finite review.
- [x] Task2: Isolated upload/SHA/native targeted tests, single sequential launcher, correct-text diagnostic and A/B before/train/after/none/swapped. Expected native statuscomplete with500updates each.
- [x] Task3: Independently read saved optimizer/export/source/task/sampler artifacts, safe compact archive and independent answer recount, report A/B outcomes/retention/limits, update README and ledger. No further training.
