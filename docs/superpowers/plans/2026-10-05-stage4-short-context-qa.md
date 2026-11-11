# Stage4 Short Context QA Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, TDD and one finite final review.

**Goal:** Execute approved fit/dev diagnosis,500additional QA-adapter updates, matched uncompressed/L1 evaluation.
**Architecture:** Reuse frozen frontend/cache/chat/QAadapter, add deterministic chronological short-task builder and size-specific sampler/evaluation. Only QA-adapter optimized, source compressor evaluated frozen.
**Tech Stack:** Existing PyTorch/Qwen/yaml; interview pytest locally, seer Python remotely.
**Spec:** docs/stage4-short-context-qa-2026-10-05.md, human-approved detailed in-chat plan.

## Global Constraints

TRAINsubject02/sensor14xx only, same295events/cacheSHA; source QAinferenceSHA0c2f5261...a4d12cc, compressorSHA3ff452...8e507. Exactly250/125/125updates bylength1/2/4, total500additional; lr.0003 seed42 scale256. No teacher/P/compressor mutation, no credentials in files, no dev selection, no automatic compressor training. NoGit checkout; isolated server deployment and hashes replace Git operations.

## Tasks

- [x] 1. RED tests: grouping chronology/tails/cohorts, matched pair targets, deterministic250/125/125sampler and per-bucket coverage, gradients only QA adapter, sequence/ordinal metrics and compressor-frozen matching.
- [x] 2. Implement parrad_short_context_tasks.py and parrad_short_context_qa.py, newconfig; targeted/regression tests; one finite whole-change review.
- [x] 3. Isolated upload/server tests, fit+dev initialgenerations,500updates,afterdev/no-memory/swapped/L1matched evaluations. No duplicate workers.
- [x] 4. Independent source/step/checkpoint readback, compact raw archiveSHA/member checks, recount and Chinese report/ledger/README.
