# SAEF-MVP completion implementation plan

Authority: user selected `计划/想法六/最小可行架构.md`. Continue the existing
2026-10-28 implementation in place; no Git checkout exists here.

Constraints: full natural recordings, frozen continuous_hidden, supervised
boundaries with unknown masks, train-only calibration, no LLM, no APK actions.
Tests run with interview Python and fresh workspace-local pytest temporary dirs.

1. Reproduce existing suite. Add regression tests for missing decision report,
   checkpoint annotation/split identity, incompatible time grids, and recordings
   without reliable labels. Watch failures before changing implementation.
2. Add focused reporting module: all boundary tolerances and event IoUs,
   signed/absolute delays, misses, unknown burden, oversegmentation, per-class
   event diagnostics, training-calibration recall/FP curves, conservative G1-G4
   candidate checks. Synthetic runs never establish research gates.
3. Bind replay to recording metadata and time basis; reject inconsistent grids;
   train only on eligible recordings while retaining all streams for evaluation.
   Audit full source/time/supervision metadata; fix nominal SVG duration.
4. Run new and existing event-gating regression tests plus fresh CLI smoke;
   review changes and update usage docs with current evidence and limitations.

Review focus: zero valid labels, zero GT boundaries, changed annotations on
identical features, mismatched configured dt, synthetic research claims.

Progress: baseline 30 passed / 1 failed (missing go-no-go.json). Default sandbox
temp is unwritable; explicit --basetemp under tmp resolves fixture errors.

Tasks 1-3: complete. Missing report, eligible-label sampling, grid mismatch,
oracle reset history, per-class event scores and audit schema each reproduced
RED then GREEN. Checkpoint annotation mutation regression passes after identity
binding. Task 4 verification: 64 passed (MVP plus event_gating data/model/metrics/
runner), CLI smoke exit 0 at outputs/saef-mvp-smoke-20261029-v1. Real audit
stopped before training: locked server feature manifest unavailable locally.

Ruling: G1 uses preselected operating points with absolute recall gap <= .02;
G3/G4 have no numerical acceptance margin in the spec, so remain review-required.
No synthetic run or single subject validation automatically establishes Go.
Ruling: no Git exists; no commits/worktree operations performed. Old outputs
and the initial failed test evidence remain intact.

Review complete. Source timing conflicts are rejected before cache access.
Training and audit reject calibration splits with no annotated internal boundary.
Both defects reproduced RED and passed GREEN. Final verification on 2026-11-04:
66 tests passed in interview, CLI smoke exit 0 at outputs/saef-mvp-smoke-20261104-v1.
