# SDD ledger — plan: docs/superpowers/plans/2026-10-27-harrad-cif-lif.md

Ruling: User approved implementation and requested preservation of old structure; execute directly without another plan approval. The workspace has no .git, so add files in place and hash existing sources rather than creating a worktree or committing.

Pre-flight: core -> model -> stream/data -> runner use a new isolated namespace. Existing HARrad adapters are read-only imports, all new behavior lives in event_gating.

Task 1: complete — 7 analytical gate tests RED→GREEN in interview. No existing sources edited.
Task 2: complete — causal state, bounded buffers and streaming tests RED→GREEN; core+model 14 passed before baseline addition.
Task 3: complete — data/provenance and partial-label metrics tests RED→GREEN, 8 passed.
Task 4: complete — source-adapter integration, four-stage smoke, checkpoint/resume/provenance, review regressions RED→GREEN; final new suite 34 passed. Full-suite environmental limitation is recorded below, not claimed green.

Ruling: Legacy HARrad checkpoints contain NumPy normalizers; load those files with weights_only=False only after verifying their trusted SHA. New event-gating checkpoints use weights_only=True. This avoids changing the old checkpoint format; callers must retain the supplied source locks.

Final review: fresh read-only reviewer found 4 Important issues. All reproduced by new tests RED→GREEN: long CIF rounding; config/prepared source mismatch; resumed current-vs-snapshot best; unverified feature inference. Updated full new suite: 34 passed. Adjacent same-class recall upgraded from review minor to spec requirement and implemented, no deferred minor.

Review rulings: real-data/GPU/server validation is explicitly unperformed, not a completion claim; teacher uses internal event endpoints by approved design; optional retrospective inference remains an explicitly disclosed training surrogate mismatch with default retrospective=False.

Validation: original HARrad regression 69 passed, 4 skipped. CLI four-stage synthetic smoke v2 passed with real_harrad_training=False. 33 original HARrad source hashes unchanged. Full pytest blocked at collection by missing h5py; supplementary full execution stopped after prolonged lack of progress at 6%, log preserved. No full-suite-green claim. No code fixes remain from the independent review.
