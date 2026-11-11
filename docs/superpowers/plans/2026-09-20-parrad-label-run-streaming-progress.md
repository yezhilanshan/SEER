# SDD ledger — plan: docs/superpowers/plans/2026-09-20-parrad-label-run-streaming.md

Setup: native execution selected because delegation was not requested; repository has no `.git`, so commit and git-worktree steps are unavailable.
Pre-flight: Task 1 produces window modes consumed by Task 4; names match.
Pre-flight: Task 2 produces stream state APIs consumed by Task 3; names match.
Pre-flight: Task 3 produces training/evaluation functions consumed by Task 4; names match.
Ruling: continuous evaluation will batch complete recording streams and chunk only by frame count; labels are metric-only. Cost if wrong: validation throughput may be lower than fixed-window batching.
Task 1: complete (no git repository; tests: interview pytest tests/test_parrad_data.py -q -> 22 passed).
Task 2: complete (no git repository; tests: interview pytest tests/test_parrad_encoder.py tests/test_parrad_stages.py -q -> 17 passed).
Task 3: complete (no git repository; tests: interview pytest tests/test_parrad_training.py -q -> 7 passed).
Task 4: complete (no git repository; tests: interview pytest PARrad data/encoder/stages/training -> 47 passed).
Final: fixed balanced class weights counting trimmed frames incorrectly — test_balanced_weights_count_only_frames_returned_by_label_run_windows RED->GREEN.
Final: fixed duplicate training recording cache across phases — test_window_dataset_can_reuse_cached_recordings_between_training_phases RED->GREEN.
Final: fixed masked padding contaminating BatchNorm statistics — test_masked_padding_does_not_change_valid_frame_batch_norm_features RED->GREEN.
Task 5: complete (remote isolated SEER env: 50 PARrad tests passed; full non-Gemma suite 70 passed, 1 unrelated deprecation warning; CUDA label-run and two-chunk TBPTT smoke passed on RTX 2080 Ti).
Final review: self-review because delegation was not requested and the workspace has no git history.
Final: Ruling: Gemma adapter tests remain outside this work per the user's explicit instruction not to fix Gemma. Cost if wrong: the Gemma Stage 2 adapter remains unusable in the current Transformers version.
Final: minor (deferred): TBPTT training log aggregates weighted chunk losses by valid frame count rather than exact class-weight denominator; optimization itself uses the correct per-chunk weighted cross-entropy.
Final: minor (deferred): one unrelated torch CPU autocast deprecation warning remains in the non-Gemma suite.
Task 6: complete (`min_label_run_frames=8`; short runs dropped and long-run tiny tails rebalanced).
Task 7: complete (stream state now continues across 256-frame sequence boundaries and resets only between recording groups).
Task 8: complete (phase-specific warmup-cosine schedules, patience-4 phase early stopping, phase-best checkpoints, and macro/raw/segment/class/sensor metrics).
Regression: local interview environment PARrad suite -> 57 passed; remote isolated SEER environment PARrad suite -> 57 passed; remote full non-Gemma suite -> 77 passed with one unrelated deprecation warning.
Pilot: remote CUDA 3+1 run completed at `stage1_label_run_streaming_pilot_3plus1`; label validation accuracy peaked at 0.19530 in epoch 2, streaming validation accuracy was 0.18016, and global-best test accuracy/macro-F1/segment accuracy were 0.18555/0.13944/0.25890.
Pilot: observed GPU SM utilization was typically 58-78% with 7.1 GiB in label-run training and 9.4 GiB in streaming training; GPU temperature reached 88-89 C and clocks fell as low as 645 MHz, so cooling is the current throughput constraint.
Production: launched the 20+5 run in `stage1_label_run_streaming_v2_full_lowgpu` with batch size 8 (remote PID 25128). Initial 10-sample GPU readings were 31-44% SM utilization, 4.3 GiB framebuffer use, 87-88 C, and 1.34-1.73 GHz clocks.
Diagnostics: added reusable train/validation/test continuous diagnostics with confusion matrices, per-class precision/recall/F1/support, subject/sensor/recording groups, and temporal switching/boundary metrics; local PARrad regression -> 59 passed and remote training/diagnostic tests -> 15 passed.
Diagnostics: fixed validation/test construction so raw labels are preserved and boundary masking is applied only inside evaluation. The earlier `test_metrics.json` segment/raw fields were optimistic because they used pre-masked recordings; `diagnostics.json` is authoritative for this checkpoint.
Diagnostics result: trimmed train/validation/test accuracy = 0.4321/0.3582/0.3542 and raw accuracy = 0.3801/0.3148/0.3020. Raw segment accuracy = 0.6568/0.5013/0.4823. Test predictions changed class 20.81 times per true action boundary and switched inside a true action on 30.63% of adjacent supervised frame pairs.
