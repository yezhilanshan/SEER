# PARrad Label-Run and Streaming Training Specification

## Objective

Replace fixed, label-agnostic Stage 1 training windows with clean label-run windows, then fine-tune the same encoder on continuous sequences with truncated backpropagation through time (TBPTT). Validation and test must never use ground-truth label boundaries to choose windows.

## Required behavior

- Label-run training discovers contiguous runs from the original frame labels.
- Frames within the configured transition radius are removed from the run rather than retained as `-100` inputs.
- Each retained run is split into non-overlapping samples of at most `window_frames`; short tails remain variable length and are padded only by `parrad_collate`.
- Every real frame in a label-run sample has the same supervised label.
- The existing fixed-window mode remains available for comparison and for continuous sequence construction.
- A streaming fine-tune phase consumes longer contiguous sequences, processes them in `window_frames` chunks, carries both TCN caches and the global update index across chunks, and detaches the carried state at each TBPTT boundary.
- Boundary-masked frames in the streaming phase remain real inputs (`frame_mask=True`) so that state evolution matches inference, while `labels=-100` excludes them only from the loss.
- Validation and test operate on continuous recordings, carry state across chunks, and use labels only for loss/metrics, never to choose chunk boundaries.
- The production configuration uses training batch size 16 and preserves the previous fixed-window mode through explicit configuration.
- Label-run samples shorter than `min_label_run_frames` are excluded; long runs are partitioned without leaving tiny cold-start tails.
- Streaming fine-tuning carries state across every sequence from the same recording and resets only at a recording boundary.
- Label-run and streaming phases have independent base learning rates with warmup and cosine decay.
- Early stopping is phase-local, and the best checkpoint from each phase is retained separately.
- Continuous evaluation records macro metrics, per-class recall, per-sensor accuracy, raw-label accuracy, and segment-majority accuracy in addition to trimmed frame accuracy.

## Non-goals

- Inferring action boundaries at deployment time.
- Adding a segment-level classifier or changing the 14-class label vocabulary.
- Changing the PARrad canonical manifests or split membership.
