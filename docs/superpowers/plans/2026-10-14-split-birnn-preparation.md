# Split BiRNN Implementation Preparation Plan

> For agentic workers: execute inline with superpowers:executing-plans. No formal fivefold training is authorized in this preparation stage.

**Goal:** Incorporate the user's review, implement and verify the data/model/streaming contracts, and measure real CUDA throughput without launching formal training.

**Architecture:** Isolated MD-only package, immutable preparation outputs, grouped source inventories, batched GRUs for training, persistent forward state with timestamped window corrections for inference.

**Tech Stack:** Existing PyTorch/NumPy, server h5py, local interview pytest; no dependency installation.

**Spec:** `docs/superpowers/specs/2026-10-14-split-birnn-reproduction-design.md`, amended by the user's pasted review of 2026-10-14. The review's four gates and preparation-only scope override the previous training proposal.

## Global Constraints

- Input MD is `[T,128]`, normalized independently per frame, no signed-log.
- Nine classes, labels -100 ignored only for supervision; real unlabeled frames still advance GRU state.
- Training batches use one shared random length in 100-1000 and start from zero GRU state; no training padding.
- Forward state continues across streaming windows. Backward GRUs start at zero for each window; the shared classifier is unique.
- Global Dice is primary; per-class Dice and CCE-only are explicitly named sensitivity options, never selected on outer test.
- CNN trainable and buffer counts separate; target 8301 is evidence to investigate, not a licence to insert unused parameters.
- No claim of exact Hospital protocol or real action latency without primary evidence.
- Preparation may run synthetic backward passes, real-data single steps, and timed throughput probes; no formal fitting or fivefold run.
- No Git repository is present; use isolated package and server capsule, no Git commits or worktree simulation.

## Review Focus

- Timing mapping must not manufacture calibrated clocks or zero unlabeled prefixes from an assumption.
- Fold source identity must group sensors and sessions and account for all selected records exactly once per outer fold.
- BatchNorm must never ingest padding; equivalence checks run in eval mode.
- Uncorrected frames and overlapping corrections must be retained in both first and final prediction views.
- Formal-training readiness must remain false if any data, structural, streaming or budget evidence is unresolved.

## Tasks

### Task 1: Source and temporal audit

Files: create `src/continuous_mmwave/split_birnn/data.py`, `audit.py`, related tests; source files under `outputs/split-birnn-preparation-20261014/sources`.

Interfaces: `merge_labels`, `standardize_frames`, `align_commands`, `audit_labels`, `build_folds`; raw audit CLI uses lazy h5py and reads only MD and CSV.

- [x] Verify review against original equations and existing converter; update spec.
- [x] Write tests for known nine-class mapping, zero variance/NaN, nonzero recording origin, ignored prefixes, command coverage, preserved frame count and grouped folds.
- [x] Run these tests and confirm missing-module RED.
- [x] Implement contracts and immutable MD-only audit; run tests GREEN.
- [ ] Read official technical README if obtainable; retain exact bytes/source hash and explicit Hospital status.
- [x] Run server audit over all 77 sessions, source shapes/values and CSV. Generate per-record counts, temporal assumptions and train-source MD/label diagnostic plot arrays; retain unknown clock and action reaction delay. Rendering uses existing local matplotlib because server lacks it.

### Task 2: Structural and loss correctness

Files: create `model.py`, `loss.py`, `tests/test_split_birnn_model.py`.

Interfaces: `SplitBiGRU.forward_sequence`, `forward_chunk`, `correct`; `BiGRUBaseline`; `combo_loss`; `parameter_audit`.

- [x] Write failing tests for forward whole/frame/chunk equality, future causality, cached backward equality, exact loss smoothing and backward-only gradients into CNN and each forward GRU.
- [x] Implement batched GRUs and a dimensionally consistent figure interpretation; count CNN and GRU parameters and BN buffers independently.
- [x] Run complete structural tests, verify shared head and 64-float cache, record any CNN mismatch as unresolved rather than silently declare exact fidelity.

### Task 3: Streaming correction semantics

Files: create `streaming.py`, `tests/test_split_birnn_streaming.py`.

Interfaces: `correction_windows`, `replay_split`, `replay_baseline`, results containing first/final logits, correction counts/endpoints and waiting delays.

- [x] Write tests against an independently constructed simple window reference with L=5,H=2, startup, overlap, short input, ignored labels, uncovered tail and recording reset.
- [x] Implement causal replay with no hidden tail flush, first/final prediction views and explicit correction events. This is offline replay of online availability, not physical edge/cloud networking.
- [x] Run tests and validate baseline uses identical schedule while recomputing standard BiGRU within every window.

### Task 4: Native verification and throughput

Files: create `probe.py`, preparation config, final preparation report and progress ledger.

- [x] Local interview tests and relevant regression tests pass; archive code hashes and transfer a fresh server capsule. Local51passed3skipped; full suite separately blocked at existing h5py import, not declared passing.
- [x] Server native tests pass, audit outputs independently validated, and real train-source MD CUDA backward step succeeds. V2 native27passed; CUDA equivalence also checked on real source.
- [x] Time synthetic lengths 100/512/1000 with warmup and CUDA synchronization, report peak memory and device metadata; record seeds/inputs and distinguish throughput probes from training.
- [x] Propose step-based budget from measured times. G3 remains pending until formal validation protocol/budget frozen and G0/G1 blockers resolved.
- [x] Perform one fresh whole-change review, fix material findings, rerun affected tests. All5findings reproduced RED natively, fixed in one V2 pass, GREEN27.
- [x] Report G0-G3 and Level1-3 honestly; do not launch formal training.

Unresolved source retrieval: official public HTML acquired, directory endpoints401 and browser unavailable. Hospital mapping and exact clocks remain unconfirmed. This preparation stage completes available evidence/software checks without treating source retrieval or all four formal gates as completed.
