# Stage4 memory training and end-to-end pilot

Goal: implement and train the shared Query Compressor and run radar → frozen Stage2/3 → finalized events → hierarchical memory → retrieval → full Qwen inference.
Architecture: user-supplied 计划/方案书v2.5 (2).md plus inherited v2.4 memory detail. Native 896-dimensional tokens, 1024 internal dimension, two decoder layers, four summary queries. Freeze the entire existing frontend, semantic projection and LLM including LoRA.
Tech stack: PyTorch, existing Qwen2Adapter, PARrad canonical TRAIN recordings, existing controller and checkpoint loaders.

Authorized scope: user's Stage4 request supersedes the old Stage3 Go prerequisite for this engineering pilot. Do not replace production weights. Do not claim formal Go or unseen-subject performance.

## Explicit pilot choices
- Use accepted Stage2 checkpoint and fixed Stage3 checkpoint/thresholds referenced by the existing frozen trace manifest. No best-fold projection selection.
- Offline compressor supervision uses chronological GT label-run commit tokens from TRAIN only, with action-template summaries. This is annotated segmentation training, not a claim about natural online boundaries or free-form question answering.
- Fit set000/set001, development set002, subjects02/03/04. No set003, validation or test arrays. First cache uses fixed subject02 sensor14xx to reduce turnaround; extend later only with documented budget.
- Train both levels with the same module. L_summary is full frozen LLM teacher-forced caption CE; L_retain is cosine loss to the frozen projection of concatenated source event tokens. No gradients into upstream weights.
- Fixed 100 AdamW updates, batch1, seed42, lr1e-4, retain coefficient1, gradient clip1. Save last checkpoint; no selection using development results.
- L0 capacity8, promotion groups4, L1 capacity8, L2 capacity32. Compact oldest L2 group when full. Read all L0 plus L1 top2/L2 top1. Cosine + recency + confidence ranking; merge disabled in first pilot to expose promotions, explicit opt-in implementation.
- Real online replay uses unchanged controller thresholds. Final incomplete recording event may be flushed with an explicit EOF marker using shadow branch if a candidate remains. No GT boundary injection online.

## Tasks
1. Tests before implementation: compressor masks and both-level gradients, frozen-LM gradient flow, hierarchical promotions/context order/merge/update/snapshot, loss ignores prompts and padding. Run in interview environment and observe missing-feature failure.
2. Implement compressor, memory, training objective and full-LM generation interfaces; verify targeted tests.
3. Implement TRAIN-only cache builder and runner (cache/train/replay), configuration and provenance. Tests for grouping/split guards/checkpoint restoration. Real integration verifies all modules, not just cached tokens.
4. Deploy isolated server copy, run native-size training and actual streaming replay, save checkpoint/losses/memory/retrieval/generation outputs and SHA manifest. One finite implementation review plus appropriate existing regression tests.
5. Download compact artifacts and report completion, losses, actual outputs, hierarchy utilization and limitations. If training exceeds turn duration, update the existing paused monitor with the new exact paths, not the old audit.

## Review focus
Architecture: same compressor at both levels; frontend remains frozen. Interfaces: native token dimensions, dtype and causal loss offsets. Consistency: explicit source/threshold hashes and fixed last-step reporting. Completeness: actual full-Qwen generation and native radar online replay. Overengineering: no additional Stage3 sweeps or formal Go gate.
