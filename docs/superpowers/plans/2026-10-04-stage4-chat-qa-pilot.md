# Stage4 chat and evidence QA pilot

Goal: continue the authorized next step: align Stage4 training with mixed L2/L1/L0 inference context and chat prompts; add evidence-grounded QA and assess actual free generation.
Architecture: retain the shared1024/two-decoder/four-query compressor and completely frozen frontend/P/Full Qwen/LoRA. Continue from the previous100step checkpoint; no new upstream fitting.
Tech stack: existing PyTorch/Qwen/Stage4 event cache, native frozen semantic retrieval and chronological hierarchy.

User-approved bounded change: previous reply proposed matching mixed contexts/chat prompts and adding QA supervision; user asked to continue. Implement inline without a new approval cycle. NoGit checkout metadata; isolated deployment plus SHA provenance.

## Locked experiment choices
- Same295 cached TRAIN GT offline events, fit set000/001 and dev set002 forsubject02/sensor14xx. No new validation/test/set003 arrays. Existing cache/frontends and100step checkpoint must match recordedSHA/digest.
- Fourteen canonical activity names. For each recording, snapshots at12/24/40/56/72/88/final events; skip prefixes with no summaries. Native capacities8/8/32 and promotiongroup4, L1top2/L2top1. Build retrieval with fixed initial100step model, save immutable selected sourceIDs and labels for matched before/after comparisons. Actual online replay uses final compressor keys and unchanged controller thresholds; report it separately.
- Each snapshot gets chronology, first/last activity, and fourteen Yes/No presence questions. Answers come only from the selected memory evidence; no activity annotations in the model input. AllL0 tokens included, selected L1/L2 summaries differentiably reconstructed with current shared weights. Recompute per-summary semantic retention against original frozen source-group key.
- Shared chat prompt tokenizer for train and generate. Only assistant answer andEOS carry CE. Loss=answerCE+2*mean per-summary retention. OriginalP not optimized.
- 500 additional actualAdamW updates, lr1e-4, batch1, seed42, clip1, FP16initialscale256. Balance task types (summary/first/last/positive/negative) where available. New optimizer, starting from previous compressor weights. Save100/250/500; no checkpoint selection by dev.
- Evaluate fixed dev tasks before/after with greedy free generation and assistantCE. Exact canonical answer accuracy separately for binary/order/first/last; format failures counted. Paired presence questions with opposite evidence and zero-memory control distinguish constant answers. Retention cosines reported separately; no inference from CE alone.
- Replay real radar and original Stage3 thresholds with final model, consistent questions; no GT online boundaries or text captions injected. Original frontier weight digest before/after unchanged. Still TRAIN development, not formalGo or unseen-subject benchmark.

## Tasks
1. Failing tests for shared chat prefix/mask/EOS, evidence-only targets, hierarchical context reconstruction and gradients, free-generation grading/opposite-memory pairs.
2. Implement focused parrad_stage4_qa.py and shared chat helpers. Verify targeted tests and available existing regressions; one finite review for concrete implementation failures.
3. Deploy isolated code, validate source artifacts, build task manifest, train500updates, evaluate both free generation and controls, run real streaming replay. Preserve all failures and avoid duplicate workers.
4. Independent step/weight readback; download compact JSON/log with archiveSHA/member validation; write Chinese result report and update execution ledger.

## Review focus
Architecture: frozen P/frontends and same shared compressor. Interface: chat prefix identical and correct assistant-only causal labels. Consistency: immutable selected evidence/task targets and explicit init/cacheSHA. Completeness: actual QA outputs/control cases/real radar replay. Avoid adding new Stage3 ablations or judging ground truth from generated captions.
