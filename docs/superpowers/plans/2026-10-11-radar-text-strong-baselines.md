# Matched Sliding-Window Classification-to-Text Baselines

> Use executing-plans inline, tests before implementation, one final independent read-only review.

**Goal:** Compare B0 original head, B1 primary H1, B2 primary H4 and B3 original Radar-to-Text on identical freshly loaded sliding windows and sensor inputs; no new training.

**Spec:** User's Experiment1 B0–B3 comparison and identical12s window/input, validation-only smoothing constraints.

## Fixed decisions

- Original align settings: nominal12s windows, stride6s, minimum8s, max3 GT runs. Includes702validation/484test windows, each split has2 partial tails. Report complete nominal12s subset700/482 as the primary strict-window result, and all original windows for the exact25% pilot comparison. Allow one frame period for sampling quantization; no invented or merged windows.
- Fresh Stage1 encoder forward in eval/no_grad on every original sliding window; never slice prior nonoverlapping features. B0/B1/B2 share the exact same hidden tensor. H1/H4 use original_optimizer selected heads from previous ablation, not probe_replay or any test-selected head. H4 uses its saved training mean/std.
- Run B3 anew using frozen original Align + original Qwen, same ordered windows and same batch2 RD/MD/length tensors as B0–B2. Same max_new_tokens96, greedy, cache, prompt/tokenizer. Assert IDs/GT/data metadata agree with prior manifests and check generation reproduction.
- Classification smoothing function unchanged: class-ID median0/.3/.5 seconds, min predicted duration0/.3/.6 seconds, merge adjacent equals, no forced GT count. Class-ID median has arbitrary order limitation; keep explicit.
- Select one common threshold from complete validation windows only, maximizing average B0/B1/B2 sequence_exact then LCS then recall, tie simpler. Lock selection JSON before test inference. Additionally each method gets an independent validation-selected threshold from the identical grid as a stronger-baseline sensitivity; export unsmoothed results too. Never run threshold selection on test.
- Export paired per-window references/predictions/text, timestamps/offsets/labels and all3 logits. Fresh GRU feature caches and raw radar remain server-side; export feature/input hashes for audit. B3 input IDs/hashes originate in same batch loop.
- Check original Stage1/Align/H1/H4/Qwen/manifest/phrase hashes before/after. Compare original B0 logits/predictions and rerun B3 with prior results; don't conceal any reproduction deviation.
- Metrics: exact sequence, multiset action recall/precision, normalizedLCS, output count deficit/excess, per14-class occurrence recall, sensor and GT action-count groups, paired exact win/loss. These remain restricted phrase proxies; no claims of general linguistic/fine-grained grounding. Overlap/dual-view correlations, observed test subject06, single independent heldout subject remain limitations.
- No fitting, installation, new large-model training or modification of original weights/data. NoGit workspace: immutable versioned source capsules and outputs instead of commits.

## Tasks

- [x] 1. Red tests for strict ID/GT pairing, validation-only common/individual threshold selection, timestamp smoothing without GT count, saved head standardization; implement shared utilities and verify.
- [x] 2. Implement audited fresh-feature and B3-rerun worker, isolated source capsule, native tests and detached server execution.
- [x] 3. Complete extraction/generation, verify reproduction/input/weights, collect immutable artifacts, compute paired comparison.
- [x] 4. Render report, complete14-class CSV and figures, independent read-only review, final hash/metric verification and docs links.

## Ledger

2026-10-11: Inspected prior protocols and artifacts. Original evaluation windows702/484 include two partial tails each (94/95valframes;91testframes). Primary strict-window result700/482 plus original protocol702/484 avoids silently calling all windows full12s. H1/H4 previous fixed-cache training remains fixed; only evaluation is re-extracted under caption sliding windows.

Task1: observed4 missing-module failures then4passed; related native interview18passed1warning. Task2: source capsule125files SHA d88109fbed61ba8c4cb3417a2b9beada37b759d3b4938e981a106decb5d0bf34; server62passed1warning. Single detached worker PID5084 started12:01:54+08:00 at /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_strong_baselines_20261011_v1. No parameter training. Validation702 windows freshly inferred; validation-only smoothing selection running. Independent read-only code review found no blocker, pending results review.

Task3 completed12:09:39+08:00: validation702/test484 regenerated; complete nominal12s700/482. Common threshold median0/min0.6s locked12:06:01 before test; independent H1/H4 same, B0 median0.3/min0. Original B0 frame argmax differences0, maxlogit1.43e-6; B3 text/action differences0. All24 original input hashes unchanged;125 source files verified; zero trainable parameters. Result archive25,595,755bytes SHA ec58a5da444e01e1ecca39c40968c09dfebf97a652b00241ad4109566e379023,16 raw files verified locally. Fresh features remain server-side.

Task4 complete: report docs/radar-text-strong-baselines-2026-10-11.md, PNG/SVG and560-row14-class CSV, README/offline guide links. Independent read-only review and follow-up found no blocker. Local independent audit passed16 raw SHA,11,860 per-window main-score rows,10,674 classification sequences rebuilt from exported logits/timestamps,40 aggregate summaries,27 validation candidates; B3 texts match original. Report/collector/verifier are post-inference local utilities, outside sealed inference capsule. Derived manifest hashes these utilities and report/CSV/plots/audit outputs separately; original raw outputs remain immutable.

Conclusion: validation-selectedB1 full12test86/482=17.84%, B3 119/482=24.69% (gap6.85pp); original484 B1 88/484=18.18%, B3 121/484=25% (gap6.82pp). B2 is parallel comparison89/484=18.39%, gap6.61pp, not test-based method selection. B1/B2 recall~60% andLCS~47% exceed B3~46%/~45%; exact-sequence advantage persists but narrows, no overall generation superiority or isolated Qwen causal claim.
