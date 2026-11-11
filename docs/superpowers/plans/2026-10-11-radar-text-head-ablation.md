# Frozen Stage1 Readout Ablation Implementation Plan

> Execute inline with superpowers:executing-plans and one final independent read-only review.

**Goal:** Attribute the Stage1 readout gap using the user's H0–H4 frozen-head experiment, without updating CNN/BiGRU/Qwen or regenerating features.

**Spec:** User's current H0–H4 table and identical-cache/split/optimizer/evaluation constraints.

**Architecture:** One cache-only training module, tests, isolated server source capsule/output, and a derived report. Only 256→14 linear weights/bias are trainable. Reuse the sealed 38 feature caches and their hashes.

## Definitions and rulings

- Reference gate: load original Stage1 classifier and apply it to stage1_gru caches, zero updates. Require identical predictions/confusions to previous full/steady frame evaluation; report maximum logit difference. This is the exact original readout reproduction, named H0@0.
- An updated head on a frozen final encoder cannot reproduce the original joint encoder/head training trajectory. Distinguish H0@0 from the H0 head after a short retraining budget; never claim those are the same experiment.
- Primary panel original_optimizer: all arms start from original checkpoint head; AdamW lr.001, weight_decay.01, grad_clip1 (original settings). Raw and standardized arms start at the same input/output function using W_std=W_raw*std and b_std=b_raw+W_raw*mean. Optimizer state restarts for all arms; original optimizer state was not saved.
- Probe panel probe_replay: all arms use the previous probe random initialization(seed17), AdamW lr.01, weight_decay.01, no clipping. H4 must replay the prior Stage1 GRU probe as closely as hardware allows. This second complete panel prevents changing learning rate/initialization in just H4.
- Within each panel only three user factors vary: H0=3epochs/no weights/no standardization; H1=20/no/no; H2=3/weights/no; H3=3/no/standardization; H4=20/weights/standardization. Batch4096 frames and seed17 identical in all arms; same shuffled frame order prefixes for 3/20 budgets.
- Original training batched4 overlapping windows and updated encoder with dropout. Fixed-cache readout batches are4096 frames from nonoverlap12s eval-mode contexts, as in previous probes; this controlled diagnostic does not reconstruct original training dynamics.
- Class weights inverse frequency n/(14*n_c), training only. Statistics mean/std(training only,std floor1e-5), original probe formula. No val/test fit.
- Select every arm only by validation macro-F1, tie earliest; export last-budget metrics too. Evaluate test only after selection, descriptive already-observed subject06. Keep all14-class full/steady metrics, bed F1, short recall and support. Short labels remain speech-command runs, not physical durations.
- Contrasts H1-H0/H2-H0/H3-H0 are single-factor effects under their panel. H4 combines factors and cannot identify individual interactions. One seed, few subjects; no significance claims. Standardization also changes effective L2 regularization and AdamW geometry, not solely post-hoc calibration.
- No Git checkout exists; preserve immutable versioned capsules instead of creating fictitious commits. No environment/package installation.

## Tasks

- [x] 1. Add red tests for factor settings, equivalent standardized warm-start function, train-only statistics, deterministic prefixes and original-head reproduction rejection; implement cache-only functions and verify in interview.
- [x] 2. Implement audited cache loading, reference/replay gates, both training panels, per-arm histories/heads/predictions and immutable provenance. Package source; server native regressions before launch.
- [x] 3. Run single detached worker, monitor meaningful progress, verify original cache/model hashes unchanged, collect machine artifacts and SHA manifest.
- [x] 4. Generate comparison report/curves/14-class metrics, independently review implementation/results and run final verification. Update documentation links.

## Ledger

2026-10-11: Existing feature caches train160328/val53454/test40088; original Stage1 budget3epochs lr.001 clip1 versus probe20epochs lr.01 no clip. Ruling: exact H0@0 gate plus two internally controlled panels is necessary to honor both reproduction endpoints without silently changing optimizer/init in H4. User authorizes continued diagnostic; proceed without another permission gate.

Task1: observed5 missing-module failures, implemented module;5passed. Related interview regressions14passed,1warning (normal execution; sandbox Torch stalls). Initial server capsule v1 native57passed/1failed because packaging omitted scripts.make_synthetic_parrad_data used by an existing test. Read failure traceback, include existing helper in fresh v2 capsule; preserve v1/failure log, no package installation or dataset changes.

Task2/3: v2 source123files SHA f5668f7031bb5e18a728723061df3d8e29d614aee4294c2cc023020a1eb8cc08; native58passed1warning. Single worker11:23:53→11:24:10+08:00 completed10heads. H0@0 all predictions and6confusions reproduced (maxlogit difference6.20e-6); probeH4 epoch5 reproduced previous val/test predictions and20epochF1 history exactly, maxhead difference1.19e-7. Originalcheckpoint and38cache hashes unchanged. Output /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_head_ablation_20261011_v1. Download49files hashes verified, archive SHA f00d04b64b137583933909ef24020a52df350011bc01e3017bb41cb87defe0d1.

Task4: Generated docs/radar-text-head-ablation-2026-10-11.md, PNG/SVG validation curves,616-row full/steady per-class CSV. Independent fresh read-only review found no blocking issue and verified all10arm numbers, gates, curves and conclusions. Incorporated explicit original547step overlapping-window vs40step cache trajectory difference, random-probe standardized initial-function difference, primaryH4-H1 test only+.14pp, and best-epoch vslast-epoch distinction. Changed significant to obvious improvement; no statistical significance claim. Report does not claim language generation improved.
