# Stage3A:500/1000-step zero-update diagnosis

User authorization: only diagnosis; no AdamW, no Semantic Difference/Candidate training/calibration, no runtime trigger/head changes. Diagnose score ranking separately from one-proposal-per-high-excursion emission. Negative proposal latency is permitted and is not itself a failure.

Frozen sources: SHA-locked original run proposal1000, checkpoint500/1000 and their saved arrays/JSON. No model forward or parameter update is necessary. Validate archive/member or native saved-array SHA before analysis. Compare identical8dev recordings/704boundaries, grouped by recording, subject, sensor, subject×sensor. No holdout or new data, no checkpoint/threshold selection for deployment.

## Definitions locked before reading score arrays

- Gaussian sigma .36s. Main score metrics directly use saved soft target y: fractional-positive mass y and negative mass1-y, tied scores grouped, soft ROC AUC and step-integrated soft PR AP; baseline mean(y). These are explicitly fractional-target metrics, not conventional hard-label AUROC/AP.
- Conventional hard-label score ROC/AP/PR: primary boundary distance≤.36 versus stable distance>.72, excluding grey region(.36,.72]; also sensitivity analyses all updates with distance≤.36 or≤.72 positive. Report positive prevalence, skipped updates and counts; no hidden binarization of soft labels. AP is noninterpolated PR area, separately label trapezoidal PR area.
- Exact PR curves over all distinct score cutoffs, ties grouped. Preserve full curve data in compressed artifacts; render comparable500/1000 curves per Local/Radar/fusion.
- Threshold sweep .10..95 increment.05 plus historical .6, for each branch. Same fixed persistence2 and ±.72s one-to-one chronological matching; report Recall, false/min, density/min, latency and post-boundary recall. A higher-threshold optimum is diagnostic only, not a selected deployed threshold.
- High segment duration=number of eligible contiguous updates×.18; valid gaps break segments. Report median/P90/P95/max, counts and high-score exposure; count GT boundaries whose timestamps lie inside the segment, and separately ±.72 vicinity counts. One high segment spanning several GT changes directly measures re-arm suppression potential, not perfect head learning.
- Boundary coverage without trigger: any high score within±.72; separately any two consecutive high updates within that window, regardless of whether trigger already disarmed. Classify original-trigger misses into no high score, high but no qualifying consecutive pair, qualifying pair without independent emitted matched proposal. Last class includes dead-zone and one-to-one matching competition; separately count qualifying times in disarmed segments whose emission is outside boundary window. This avoids attributing all trigger gaps to dead-zone blindly.
- Distribution: quantiles min/P1/P5/P25/P50/P75/P90/P95/P99/max; mean/std, saturation≤.05 or≥.95; boundary/stable/grey and overall. Fusion inflation: fraction fused≥.6 while both individual scores<.6, mean uplift above max(heads), head correlation. Apply same calculations500/1000 and every group.

## Work

1. Synthetic tests for perfect/reversed/tied hard ranking and fractional metrics; segment spanning multiple boundaries; coverage vs one-shot emission, gaps and OR inflation.
2. Pure NumPy read-only diagnostic implementation; independent original metric check; JSON/CSV/full PR NPZ and scientific plots. Native fallback if large archive transfer unreliable: SHA-check native arrays and checkpoints, detached read-only diagnostic, download small compressed diagnostic+raw scores rather than giant frontend/weights.
3. Report each criterion with500→1000 differences. No arbitrary numerical 'good AUPRC' gate after viewing data, no causal certainty from descriptive development metrics. Head and trigger can both fail; a trigger failure is not proof head is good. One seed, upstream-seen TRAIN dev subjects; no formalGo.
