# Primary-age Projection pilot, preregistered 2026-10-03

This independent diagnostic implements the objective-audit corrections. It does not
replace the Stage2 model or launch the full pilot. One held fold subject_02, seed1701,
fit other subjects set000/001 and dev set002; no held/calibration numeric access and
no validation/test access. Existing Stage2 upstream training included TRAIN subjects;
this is not a claim of end-to-end unseen-subject generalization.

Primary transition fit rows must be natural, probe-eligible age6/7 endpoints. Each
age/class bucket has total mass1/4, uniformly within bucket, with no subsequent
Candidate normalization. Age3/4/5 do not enter transition loss. Log bucket counts,
effective mass and Candidate mass range. Keep prior all-fit teacher-key scale and
feature normalization frozen to isolate age/weight/head-fitting changes; early fit
rows can contribute only these initialization statistics and the semantic bank.

Three shared130-input linear heads are FP64, ridge0.001 on weights (unpenalized bias),
LBFGS strong_wolfe max400, grad tolerance1e-7, change tolerance1e-9. Convergence uses
the existing rule: gradient <= tolerance, or early exit before cap and change <=
tolerance. This is numerical convergence, not a claim of an exact optimum. Heads
fit only primary fit rows with P frozen before adaptation, and refit from their
previous state after every P step. Warmup must beat balanced constant BCE ln2 and
pass convergence and frozen-P/teacher checks before unlocking P. A failed refit
stops that condition, retains step0 selection and diagnostics, without relaxing
solver settings or increasing budget.

Compare frozen control and lambda_relation1/10. Each joint condition gets at most
20 full-primary-fit Projection AdamW steps lr1e-5, decay0, clip1. Heads are frozen
while differentiating the exact weighted transition objective; no differentiation
through the head solver. Include average head ridge in the reported task objective.
Semantic relation and stream/commit consistency use the prior teacher-bank loss
on all434 fit events and42 text templates, temperature0.07, consistency0.1,
anchor1. Frozen teacher and upstream, semantic dimension128. Forks share the same
converged warmup. Evaluation only at steps0/5/10/20, unchanged independent per-age
dev probes and unchanged semantic gates. Select highest eligible original score,
ties earlier step; head diagnostics cannot select a checkpoint. No further seeds,
folds, epochs or solver budget changes based on observed scores.

Checkpoint after each P update before head fitting and evaluation. Resume pending
phases without repeating optimizer updates. Missing warmup checkpoint must refuse
resume before overwriting evidence. Preserve hashes, solver diagnostics, exact
weight statistics, source integrity and selection locks. Report small-experiment
gains as exploratory dev results only, with all gating failures visible.
