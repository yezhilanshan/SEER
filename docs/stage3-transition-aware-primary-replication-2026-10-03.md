# Primary recipe replication preregistration, 2026-10-03

User authorized the next stability check. Primary pilot already completed and its
artifacts are locked. Do not change the accepted recipe, evaluate held data, or
expand the Projection budget based on results.

## Fixed new run

Run one new fold held_subject=subject_03, seed=3407. TRAIN subjects02/04 set000/001
are fit; subjects02/04 set002 are dev. Subject03 and set003 numerical inputs remain
excluded from this run; validation/test remain excluded. These are fold-specific
roles: some subjects were fit/dev in the previous fold, so folds are correlated,
not independent held test results. Stage2 upstream saw TRAIN subjects.

Start the original Stage2 Projection, not the previous adapted checkpoint. Preserve
primary age6/7 natural probe-eligible row weights (each age/class mass1/4), frozen
all-fit teacher normalization, FP64 shared heads ridge.001 LBFGS400 with existing
tolerances/rules and every-step refitting. Frozen control and relation1/10 each
joint condition gets exactly the existing at-most20 fullfit steps lr1e-5 decay0
clip1, milestones0/5/10/20. Teacher relation+consistency.1+anchor1 and original dev
probe/semantic gates unchanged. Rehearsal window cap4 per recording/class,
deterministic evenly-spaced clean windows and all42 templates unchanged.

If existing role-valid caches are unavailable, build fit/dev caches once from
frozen Stage2 using the existing prepare_rehearsal implementation, preserving all
manifest dependency and upstream hashes and role checks. No retraining upstream,
no cache relabeling, no original artifacts overwritten. Runtime/environment
failures may be repaired/resumed without changing the experiment settings.

## Seed interpretation and success criteria

The present recipe has zero-initialized fitted heads, pretrained copied P, full
batch updates, deterministic fixed feature seeds and no dropout/sampling. Training
seed changes RNG bookkeeping but appears not to change the optimization trajectory.
Verify actual model/optimizer behavior on a fixed synthetic input with1701 vs3407;
report equality explicitly. Do not claim multiple independent stochastic trials
from changing this unused seed or add randomization merely to create differences.

Primary replication criterion: for each coefficient separately, a new nonzero-step
checkpoint must improve the *same new fold's* originalP baseline score and pass
all original semantic gates. Evaluate coefficients1/10 both; report all milestones
and failures. Compare fixed step20 as well as the prescribed selected checkpoint;
do not pick a new coefficient after reading the new fold. A baseline or evaluation
solver failure is a blocked diagnostic, not evidence of gain/loss. Keep solver
budgets and thresholds unchanged. Numerical head-refit failures stop only that
condition and lock baseline as implemented. No automatic extra folds/seeds/steps.

The new fold is a repeat of the method on changed fit/dev subjects, not a formal
independent test or proof of individual correction causality. Report paired fold
gains without treating overlapping folds as independent significance evidence.
No production replacement or full-pilot launch.
