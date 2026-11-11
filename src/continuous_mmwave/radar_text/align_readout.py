"""P0 controlled frozen-Align readout recipes and paired count diagnostics."""
import numpy as np

from .head_ablation import arm_specs
from .sliding_baselines import summarize
from .runner import check_holdout_subjects


def p0_specs():
    return [s for s in arm_specs('original_optimizer') if s['arm'] in ('H1','H4')]


def validate_encoder_identity(actual_sha256, previous_inputs):
    hashes={v for k,v in previous_inputs.items() if k.replace('\\','/').endswith('/align/best.pt')}
    if len(hashes)!=1 or hashes!={actual_sha256}:
        raise ValueError('actual Align encoder differs from previous B3 encoder')
    return actual_sha256


def validate_subjects(checkpoint, records):
    groups={split:{str(r['subject_id']) for r in rows} for split,rows in records.items()}
    if groups['train']!=set(checkpoint['training_subjects']):
        raise ValueError('feature training subjects differ from original training subjects')
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('subject splits overlap')
    check_holdout_subjects(checkpoint,records['validation'])
    check_holdout_subjects(checkpoint,records['test'],test=True)


def validate_window_inputs(actual, expected):
    keys=('sample_id','actions','input_sha256','start','stop','full_window')
    if len(actual)!=len(expected) or len({r['sample_id'] for r in actual})!=len(actual):
        raise ValueError('missing or duplicate windows')
    if any(any(a[k]!=b[k] for k in keys) for a,b in zip(actual,expected)):
        raise ValueError('paired radar window, reference or input changed')
    return len(actual)


def count_metrics(rows):
    delta=np.array([len(r['predicted_actions'])-len(r['reference_actions']) for r in rows])
    if not len(delta):raise ValueError('count metrics require windows')
    return dict(count_mae=float(np.abs(delta).mean()),count_bias_pred_minus_gt=float(delta.mean()),
                mean_under_count=float(np.maximum(-delta,0).mean()),mean_over_count=float(np.maximum(delta,0).mean()),
                under_windows=int((delta<0).sum()),over_windows=int((delta>0).sum()),equal_count_windows=int((delta==0).sum()))


def readout_summary(rows):
    return dict(summarize(rows),**count_metrics(rows))
