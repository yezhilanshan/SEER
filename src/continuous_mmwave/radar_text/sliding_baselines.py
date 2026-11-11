"""Matched-window scoring and validation-only selection for classification templates."""
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .data import caption_from_actions
from .diagnostics import smooth_predictions
from .metrics import score_caption


METRICS = ('sequence_exact','action_precision','action_recall','order_lcs','template_exact','unmatched')
GRID = [(median, minimum) for median in (0., .3, .5) for minimum in (0., .3, .6)]


def complete_window(duration, frame_period):
    return duration + frame_period >= 12.-1e-6


def pair_previous(windows, rows):
    ids = [r['sample_id'] for r in rows]
    expected = [r['sample_id'] for r in windows]
    if len(set(ids)) != len(ids) or len(set(expected)) != len(expected) or set(ids) != set(expected):
        raise ValueError('missing, extra or duplicate sample IDs')
    by_id = {r['sample_id']: r for r in rows}
    ordered = [by_id[r['sample_id']] for r in windows]
    if any(a['actions'] != b['reference_actions'] for a,b in zip(windows, ordered)):
        raise ValueError('reference actions changed for matched window')
    return ordered


def apply_saved_head(hidden, saved):
    weight, bias = (saved['head'][k].to(hidden.device) for k in ('weight','bias'))
    mean, std = (saved[k].to(hidden.device) for k in ('mean','std'))
    if not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError('invalid saved training standardization')
    return F.linear((hidden-mean)/std, weight, bias)


def score_windows(windows, predictions, timestamps, phrases, threshold):
    if not len(windows) == len(predictions) == len(timestamps):
        raise ValueError('window/prediction count mismatch')
    output = []
    for row,pred,times in zip(windows,predictions,timestamps):
        if len(pred) != len(times):
            raise ValueError('frame length mismatch')
        actions = smooth_predictions(pred, times, **threshold)
        text = caption_from_actions(actions, phrases) if actions else 'No activity was predicted.'
        output.append(dict(sample_id=row['sample_id'], subject=row['subject'], sensor=row['sensor'],
                           full_window=row['full_window'], reference_actions=row['actions'],
                           generated=text, **score_caption(text, row['actions'], phrases)))
    return output


def summarize(rows):
    if not rows:
        return dict(windows=0, metrics={key:None for key in METRICS}, per_class=[])
    deficit = [max(0,len(r['reference_actions'])-len(r['predicted_actions'])) for r in rows]
    excess = [max(0,len(r['predicted_actions'])-len(r['reference_actions'])) for r in rows]
    support, predicted, common = Counter(), Counter(), Counter()
    for row in rows:
        a,b = Counter(row['reference_actions']),Counter(row['predicted_actions'])
        support.update(a);predicted.update(b);common.update(a&b)
    return dict(windows=len(rows), exact_count=int(sum(r['sequence_exact'] for r in rows)),
                metrics={key:float(np.mean([row[key] for row in rows])) for key in METRICS},
                mean_predicted_actions=float(np.mean([len(r['predicted_actions']) for r in rows])),
                mean_reference_actions=float(np.mean([len(r['reference_actions']) for r in rows])),
                count_deficit_windows=int(np.count_nonzero(deficit)), count_excess_windows=int(np.count_nonzero(excess)),
                count_deficit_occurrences=sum(deficit), count_excess_occurrences=sum(excess),
                per_class=[dict(class_id=i, support=support[i], predicted=predicted[i], covered=common[i],
                                recall=common[i]/support[i] if support[i] else None,
                                precision=common[i]/predicted[i] if predicted[i] else None) for i in range(14)])


def choose_thresholds(grid, *, split):
    if split != 'validation':
        raise ValueError('thresholds may only be selected on validation')
    if set(grid) != {'B0','B1','B2'} or not all(grid.values()):
        raise ValueError('all three classification baselines require validation candidates')
    def pair(row):return (row['median_seconds'],row['minimum_seconds'])
    index = {method:{pair(row):row for row in rows} for method,rows in grid.items()}
    keys = set(index['B0'])
    if any(set(rows)!=keys for rows in index.values()):
        raise ValueError('classification baselines must use identical candidate thresholds')
    def rank(row):
        return tuple(row['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall')) + (-row['median_seconds'],-row['minimum_seconds'])
    common_grid = []
    for median,minimum in sorted(keys):
        metric = {key:float(np.mean([index[m][(median,minimum)]['metrics'][key] for m in index]))
                  for key in ('sequence_exact','order_lcs','action_recall')}
        common_grid.append(dict(median_seconds=median, minimum_seconds=minimum, metrics=metric))
    common = max(common_grid,key=rank)
    individual = {method:max(rows,key=rank) for method,rows in grid.items()}
    return dict(selection_split=split, selection_subset='complete nominal12s windows',
                objective='mean B0/B1/B2 sequence_exact, LCS, recall; ties simpler',
                common={k:common[k] for k in ('median_seconds','minimum_seconds')},
                individual={m:{k:r[k] for k in ('median_seconds','minimum_seconds')} for m,r in individual.items()},
                common_candidates=common_grid, per_method_candidates=grid)


def paired_exact(first, second):
    pair_previous([dict(sample_id=r['sample_id'],actions=r['reference_actions']) for r in first],second)
    by_id = {r['sample_id']:r for r in second}
    pairs = [(bool(r['sequence_exact']),bool(by_id[r['sample_id']]['sequence_exact'])) for r in first]
    return dict(windows=len(pairs),first_only=sum(a and not b for a,b in pairs),
                second_only=sum(b and not a for a,b in pairs),both_correct=sum(a and b for a,b in pairs),
                both_wrong=sum(not a and not b for a,b in pairs),
                note='Paired descriptive counts only; overlapping windows and synchronized views are correlated.')
