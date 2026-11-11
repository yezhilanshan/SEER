"""Instance-preserving sequence and partial-annotation natural-stream metrics."""
import numpy as np


def sequence_metrics(predicted, target):
    previous = list(range(len(target)+1))
    for i, p in enumerate(predicted, 1):
        current = [i]
        for j, t in enumerate(target, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(p!=t)))
        previous = current
    repeated={i for i,t in enumerate(target) if (i>0 and t==target[i-1]) or (i+1<len(target) and t==target[i+1])}
    matched={j for _,j in match_ordered(predicted,target,lambda i,j:0.,lambda i,j:predicted[i]==target[j])}
    recovered=len(repeated & matched)
    return dict(sequence_exact=int(list(predicted)==list(target)), count_mae=abs(len(predicted)-len(target)),
                edit_distance=previous[-1], predicted_count=len(predicted), target_count=len(target),
                oversegmentation_ratio=len(predicted)/len(target) if target else None,
                adjacent_same_class_instance_support=len(repeated),adjacent_same_class_instances_matched=recovered,
                adjacent_same_class_instance_recall=recovered/len(repeated) if repeated else None)


def match_ordered(predicted, truth, similarity, eligible):
    """Max cardinality, then summed similarity, then earliest indices; intervals are ordered."""
    n, m = len(predicted), len(truth)
    counts = np.zeros((n+1, m+1), np.int32)
    quality = np.zeros((n+1, m+1), np.float64)
    action = np.zeros((n, m), np.int8)
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            choices = [(int(counts[i+1,j]), quality[i+1,j], 0, 1),
                       (int(counts[i,j+1]), quality[i,j+1], 1, 2)]
            if eligible(i, j):
                choices.append((1+int(counts[i+1,j+1]), similarity(i,j)+quality[i+1,j+1], 2, 3))
            c, q, _, move = max(choices)
            counts[i,j], quality[i,j], action[i,j] = c, q, move
    pairs, i, j = [], 0, 0
    while i<n and j<m:
        move = action[i,j]
        if move==3:
            pairs.append((i,j)); i+=1; j+=1
        elif move==1:
            i+=1
        else:
            j+=1
    return pairs


def _rates(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def _boundary(predicted, truth, confirmed, known, tolerance, kind, period):
    # Initial/EOF forced boundaries are absent from predicted; recording-edge truth is excluded explicitly.
    edge_count = sum(p<=0 or p>=len(known) for p in truth)
    truth = [p for p in truth if 0<p<len(known)]
    pairs = match_ordered(predicted, truth, lambda i,j:-abs(predicted[i]-truth[j]),
                          lambda i,j:abs(predicted[i]-truth[j])<=tolerance)
    used = {i for i,_ in pairs}
    fp = sum(bool(known[p if kind=='start' else p-1]) for i,p in enumerate(predicted) if i not in used)
    result = _rates(len(pairs), fp, len(truth)-len(pairs))
    result.update(predicted=len(predicted), excluded_truth_at_recording_edges=edge_count,
                  unscored_unknown=len(predicted)-len(pairs)-fp,
                  matches=[dict(predicted=predicted[i], truth=truth[j],
                                localization_error_s=(predicted[i]-truth[j])*period,
                                confirmation_delay_s=(confirmed[i]-truth[j])*period) for i,j in pairs])
    return result


def score_recording(events, instances, frames, *, period=1/15, tolerance=4):
    if frames < 1 or period <= 0 or tolerance < 0:
        raise ValueError('invalid recording metric parameters')
    truth = sorted(instances, key=lambda r:r['start'])
    known = np.zeros(frames, dtype=bool)
    end = 0
    for t in truth:
        if not end <= t['start'] < t['stop'] <= frames:
            raise ValueError('overlapping or invalid truth interval')
        known[t['start']:t['stop']] = True; end=t['stop']
    end = 0
    for p in events:
        if not end <= p['start_frame'] < p['stop_frame'] <= frames:
            raise ValueError('predicted events must be ordered, nonoverlapping, nonempty intervals')
        end=p['stop_frame']
    def iou(i,j):
        p,t=events[i],truth[j]
        overlap=max(0,min(p['stop_frame'],t['stop'])-max(p['start_frame'],t['start']))
        return overlap/(max(p['stop_frame'],t['stop'])-min(p['start_frame'],t['start']))
    pairs=match_ordered(events,truth,iou,lambda i,j:events[i]['class_id']==truth[j]['label'] and iou(i,j)>=.5)
    used={i for i,_ in pairs}; fp=unknown=mixed=0
    burdens={'known':0,'unknown':0,'mixed':0}
    for i,p in enumerate(events):
        if i in used:
            continue
        mask=known[p['start_frame']:p['stop_frame']]
        if mask.all(): fp+=1; kind='known'
        elif not mask.any(): unknown+=1; kind='unknown'
        else: mixed+=1; kind='mixed'
        burdens[kind]+=len(mask)
    metric=_rates(len(pairs),fp,len(truth)-len(pairs))
    unknown_minutes=int((~known).sum())*period/60
    metric.update(unscored_unknown=unknown,unscored_mixed=mixed,
                  unmatched_duration_s={k:v*period for k,v in burdens.items()},
                  unknown_predictions_per_nominal_minute=unknown/unknown_minutes if unknown_minutes else None,
                  predicted_frame_coverage=sum(p['stop_frame']-p['start_frame'] for p in events)/frames,
                  matches=[dict(prediction=i,truth=j,iou=iou(i,j)) for i,j in pairs])
    stop_events=[p for p in events if p['closure']=='fire' and p['stop_frame']<frames]
    start_events=[(p,events[i-1]['confirmed_at_frame']) for i,p in enumerate(events) if i and p['start_frame']>0]
    return dict(events=metric,time_basis='nominal',frames=frames,unknown_frames=int((~known).sum()),
                start_boundary=_boundary([p['start_frame'] for p,_ in start_events], [t['start'] for t in truth],
                                         [c for _,c in start_events],known,tolerance,'start',period),
                end_boundary=_boundary([p['stop_frame'] for p in stop_events], [t['stop'] for t in truth],
                                       [p['confirmed_at_frame'] for p in stop_events],known,tolerance,'end',period))
