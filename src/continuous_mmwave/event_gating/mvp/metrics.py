"""Seconds-based, one-fire-one-boundary scoring on full partial-known streams."""
import numpy as np
from ..metrics import match_ordered, _rates, sequence_metrics


def evaluate_rollout(result, record, period=1/15):
    frames = record['frames']
    if frames < 1 or not np.isfinite(period) or period <= 0:
        raise ValueError('invalid recording metric parameters')
    truth = sorted(record['instances'], key=lambda r:r['start'])
    known = np.zeros(frames, bool)
    end = 0
    for row in truth:
        if not end <= row['start'] < row['stop'] <= frames:
            raise ValueError('overlapping or invalid truth interval')
        known[row['start']:row['stop']] = True
        end = row['stop']
    trace = result['trace']
    if (np.shape(trace['fire']) != (frames,) or not np.array_equal(trace['frame'], np.arange(frames))):
        raise ValueError('full native frame trace required')
    fires = np.flatnonzero(trace['fire']).tolist()
    if 0 in fires: raise ValueError('recording start is initialization, not a fire')
    edges = sorted({b for t in truth for b in (t['start'], t['stop'])})
    boundaries = [b for b in edges if 0 < b < frames]
    # A fire splits [t-1,t); either known adjacent frame defines the valid region.
    valid = known.copy()
    valid[1:] |= known[:-1]
    valid[0] = False
    valid_seconds = int(valid.sum())*period
    unknown_seconds = int((~valid).sum())*period
    boundary = {}
    for tolerance in (.2,.5,1.):
        pairs = match_ordered(fires, boundaries,
                    lambda i,j:-abs(fires[i]-boundaries[j]),
                    lambda i,j:abs(fires[i]-boundaries[j])*period<=tolerance+1e-12)
        used = {i for i,_ in pairs}
        fp = sum(bool(valid[p]) for i,p in enumerate(fires) if i not in used)
        unknown = len(fires)-len(pairs)-fp
        delays = [(fires[i]-boundaries[j])*period for i,j in pairs]
        score = _rates(len(pairs),fp,len(boundaries)-len(pairs))
        score.update(predicted=len(fires), unscored_unknown=unknown,
                     excluded_truth_at_recording_edges=len(edges)-len(boundaries),
                     false_fires_per_minute=fp/(valid_seconds/60) if valid_seconds else None,
                     unknown_fires_per_minute=unknown/(unknown_seconds/60) if unknown_seconds else None,
                     valid_duration_s=valid_seconds, unknown_duration_s=unknown_seconds,
                     signed_delay_mean_s=float(np.mean(delays)) if delays else None,
                     absolute_error_mean_s=float(np.mean(np.abs(delays))) if delays else None,
                     missed_ratio=(len(boundaries)-len(pairs))/len(boundaries) if boundaries else None,
                     matches=[dict(predicted=fires[i],truth=boundaries[j],delay_s=d) for (i,j),d in zip(pairs,delays)])
        boundary[str(tolerance)] = score
    events = result['events']
    end=0
    for p in events:
        if not end <= p['start_frame'] < p['stop_frame'] <= frames:
            raise ValueError('predicted events must be ordered, nonoverlapping full intervals')
        end=p['stop_frame']
    def iou(i,j):
        p,t=events[i],truth[j]
        return max(0,min(p['stop_frame'],t['stop'])-max(p['start_frame'],t['start'])) / (
            max(p['stop_frame'],t['stop'])-min(p['start_frame'],t['start']))
    scores={}
    for threshold in (.3,.5):
        pairs=match_ordered(events,truth,iou,lambda i,j:events[i]['class_id']==truth[j]['label'] and iou(i,j)>=threshold)
        used={i for i,_ in pairs}
        counts=dict(known=0,unknown=0,mixed=0)
        false_classes=[]
        durations=dict(known=0.,unknown=0.,mixed=0.)
        for i,p in enumerate(events):
            if i in used: continue
            mask=known[p['start_frame']:p['stop_frame']]
            kind='known' if mask.all() else 'mixed' if mask.any() else 'unknown'
            counts[kind]+=1; durations[kind]+=len(mask)*period
            if kind=='known': false_classes.append(p['class_id'])
        score=_rates(len(pairs),counts['known'],len(truth)-len(pairs))
        score.update(unscored_unknown=counts['unknown'],unscored_mixed=counts['mixed'],
                     unmatched_duration_s=durations,
                     matches=[dict(prediction=i,truth=j,iou=iou(i,j)) for i,j in pairs])
        matched_classes=[truth[j]['label'] for _,j in pairs]
        classes=sorted({t['label'] for t in truth}|set(false_classes))
        score['per_class']={}
        for label in classes:
            tp=matched_classes.count(label)
            support=sum(t['label']==label for t in truth)
            score['per_class'][str(label)]=dict(_rates(tp,false_classes.count(label),support-tp),support=support)
        scores[str(threshold)]=score
    interior=[sum(t['start']<p<t['stop'] for p in fires) for t in truth]
    matched_fires={m['predicted'] for m in boundary['0.5']['matches']}
    extra=[sum(t['start']<p<t['stop'] and p not in matched_fires for p in fires) for t in truth]
    # Sequence comparison only for predictions wholly known; mixed intervals are
    # excluded intact, rather than clipped into artificially high-IoU events.
    strict=[p['class_id'] for p in events if known[p['start_frame']:p['stop_frame']].all()]
    return dict(boundary=boundary,events=scores,sequence=sequence_metrics(strict,[t['label'] for t in truth]),
                time_basis='nominal',frame_period_s=period,frames=frames,
                oversegmentation=dict(fires_inside_instances=sum(interior),
                                      extra_fires_inside_instances=sum(extra),
                                      primary_tolerance_s=.5,
                                      affected_instances=sum(n>0 for n in extra),
                                      affected_instance_ratio=sum(n>0 for n in extra)/len(truth) if truth else None,
                                      extra_fires_per_instance=sum(extra)/len(truth) if truth else None,
                                      per_instance=extra, raw_per_instance=interior),
                diagnostics=dict(fire_count=len(fires),known_duration_s=int(known.sum())*period,
                                 unknown_duration_s=int((~known).sum())*period,
                                 boundary_valid_region='either adjacent frame annotated; start excluded',
                                 interval_policy='full interval IoU; unmatched mixed/unknown unscored'))
