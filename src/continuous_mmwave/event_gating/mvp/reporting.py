"""Complete MVP summaries; numerical candidates are not research conclusions."""
import numpy as np
from ..metrics import _rates
from ..data import write_json


def summarize_arm(arm):
    rows = [r['metrics'] for r in arm['records']]
    summary = dict(recordings=len(rows), boundary={}, events={})
    for tolerance in ('0.2', '0.5', '1.0'):
        scores = [r['boundary'][tolerance] for r in rows]
        score = _rates(*(sum(s[k] for s in scores) for k in ('tp', 'fp', 'fn')))
        valid = sum(s['valid_duration_s'] for s in scores)
        unknown = sum(s['unknown_duration_s'] for s in scores)
        delays = [m['delay_s'] for s in scores for m in s['matches']]
        unknown_fires = sum(s['unscored_unknown'] for s in scores)
        score.update(record_mean_f1=float(np.mean([s['f1'] for s in scores])),
                     valid_duration_s=valid, unknown_duration_s=unknown,
                     unscored_unknown=unknown_fires,
                     false_fires_per_minute=score['fp']*60/valid if valid else None,
                     unknown_fires_per_minute=unknown_fires*60/unknown if unknown else None,
                     signed_delay_mean_s=float(np.mean(delays)) if delays else None,
                     absolute_error_mean_s=float(np.mean(np.abs(delays))) if delays else None,
                     missed_ratio=score['fn']/(score['tp']+score['fn']) if score['tp']+score['fn'] else None)
        summary['boundary'][tolerance] = score
    for threshold in ('0.3', '0.5'):
        scores = [r['events'][threshold] for r in rows]
        score = _rates(*(sum(s[k] for s in scores) for k in ('tp', 'fp', 'fn')))
        score.update({k:sum(s[k] for s in scores) for k in ('unscored_unknown','unscored_mixed')})
        classes=sorted({k for s in scores for k in s['per_class']})
        score['per_class']={}
        for label in classes:
            values=[s['per_class'][label] for s in scores if label in s['per_class']]
            score['per_class'][label]=dict(_rates(*(sum(v[k] for v in values) for k in ('tp','fp','fn'))),
                                            support=sum(v['support'] for v in values))
        summary['events'][threshold] = score
    instances = sum(len(r['oversegmentation']['per_instance']) for r in rows)
    summary['oversegmentation'] = {
        k:sum(r['oversegmentation'][k] for r in rows)
        for k in ('fires_inside_instances','extra_fires_inside_instances','affected_instances')}
    summary['oversegmentation'].update(instances=instances,
        affected_instance_ratio=summary['oversegmentation']['affected_instances']/instances if instances else None)
    summary['sequence'] = dict(policy='wholly known predictions only; mixed intervals excluded',
        records=[dict(file=r['file'], **r['metrics']['sequence']) for r in arm['records']])
    return summary


def decision_report(summaries, synthetic_only):
    b1, b2, b3 = [summaries[a]['boundary']['0.5'] for a in ('B1','B2','B3')]
    recall_gap = b2['recall']-b1['recall']
    rates = (b1['false_fires_per_minute'], b2['false_fires_per_minute'])
    reduction = 1-rates[1]/rates[0] if rates[0] is not None and rates[0]>0 and rates[1] is not None else None
    supported = all(b['tp']+b['fn']>0 for b in (b1,b2,b3))
    return dict(status='synthetic_only' if synthetic_only else 'requires_real_data_review',
        research_go=False,
        reason='Synthetic mechanisms cannot establish research gates.' if synthetic_only else
               'Single validation subject and one seed require cross-subject/class consistency and trace review before Go.',
        G1=dict(candidate_pass=bool(abs(recall_gap)<=.02 and reduction>=.2) if supported and reduction is not None else None,
                recall_difference=recall_gap, false_fire_relative_reduction=reduction,
                recall_comparable=abs(recall_gap)<=.02, source='training-calibrated operating points; no validation reselection'),
        G2=dict(candidate_pass=bool(b3['f1']>b2['f1'] and
                summaries['B3']['oversegmentation']['extra_fires_inside_instances']<=
                summaries['B2']['oversegmentation']['extra_fires_inside_instances']) if supported else None,
                boundary_f1_difference=b3['f1']-b2['f1'], consistency='requires per-class and independent-subject review'),
        G3=dict(candidate_pass=None, reason='Inspect autonomous traces, repeated fires, delays and misses; no preregistered numerical cutoff.'),
        G4=dict(candidate_pass=None, event_f1_difference={k:summaries['B3']['events'][k]['f1']-
                summaries['B2']['events'][k]['f1'] for k in ('0.3','0.5')},
                reason='Shared decoder retained; semantic noninferiority margin is not specified.'))


def write_reports(output, arms, synthetic_only):
    summary = {a:summarize_arm(v) for a,v in arms.items()}
    decision = decision_report(summary, synthetic_only)
    write_json(output/'summary.json', summary)
    write_json(output/'go-no-go.json', decision)
    lines = ['# SAEF-MVP full-recording report', '', f'Synthetic only: {synthetic_only}.',
             'Calibration uses training files only. Validation never selects parameters.', '',
             '| Arm | Boundary tolerance (s) | P | R | F1 | FP/min | Missed ratio | Signed delay (s) | Absolute error (s) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for arm, row in summary.items():
        for tol, s in row['boundary'].items():
            values = [s[k] for k in ('precision','recall','f1','false_fires_per_minute',
                      'missed_ratio','signed_delay_mean_s','absolute_error_mean_s')]
            lines.append(f'| {arm} | {tol} | '+ ' | '.join(map(number,values))+' |')
    lines += ['', '| Arm | Event IoU | P | R | F1 | Unknown unmatched | Mixed unmatched |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary.items():
        for iou, s in row['events'].items():
            lines.append(f'| {arm} | {iou} | {s["precision"]:.4f} | {s["recall"]:.4f} | {s["f1"]:.4f} | {s["unscored_unknown"]} | {s["unscored_mixed"]} |')
    lines += ['', 'Counts are pooled across recordings; record-mean boundary F1 is also saved in summary.json.',
              'Unknown and unmatched mixed intervals are reported separately; they are not strict false positives.',
              'See summary.json for denominators, oversegmentation and partial-label sequence/count scores.',
              'See calibration.json for training-only recall/FP operating curves; tau-sensitivity.json fixes thresholds on calibration.',
              'See per-record trace.npz, metrics.json and timeline.svg for successful and failed cases.',
              '', f'Decision: {decision["status"]}. {decision["reason"]}',
              'G1/G2 are candidate numerical checks only; G3/G4 require review. No research Go is inferred automatically.']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
