"""Independent artifact checks: no model construction and no shared metric implementation."""
import hashlib
import json
import math
from pathlib import Path


def _require(condition, message):
    if not condition: raise ValueError(message)


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_run(path):
    path=Path(path); status=json.loads((path/'status.json').read_text(encoding='utf-8'))
    _require(status['status']=='complete','run is not complete')
    files={p.name:_hash(p) for p in sorted(path.iterdir()) if p.is_file()}
    source=json.loads((path/'source.json').read_text(encoding='utf-8'))
    checked=0
    if (path/'probe.json').exists():
        result=json.loads((path/'probe.json').read_text(encoding='utf-8'))
        _require(all(result.get(k) is True for k in ('input_gradient_nonzero','frozen_unchanged','group_isolation_passed')),'probe invariant failed')
        _require(result['learned_reload_max_abs']==0,'probe reload mismatch')
        checked+=1
    for checkpoint in sorted(path.glob('step-*.pt')):
        import torch
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # JSON normalizes tuples to lists; compare exported provenance after the same normalization.
        _require(json.loads(json.dumps(payload['source']))==source,'checkpoint provenance mismatch')
        _require(payload['step']>0 and bool(payload['optimizer']['state']),'empty optimizer evidence')
        _require(set(payload['learned'])==set(payload['optimizer_names']),'optimizer learned parameter mismatch')
        _require(all(torch.isfinite(p).all() for p in payload['learned'].values()),'nonfinite learned checkpoint')
        _require(all(int(s['step'])==payload['step'] and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
                     for s in payload['optimizer']['state'].values()),'optimizer step/moment mismatch')
        checked+=1
    prediction_path=path/'predictions.jsonl'
    if prediction_path.exists():
        rows=[json.loads(x) for x in prediction_path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            for trace in row['trace']:
                candidates=trace['candidates']; selected=trace['selected']
                _require(len(trace['scores'])==len(candidates) and all(math.isfinite(s) for s in trace['scores']),'invalid retrieval scores')
                _require(all(0<=c['start_frame']<c['end_frame']<=row['cutoff_frame'] and c['level']<=trace['group'] for c in candidates),
                         'future or out-of-group candidate')
                _require(all(s in candidates for s in selected),'selected trace not a legal candidate')
                _require([s['start_frame'] for s in selected]==sorted(s['start_frame'] for s in selected),'trace time order mismatch')
            _require(len(row['trace'])==3*row['rounds'],'missing retrieval round')
            reference=row['reference']['sequence']; predicted=row['predicted_sequence']; metrics=row['metrics']
            if predicted is None:
                _require(not metrics['parseable'] and not metrics['exact'],'unparseable prediction credited')
            else:
                # Independent full edit-distance matrix, distinct from the rolling-row evaluator.
                table=[[0]*(len(predicted)+1) for _ in range(len(reference)+1)]
                for i in range(len(reference)+1): table[i][0]=i
                for j in range(len(predicted)+1): table[0][j]=j
                for i in range(1,len(reference)+1):
                    for j in range(1,len(predicted)+1):
                        table[i][j]=min(table[i-1][j]+1,table[i][j-1]+1,table[i-1][j-1]+int(reference[i-1]!=predicted[j-1]))
                _require(metrics['edit_distance']==table[-1][-1] and metrics['exact']==(reference==predicted),'sequence metric mismatch')
            checked+=1
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        _require(summary['queries']==len(rows),'summary row count mismatch')
    _require(checked>0,'no supported artifacts verified')
    return dict(passed=True,artifacts_checked=checked,files_sha256=files)


def verify_original_files(manifest):
    rows=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    for row in rows:
        _require(_hash(row['path']).lower()==row['sha256'].lower(),f"original file changed: {row['path']}")
    return len(rows)
