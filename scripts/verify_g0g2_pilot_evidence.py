"""Independently recompute saved pilot evidence without experiment metric helpers."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import torch
from continuous_mmwave.latent_memory.data import PHRASES


def check_row(row):
    text=' '.join(row['text'].lower().split())
    if text=='no annotated action has been observed yet.': pred,active=[],[]
    else:
        assert text.startswith('the person ') and text.endswith('.'), row['uid']
        pred=[]; active=[]
        for chunk in re.split(r',\s*then\s+',text[11:-1]):
            state=chunk.endswith(' (ongoing)')
            phrase=chunk[:-10] if state else chunk
            pred.append(PHRASES.index(phrase)); active.append(state)
    truth=row['reference']['sequence']; flags=row['reference']['ongoing']
    # Full matrix deliberately differs from the production rolling-row DP.
    dp=[[0]*(len(pred)+1) for _ in range(len(truth)+1)]
    for i in range(len(truth)+1): dp[i][0]=i
    for j in range(len(pred)+1): dp[0][j]=j
    for i,a in enumerate(truth,1):
        for j,b in enumerate(pred,1): dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a!=b))
    truncated=row['truncated']; assert not truncated, 'separate partial-output verification needed'
    expected=dict(edit_score=1-dp[-1][-1]/max(len(truth),len(pred),1),sequence_em=float(truth==pred),
        state_em=float(truth==pred and flags==active),total_count_mae=abs(len(truth)-len(pred)),
        class_count_l1=sum(abs(Counter(truth)[k]-Counter(pred)[k]) for k in range(12)),parseable=True,truncated=False)
    for k,v in expected.items(): assert abs(v-row['metrics'][k])<1e-12,(row['uid'],k)
    return expected


def summarize(rows):
    by_record=defaultdict(list)
    values=[]
    for row in rows:
        value=check_row(row); by_record[row['file']].append(value); values.append(value)
    fields=values[0]
    macro={k:sum(sum(v[k] for v in group)/len(group) for group in by_record.values())/len(by_record) for k in fields}
    return dict(windows=len(rows),recordings=len(by_record),record_macro=macro,
        window_mean={k:sum(v[k] for v in values)/len(values) for k in fields},
        sequence_exact=sum(v['sequence_em'] for v in values),state_exact=sum(v['state_em'] for v in values))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('directory'); args=parser.parse_args()
    root=Path(args.directory); parent=root.parent
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    rows=lambda name:[json.loads(line) for line in (root/name).read_text(encoding='utf-8').splitlines()]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads((parent/'evidence-files.json').read_text())
    for name,meta in manifest.items():
        p=root/name; assert p.stat().st_size==meta['bytes'] and digest(p)==meta['sha256'], name
    result=read('result.json'); summaries={}
    for key,name in [('before','before-correct'),('correct','after-correct'),('zero','after-zero'),('swapped','after-swapped')]:
        group=rows(name+'.jsonl'); summaries[key]=summarize(group)
        for k,v in summaries[key]['record_macro'].items(): assert abs(result[key]['record_macro'][k]-v)<1e-12,(key,k)
    oracle=rows('text-oracle.jsonl')
    summaries['oracle']={kind:summarize([r for r in oracle if r['kind']==kind]) for kind in sorted({r['kind'] for r in oracle})}
    for kind,group in summaries['oracle'].items():
        for k,v in group['record_macro'].items(): assert abs(result['oracle'][kind]['record_macro'][k]-v)<1e-12,(kind,k)
    train=rows('alignment-pilot/training.jsonl'); assert [r['step'] for r in train]==list(range(1,201))
    initial=torch.load(root/'alignment-pilot/step-000000.pt',map_location='cpu',weights_only=False)
    final=torch.load(root/'alignment-pilot/step-000200.pt',map_location='cpu',weights_only=False)
    assert initial['source']==final['source'] and initial['frozen_digest']==final['frozen_digest']
    assert final['step']==200 and final['phase']=='alignment'
    assert {int(s['step']) for s in final['optimizer']['state'].values()}=={200}
    changed=[n for n,p in initial['new_state'].items() if not torch.equal(p,final['new_state'][n])]
    assert changed and all(n.startswith(('projector.','time_projection.')) for n in changed)
    pilot=read('pilot-panel.json'); assert len(pilot)==32 and all(e['cohort']=='fit' for e in pilot)
    assert {r['uid'] for r in train}=={e['uid'] for e in pilot}
    assert {y for e in pilot for y in e['reference']['sequence']}==set(range(12))
    controls={r['uid']:r for r in rows('after-swapped.jsonl')}; donors=read('donors.json'); by_uid={e['uid']:e for e in pilot}
    for uid,other in donors.items():
        assert controls[uid]['actual_evidence_uid']==other
        assert by_uid[uid]['seconds']==by_uid[other]['seconds']
        assert by_uid[uid]['reference']['sequence']!=by_uid[other]['reference']['sequence']
    zero={r['uid']:r for r in rows('after-zero.jsonl')}
    for row in rows('after-correct.jsonl'): assert row['prefix_tokens']==zero[row['uid']]['prefix_tokens']==controls[row['uid']]['prefix_tokens']
    protected=json.loads((parent/'protected-files.json').read_text(encoding='utf-8-sig'))
    assert all(digest(r['path'])==r['sha256'].lower() for r in protected)
    assert not result['passed'] and not result['g2_started'] and not result['formal_started']
    source=json.loads((parent/'source-readback.json').read_text()); assert source['mismatches']==[]
    report=dict(verified=True,evidence_files=len(manifest),protected_unchanged=len(protected),source_readback=source,
        changed_new_parameters=changed,successful_updates=200,unique_training_windows=32,
        first10_loss=sum(r['loss'] for r in train[:10])/10,last10_loss=sum(r['loss'] for r in train[-10:])/10,
        summaries=summaries,scope='training-panel diagnostic; no G2 or unseen-subject efficacy inference')
    (parent/'independent-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='summaries'},indent=2))


if __name__=='__main__': main()
