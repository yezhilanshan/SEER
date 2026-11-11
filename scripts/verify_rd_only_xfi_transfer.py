"""Independent selected-model metrics, inputs, logits and validation selectors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_radar_text_p1 import verify,close

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frame_check(logits,labels,summary):
    prediction=logits.argmax(-1);matrix=np.zeros((14,14),np.int64)
    for truth,pred in zip(labels.tolist(),prediction.tolist()):matrix[truth,pred]+=1
    assert matrix.tolist()==summary['confusion_matrix'];support=matrix.sum(1);predicted=matrix.sum(0);tp=matrix.diagonal()
    pr=tp/np.maximum(1,predicted);re=tp/np.maximum(1,support);f1=2*pr*re/np.maximum(1e-30,pr+re)
    close(tp.sum()/matrix.sum(),summary['frame_accuracy']);close(pr.mean(),summary['macro_precision']);close(re.mean(),summary['macro_recall']);close(f1.mean(),summary['macro_f1'])
    for i,row in enumerate(summary['per_class']):
        assert row['support']==support[i] and row['predicted']==predicted[i]
        close(row['precision'],pr[i]);close(row['recall'],re[i]);close(row['f1'],f1[i])

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--p1',required=True)
    args=p.parse_args();root=Path(args.results);raw=root/'experiment';old=Path(args.p1)/'experiment'
    for name,entry in load(root/'result-manifest.json').items():assert sha(root/name)==entry['sha256'] and (root/name).stat().st_size==entry['bytes']
    report=load(raw/'results.json');training=load(raw/'training.json');selection=load(raw/'selection.json')
    events=[json.loads(line) for line in (root/'launcher/worker.log').read_text().splitlines() if line.startswith('{')]
    test_start=next(i for i,e in enumerate(events) if e['state']=='extract_RD_only' and e['split']=='test')
    assert {e['arm'] for e in events[:test_start] if e['state']=='select_validation_postprocessing' and 'arm' in e}=={'R0','R1','R2'}
    assert events[-1]['state']=='complete'
    selected_frame={};sequences=0;count_support={}
    for name,entry in training.items():
        assert [h['epoch'] for h in entry['history']]==list(range(1,21))
        assert [e['epoch'] for e in events if e['state']=='train_temporal_readout' and e.get('arm')==name and 'epoch' in e]==list(range(1,21))
        for mode,chosen in entry['selected'].items():
            def rank(h):
                v=h['validation'];f=v['frame']['macro_f1'];o=v['window']['ordered']['metrics']['sequence_exact'];c=v['window']['count_accuracy']
                return (o,f,c) if mode=='ordered' else (f,o,c)
            assert chosen['epoch']==max(entry['history'],key=rank)['epoch']==selection['epochs'][name][mode]
            assert sha(raw/f'readouts/{name}-{mode}.pt')==chosen['sha256']
        selected_frame[name]=next(h['validation'] for h in entry['history'] if h['epoch']==entry['selected']['frame']['epoch'])
        def seqrank(x):return tuple(x['validation']['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall'))
        chosen=selection['arms'][name]
        assert len(chosen['candidates'])==36 and len(chosen['count_candidates'])==3
        assert max(chosen['candidates'],key=seqrank)['config']==chosen['hmm']
        assert max(chosen['count_candidates'],key=seqrank)['config']==chosen['count']
    assert selection['formal_arm']==max(training,key=lambda n:(selected_frame[n]['window']['ordered']['metrics']['sequence_exact'],selected_frame[n]['frame']['macro_f1'],selected_frame[n]['window']['count_accuracy']))
    for split in ('validation','test'):
        windows=load(raw/f'windows-{split}.json');expected=load(old/f'windows-{split}.json')
        assert windows==expected;mask=np.array([w['full_window'] for w in windows]);truth=np.array([len(w['actions']) for w in windows])
        count_support[split]=np.bincount(truth[mask],minlength=4)[1:].tolist()
        pred=load(raw/f'predictions-{split}.json')
        with np.load(raw/f'logits-{split}.npz') as arrays:
            labels=arrays['labels'];times=arrays['timestamps'];off=arrays['offsets']
            for name in training:
                entry=report[split][name];frame_check(arrays[name+'_frame'],labels,entry['frame'])
                frame_check(arrays[name+'_ordered_frame'],labels,entry['ordered_selected']['frame'])
                steady=np.ones(len(labels),bool)
                for a,b in zip(off[:-1],off[1:]):
                    changes=np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1
                    for j in changes:steady[a:b]&=np.abs(times[a:b]-(times[j]+times[j-1])/2)>.3+1e-8
                frame_check(arrays[name+'_frame'][steady],labels[steady],entry['frame_steady'])
                for mode in ('frame','ordered'):
                    output=arrays[name+('_window' if mode=='frame' else '_ordered_window')]
                    counts=output[:,14:17].argmax(-1)+1;slots=output[:,17:].reshape(-1,3,15).argmax(-1)
                    sequences_decoded=[]
                    for slot in slots.tolist():sequences_decoded.append(slot[:slot.index(14)] if 14 in slot else slot)
                    key='ordered' if mode=='frame' else 'ordered_selected';countkey='counts' if mode=='frame' else 'ordered_selected_counts'
                    assert pred[name][countkey]==counts.tolist()
                    assert [r['predicted_actions'] for r in pred[name][key]]==sequences_decoded
                    summary=entry['window'] if mode=='frame' else entry['ordered_selected']['window']
                    close((counts[mask]==truth[mask]).mean(),summary['count_accuracy']);close(np.abs(counts[mask]-truth[mask]).mean(),summary['count_mae'])
                    cm=np.zeros((3,3),int);np.add.at(cm,(truth[mask]-1,counts[mask]-1),1);assert cm.tolist()==summary['count_confusion']
                    pr=output[:,:14]>=0;gt=np.zeros_like(pr)
                    for i,w in enumerate(windows):gt[i,w['actions']]=True
                    tp=(pr[mask]&gt[mask]).sum(0);fp=(pr[mask]&~gt[mask]).sum(0);fn=(~pr[mask]&gt[mask]).sum(0)
                    close((2*tp/np.maximum(1,2*tp+fp+fn)).mean(),summary['presence_macro_f1'])
                for method in ('ordered','ordered_selected','hmm','predicted_count'):
                    rows=pred[name][method];assert [r['sample_id'] for r in rows]==[w['sample_id'] for w in windows]
                    assert [r['reference_actions'] for r in rows]==[w['actions'] for w in windows]
                    if method=='predicted_count':assert [len(r['predicted_actions']) for r in rows]==pred[name]['counts']
                    for subset in ('full12','original_protocol'):
                        rr=rows if subset=='original_protocol' else [r for r in rows if r['full_window']]
                        verify(rr,entry['sequences'][method][subset]);sequences+=1
                    if split=='validation' and method in ('hmm','predicted_count'):
                        candidates=selection['arms'][name]['candidates' if method=='hmm' else 'count_candidates']
                        cfg=selection['arms'][name]['hmm' if method=='hmm' else 'count']
                        assert entry['sequences'][method]['full12']==next(c['validation'] for c in candidates if c['config']==cfg)
        for name in training:assert report[split][name]['window']['ordered']==report[split][name]['sequences']['ordered']['full12']
    result=dict(passed=True,independent_sequence_summaries=sequences,frame_confusions=18,validation_model_selection=True,
        validation_grid_selection=True,paired_windows_unchanged=True,count_support=count_support,formal_arm=selection['formal_arm'])
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
