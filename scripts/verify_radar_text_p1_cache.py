"""Read-only native-cache audit and explicitly post-hoc train-majority diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.radar_text.p1 import EvidenceProbe,decode_outputs,probability_smooth,hmm_path,segment_paths,sequence_summary
from continuous_mmwave.radar_text.diagnostics import smooth_predictions


def load(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser()
    for k in ('results','pilot','output'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();root,pilot,out=map(Path,(args.results,args.pilot,args.output));out.mkdir(exist_ok=False)
    torch.set_num_threads(4);maxlen=load(root/'verification.json')['max_H_length'];selection=load(root/'selection.json')
    caches={}
    for s in ('train','validation','test'):
        # NPZ indexing rereads an entire member; materialize once before slicing.
        with np.load(root/f'features/{s}.npz') as archive:caches[s]={k:archive[k] for k in archive.files}
    windows={s:load(root/f'windows-{s}.json') for s in caches};saved=torch.load(pilot/'align/best.pt',map_location='cpu',weights_only=False)
    projector=nn.Sequential(nn.Linear(256,896),nn.GELU(),nn.Linear(896,896));projector.load_state_dict(saved['projector']);projector.eval()
    arrays={};pool_error=0.;projector_error=0.
    for s,cache in caches.items():
        hs=[cache['H'][l:r] for l,r in zip(cache['offsets'][:-1],cache['offsets'][1:])]
        z=np.stack([F.adaptive_avg_pool1d(torch.from_numpy(h).T,8).T.numpy() for h in hs])
        pool_error=max(pool_error,float(np.abs(z-cache['Z']).max()));assert np.allclose(z,cache['Z'],atol=5e-6,rtol=5e-6)
        with torch.no_grad():p=projector(torch.from_numpy(cache['Z'])).numpy()
        projector_error=max(projector_error,float(np.abs(p-cache['P']).max()));assert np.allclose(p,cache['P'],atol=5e-6,rtol=5e-6)
        padded=np.zeros((len(hs),maxlen,256),np.float32)
        for i,h in enumerate(hs):padded[i,:len(h)]=h
        arrays[s]={'H':padded.reshape(len(hs),-1),'Z':cache['Z'].reshape(len(hs),-1),'P':cache['P'].reshape(len(hs),-1)}
    stats_error=0.;predictions=0
    for name in load(root/'probe-training.json'):
        payload=torch.load(root/'probes'/f'{name}.pt',map_location='cpu',weights_only=True);kind='Z' if name=='independent-count' else name.split('-')[0]
        tr=arrays['train'][kind];mean=tr.mean(0);std=tr.std(0).clip(1e-5)
        stats_error=max(stats_error,float(np.abs(mean-payload['mean'].numpy()).max()),float(np.abs(std-payload['std'].numpy()).max()))
        np.testing.assert_allclose(mean,payload['mean'],atol=1e-6,rtol=1e-6);np.testing.assert_allclose(std,payload['std'],atol=1e-6,rtol=1e-6)
        model=EvidenceProbe(payload['width'],payload['nonlinear'],outputs=3 if payload['count_only'] else 62);model.load_state_dict(payload['state']);model.eval()
        for s in ('validation','test'):
            x=(arrays[s][kind]-mean)/std
            with torch.no_grad():value=model(torch.from_numpy(x))
            pred=load(root/f'probe-predictions-{s}.json')[name]
            if payload['count_only']:
                assert (value.argmax(-1).numpy()+1).tolist()==pred['count'];np.testing.assert_allclose(value,pred['logits'],atol=5e-5,rtol=5e-5)
            else:
                presence,count,ordered=decode_outputs(value);assert presence.tolist()==pred['presence'] and count.tolist()==pred['count']
                assert ordered==[r['predicted_actions'] for r in pred['ordered']]
            predictions+=len(x)
    prior=load(root/'transition-prior.json');transition=np.array(prior['log_transition']);start=np.array(prior['log_start'])
    majority=int(np.bincount([len(w['actions']) for w in windows['train']],minlength=4)[1:].argmax()+1)
    diagnostic={};decoded=0
    for s in ('validation','test'):
        cache=caches[s];expected=load(root/f'calibration-predictions-{s}.json');counts=load(root/f'probe-predictions-{s}.json')['independent-count']['count'];rows=[]
        assert majority==load(root/f'probe-predictions-{s}.json')['majority-count']
        for i,(left,right) in enumerate(zip(cache['offsets'][:-1],cache['offsets'][1:])):
            logits=cache['logits'][left:right];times=cache['timestamps'][left:right];p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            for name in ('probability','hmm'):
                cfg=selection[name];pr=probability_smooth(p,times,cfg['seconds']);path=hmm_path(np.log(pr.clip(1e-12)),transition,start,cfg['weight'])
                seq=smooth_predictions(path,times,minimum_seconds=cfg['minimum']);assert seq==expected[name][i]['predicted_actions']
            cfg=selection['count'];pr=probability_smooth(p,times,cfg['seconds']);paths=segment_paths(np.log(pr.clip(1e-12)),times,cfg['minimum'],cfg['weight']*transition,cfg['weight']*start)
            assert paths[counts[i]]==expected['predicted-count'][i]['predicted_actions']
            assert paths[len(windows[s][i]['actions'])]==expected['ORACLE-GT-count'][i]['predicted_actions']
            rows.append(dict(sample_id=windows[s][i]['sample_id'],full_window=windows[s][i]['full_window'],reference_actions=windows[s][i]['actions'],predicted_actions=paths[majority]));decoded+=1
        diagnostic[s]={subset:sequence_summary([r for r in rows if subset=='original_protocol' or r['full_window']]) for subset in ('full12','original_protocol')}
        (out/f'majority-count-predictions-{s}.json').write_text(json.dumps(rows,indent=2))
    (out/'majority-count-diagnostic.json').write_text(json.dumps(dict(post_hoc=True,reason='Audit found severe count imbalance after primary results; no new tuning or main-method selection.',
        majority_from_train=majority,decoder=selection['count'],results=diagnostic),indent=2))
    audit=dict(state='passed',pool_max_abs_error=pool_error,projector_max_abs_error=projector_error,train_statistics_max_abs_error=stats_error,
               probe_window_predictions_reproduced=predictions,calibration_windows_reproduced=decoded,
               cache_nodes='H native BiGRU, Z adaptive_avg_pool1d(H,8), P original projector(Z)',
               limits='Decoder reconstruction uses production implementation; optimal segment DP independently brute-force tested. No encoder/LLM training or raw-output changes.')
    (out/'server-cache-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':main()
