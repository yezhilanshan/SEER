"""Read-only server audit: raw RD -> cached frame features -> saved readout logits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from continuous_mmwave.radar_text.rd_transfer import make_backbone,normalize_rd,TemporalReadout
from continuous_mmwave.radar_text.runner import datasets,read_settings
from run_rd_only_xfi_transfer import sha,padded

def main():
    p=argparse.ArgumentParser()
    for k in ('output','pilot','xfi','destination'):p.add_argument('--'+k,required=True)
    args=p.parse_args();root=Path(args.output);torch.set_num_threads(4)
    sets,_=datasets(read_settings(Path(args.pilot)/'align.yaml'),('train','validation','test'))
    manifest=json.loads((root/'feature-cache-manifest.json').read_text());checks=[];stats=[]
    for name in ('R0','R1','R2'):
        state=torch.load(root/f'backbones/{name}.pt',map_location='cpu',weights_only=True)
        backbone=make_backbone('stage1-cnn',{k[len('network.'):]:v for k,v in state.items()}) if name=='R0' else make_backbone('random-resnet')
        backbone.load_state_dict(state,strict=True);backbone.cuda().train()
        assert all(not m.training for m in backbone.modules())
        before={k:v.clone() for k,v in backbone.state_dict().items()}
        for split in ('train','validation','test'):
            path=root/f'features/{split}-{name}.npz';assert sha(path)==manifest[f'{split}-{name}']['sha256']
            with np.load(path) as arrays:cache={k:arrays[k] for k in arrays.files}
            payloads={mode:torch.load(root/f'readouts/{name}-{mode}.pt',map_location='cpu',weights_only=True) for mode in ('frame','ordered')}
            if split=='train':
                mean=cache['X'].mean(0);std=cache['X'].std(0).clip(1e-5)
                for mode,payload in payloads.items():
                    assert np.array_equal(mean,payload['mean'].numpy()) and np.array_equal(std,payload['std'].numpy())
                    stats.append(dict(arm=name,selector=mode,train_statistics_exact=True))
            indices=sorted({0,len(sets[split])//2,len(sets[split])-1})
            for index in indices:
                sample=sets[split][index];raw=sample['rd'].unsqueeze(0).cuda();n=raw.shape[1]
                with torch.no_grad():
                    rd=normalize_rd(raw,torch.tensor([n],device='cuda'))[0]
                    value=torch.cat([backbone(v) for v in rd.split(64)]).cpu().numpy()
                a,b=cache['offsets'][index:index+2];expected=cache['X'][a:b]
                np.testing.assert_allclose(value,expected,rtol=5e-6,atol=5e-6)
                assert np.array_equal(sample['labels'].numpy(),cache['labels'][a:b]) and np.array_equal(sample['timestamps'].numpy(),cache['timestamps'][a:b])
                checks.append(dict(arm=name,split=split,index=index,frame_max_error=float(np.max(np.abs(value-expected)))))
            if split!='train':
                with np.load(root/f'logits-{split}.npz') as arrays:expected={k:arrays[k] for k in arrays.files if k.startswith(name+'_')}
                for mode,payload in payloads.items():
                    model=TemporalReadout(payload['width']).cuda().eval();model.load_state_dict(payload['state'])
                    x,lengths,_=padded(cache,payload['mean'].numpy(),payload['std'].numpy(),'cuda')
                    # Same evaluation batch32 avoids backend-dependent batch-size roundoff.
                    with torch.no_grad():
                        _,_,f,o=model(x[:32],lengths[:32]);frames=np.concatenate([f[i,:n].cpu().numpy() for i,n in enumerate(lengths[:32].tolist())])
                    fk=name+('_frame' if mode=='frame' else '_ordered_frame');wk=name+('_window' if mode=='frame' else '_ordered_window')
                    np.testing.assert_allclose(frames,expected[fk][:len(frames)],rtol=5e-5,atol=5e-5)
                    np.testing.assert_allclose(o.cpu().numpy(),expected[wk][:32],rtol=5e-5,atol=5e-5)
                    checks.append(dict(arm=name,split=split,selector=mode,readout_windows=32,
                        frame_max_error=float(np.max(np.abs(frames-expected[fk][:len(frames)]))),window_max_error=float(np.max(np.abs(o.cpu().numpy()-expected[wk][:32])))))
                    del x,model
            del cache
        assert all(torch.equal(v,before[k]) for k,v in backbone.state_dict().items())
    result=dict(passed=True,raw_RD_windows=27,readout_windows=384,train_statistics=stats,checks=checks,
        caches_sha_verified=9,backbones_BN_eval_bitwise_unchanged=True)
    Path(args.destination).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('checks','train_statistics')}))

if __name__=='__main__':main()
