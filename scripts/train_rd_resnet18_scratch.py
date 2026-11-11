"""End-to-end RD ResNet18; train/validation only until final selection is locked."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer,accumulation_divisor,optimizer_update,save_model,load_model
from continuous_mmwave.radar_text.data import OfflineCaptionDataset,RecordingWindowSampler,collate_windows,load_phrases
from continuous_mmwave.radar_text.diagnostics import classification_metrics,confusion
from continuous_mmwave.data_parrad import assert_disjoint_splits
from continuous_mmwave.radar_text.signed_log_ablation import parameter_sha,PairedWindowSampler,validation_diagnostics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def label_counts(data):
    counts=np.zeros(14,np.int64);last=None
    for rid,a,b,_,_ in data.windows:
        if rid!=last:
            row=data.recordings.rows[rid];p=Path(row['path']);p=p if p.is_absolute() else data.recordings.manifest.parent/p
            with np.load(p,allow_pickle=False) as arrays:labels=arrays['labels']
            last=rid
        x=labels[a:b];counts+=np.bincount(x[x>=0],minlength=14)
    if not np.all(counts>0):raise ValueError('train windows lack one or more of the14classes')
    return counts


@torch.no_grad()
def evaluate(model,data,batch_size,device,logits_path=None):
    model.eval();matrix=np.zeros((14,14),np.int64);by_subject=defaultdict(lambda:np.zeros((14,14),np.int64));loss_sum=0.;frames=0;logits_all=[];labels_all=[];offsets=[0]
    loader=DataLoader(data,batch_size=batch_size,collate_fn=collate_windows,num_workers=0)
    for batch in loader:
        logits=model(batch['rd'].to(device),batch['lengths'].to(device));labels=batch['labels'].to(device)
        loss_sum+=float(F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum'));frames+=int((labels>=0).sum())
        for i,(n,subject) in enumerate(zip(batch['lengths'].tolist(),batch['subjects'])):
            raw=logits[i,:n].float().cpu().numpy();actual=batch['labels'][i,:n].numpy();cm=confusion(actual,raw.argmax(-1));matrix+=cm;by_subject[subject]+=cm
            if logits_path:logits_all.append(raw);labels_all.append(actual);offsets.append(offsets[-1]+n)
    result=classification_metrics(matrix);result['unweighted_ce']=loss_sum/max(1,frames)
    result['by_subject']={s:classification_metrics(cm) for s,cm in sorted(by_subject.items())}
    result['subject_macro_f1']=float(np.mean([v['macro_f1'] for v in result['by_subject'].values()]))
    if logits_path:np.savez(logits_path,logits=np.concatenate(logits_all),labels=np.concatenate(labels_all),offsets=np.array(offsets))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--validation-only',action='store_true',help='Do not load or evaluate test values')
    p.add_argument('--paired-initial',type=Path,help='Shared tensor/RNG initialization for controlled paired training')
    args=p.parse_args()
    config=json.loads(args.config.read_text());root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=datetime.now().astimezone().isoformat();clock=time.monotonic();options=config['training'];device=args.device
    def status(state,**more):
        value=dict(state=state,pid=os.getpid(),started=started,elapsed_seconds=time.monotonic()-clock,**more);dump(root/'status.json',value);print(json.dumps(value),flush=True)
    try:
        status('preflight');source=Path(__file__).resolve().parents[1];sources=json.loads((source/'source-manifest.json').read_text())
        assert all(sha(source/name)==digest for name,digest in sources.items()),'source capsule changed'
        split=Path(config['split_directory']);policy=json.loads((split/'protocol.json').read_text());sealed=json.loads((split/'artifact-manifest.json').read_text());data_root=Path(policy['dataset'])
        for name,entry in sealed.items():assert sha(split/name)==entry['sha256']
        rows={s:[json.loads(l) for l in (split/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation','test')};assert_disjoint_splits(**rows)
        subjects={s:sorted({r['subject_id'] for r in rr}) for s,rr in rows.items()};assert subjects==policy['subjects']
        for audit in json.loads((data_root/'recording-audit.json').read_text()):
            st=Path(audit['path']).stat();assert dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)==audit['stamp']
        torch.set_num_threads(4);seed=options['seed'];torch.manual_seed(seed);np.random.seed(seed)
        if device=='cuda':assert torch.cuda.is_available();torch.cuda.manual_seed_all(seed)
        phrases=load_phrases(data_root/'action_phrases.json');sets={s:OfflineCaptionDataset(split/f'{s}.jsonl',phrases,**policy['window_protocol']) for s in ('train','validation')}
        for s,d in sets.items():assert len(d)==policy['expected_statistics'][s]['windows']
        model=RDResNetPretrainer(**config['model']).to(device);assert all(p.requires_grad for p in model.parameters())
        if args.paired_initial:
            shared=torch.load(args.paired_initial,map_location='cpu',weights_only=True)
            model.load_state_dict(shared['state_dict'],strict=True)
            assert parameter_sha(model)==shared['parameter_sha'],'paired initial parameters differ'
            torch.set_rng_state(shared['cpu_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(shared['cuda_rng'])
            shutil.copyfile(args.paired_initial,root/'initial-parameters.pt')
        initial_parameter_sha=parameter_sha(model)
        dump(root/'initialization.json',dict(parameter_sha256=initial_parameter_sha,
            shared_file_sha256=sha(args.paired_initial) if args.paired_initial else None,
            normalization=model.encoder.normalization,seed=seed,verified_before_training=True))
        counts=label_counts(sets['train']);weights=torch.tensor(counts.sum()/(14*counts),device=device,dtype=torch.float32)
        optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=options['encoder_lr']),dict(params=model.classifier.parameters(),lr=options['classifier_lr'])],weight_decay=options['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=options['epochs'],eta_min=1e-6)
        scaler=torch.amp.GradScaler('cuda',enabled=config['model']['amp'] and device=='cuda',init_scale=1024.)
        save_model(root/'initial.pt',model,epoch=0,subjects=subjects)
        provenance=dict(config=config,split_protocol=policy,subjects=subjects,source_sha256=sources,initial_checkpoint_sha256=sha(root/'initial.pt'),
            torch_version=str(torch.__version__),device=torch.cuda.get_device_name() if device=='cuda' else device,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            initialization='Kaiming random single-channel ResNet18; no pretrained checkpoint loaded',input='RD only; MD retained by loader but never passed to model',
            label_counts_train_windows=counts.tolist(),class_weights=weights.cpu().tolist(),selection='mean of validation subjects Macro-F1; pooled Macro-F1 then negative unweighted CE break ties',
            evaluation_protocol='frame observations inside eligible12s/6s/min8s/max3action windows; overlaps count repeatedly, not unique recording frames',smoke=args.smoke,test_values_accessed=False,
            initial_parameter_sha256=initial_parameter_sha,paired_initial_sha256=sha(args.paired_initial) if args.paired_initial else None,
            window_ids_sha256={s:hashlib.sha256('\n'.join(w[3] for w in d.windows).encode()).hexdigest() for s,d in sets.items()},
            validation_only=args.validation_only,normalization=model.encoder.normalization,
            numerics=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                note='Identical settings, weights and sampling are enforced; bitwise CUDA determinism is not claimed.'))
        dump(root/'provenance.json',provenance);dump(root/'config.json',config)
        sampler=PairedWindowSampler(sets['train'],seed) if args.paired_initial else RecordingWindowSampler(sets['train'])
        loader=DataLoader(sets['train'],batch_size=options['batch_size'],sampler=sampler,collate_fn=collate_windows,num_workers=0)
        history=[];best=None;steps=0;overflows=0;optimizer.zero_grad(set_to_none=True)
        epochs=1 if args.smoke else options['epochs']
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.;batches=0;epoch_started=time.monotonic()
            for index,batch in enumerate(loader):
                before=model.encoder.backbone.conv1.weight.detach().clone() if steps==0 else None
                logits=model(batch['rd'].to(device),batch['lengths'].to(device));labels=batch['labels'].to(device)
                loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),weight=weights,ignore_index=-100)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                group=accumulation_divisor(index,len(loader),options['gradient_accumulation'])
                scaler.scale(loss/group).backward();loss_sum+=float(loss.detach());batches+=1
                update=(index+1)%options['gradient_accumulation']==0 or index+1==len(loader);updated=False
                if update:
                    updated,norm=optimizer_update(model,optimizer,scaler,options['grad_clip'])
                    if updated and steps==0:
                        grad={name:float(param.grad.detach().norm()) for name,param in model.named_parameters() if param.grad is not None}
                        assert len(grad)==sum(1 for _ in model.parameters()) and all(np.isfinite(list(grad.values())))
                    if updated:steps+=1
                    else:
                        overflows+=1;status('amp_overflow_skipped',epoch=epoch,batch=index+1,optimizer_steps=steps,overflows=overflows,new_scale=scaler.get_scale())
                    if updated and steps==1:
                        delta=float((model.encoder.backbone.conv1.weight.detach()-before).abs().max());assert delta>0,'backbone failed to update'
                        proof=dict(backbone_max_abs_update=delta,gradient_norms=grad,optimizer_steps=steps,all_parameters_trainable=True,bn_updates=int(model.encoder.backbone.bn1.num_batches_tracked),max_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
                        dump(root/'gradient-proof.json',proof);save_model(root/'first-update.pt',model,epoch=epoch,subjects=subjects,optimizer_steps=steps)
                        status('training',epoch=epoch,batch=index+1,batches=len(loader),optimizer_steps=steps,loss=float(loss.detach()),backbone_update=delta)
                    optimizer.zero_grad(set_to_none=True)
                if (index+1)%options['log_every']==0:status('training',epoch=epoch,batch=index+1,batches=len(loader),optimizer_steps=steps,mean_train_loss=loss_sum/batches)
                if updated and steps%options['recovery_every_steps']==0:
                    save_model(root/'recovery.pt',model,epoch=epoch,batch=index+1,subjects=subjects,optimizer_steps=steps,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),scheduler=scheduler.state_dict(),note='partial epoch; not an exact mid-epoch resume checkpoint')
                if args.smoke and steps>=2:break
            if not steps:raise ValueError('epoch completed without any successful optimizer update')
            status('validation',epoch=epoch,optimizer_steps=steps,amp_overflows=overflows)
            if args.smoke:
                # Small real held-out batch; complete validation remains a formal-run operation.
                from torch.utils.data import Subset
                validation=evaluate(model,Subset(sets['validation'],range(2)),2,device)
            else:validation=evaluate(model,sets['validation'],options['batch_size'],device)
            score=(validation['subject_macro_f1'],validation['macro_f1'],-validation['unweighted_ce'])
            history.append(dict(epoch=epoch,train_loss=loss_sum/batches,validation=validation,optimizer_steps=steps,amp_overflows=overflows,epoch_seconds=time.monotonic()-epoch_started,lrs=[g['lr'] for g in optimizer.param_groups]))
            if args.paired_initial:
                dump(root/'sampler-orders.json',sampler.orders)
                dump(root/'sampler-sha256.json',[hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest() for order in sampler.orders])
            dump(root/'history.json',history)
            if best is None or score>best:
                best=score;save_model(root/'best.pt',model,epoch=epoch,subjects=subjects,selection_score=list(score));dump(root/'best.json',dict(epoch=epoch,score=list(score),checkpoint_sha256=sha(root/'best.pt')))
            scheduler.step();save_model(root/'last.pt',model,epoch=epoch,subjects=subjects,optimizer_steps=steps,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],config=config)
            status('epoch_complete',epoch=epoch,optimizer_steps=steps,train_loss=loss_sum/batches,validation_subject_macro_f1=validation['subject_macro_f1'],validation_pooled_macro_f1=validation['macro_f1'])
        if args.smoke:return status('smoke_complete',optimizer_steps=steps,test_values_accessed=False)
        selected=json.loads((root/'best.json').read_text());dump(root/'selection.json',dict(locked_at=datetime.now().astimezone().isoformat(),**selected,test_values_accessed=False))
        model,_=load_model(root/'best.pt',device);status('final_evaluation',selected_epoch=selected['epoch'])
        results={}
        for s in (('validation',) if args.validation_only else ('validation','test')):
            d=sets[s] if s in sets else OfflineCaptionDataset(split/f'{s}.jsonl',phrases,**policy['window_protocol'])
            assert len(d)==policy['expected_statistics'][s]['windows']
            results[s]=evaluate(model,d,options['batch_size'],device,root/f'logits-{s}.npz')
            dump(root/f'windows-{s}.json',[dict(sample_id=w[3],start=w[1],stop=w[2],actions=w[4],subject=d.recordings.rows[w[0]]['subject_id']) for w in d.windows])
        dump(root/'validation-diagnostics.json',validation_diagnostics(sets['validation'],root/'logits-validation.npz'))
        if not args.validation_only:
            for group,ss in policy['test_subgroups'].items():
                matrix=np.sum([results['test']['by_subject'][s]['confusion_matrix'] for s in ss],axis=0)
                results['test'][group]=classification_metrics(matrix)
        dump(root/'results.json',results);dump(root/'artifact-manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.iterdir() if p.is_file() and p.name!='status.json'})
        status('complete',epochs=epochs,optimizer_steps=steps,selected_epoch=selected['epoch'],
            validation_subject_macro_f1=results['validation']['subject_macro_f1'],test_values_accessed=not args.validation_only)
    except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
