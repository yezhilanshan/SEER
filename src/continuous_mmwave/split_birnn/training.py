"""Recording-uniform training and full-state deterministic resume contracts."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from .model import SplitBiGRU,BiGRUBaseline
from .loss import combo_loss


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False


class RecordingSampler:
    def __init__(self,rows,*,seed,min_length=100,max_length=1000):
        self.rows=sorted(rows,key=lambda r:r['recording_id'])
        if not self.rows or len({r['recording_id'] for r in self.rows})!=len(self.rows):
            raise ValueError('nonempty unique recording sources required')
        self.min_length,self.max_length=min_length,max_length
        if not 1<=min_length<=max_length: raise ValueError('invalid snippet range')
        self.rng=np.random.default_rng(seed);self.arrays=[]
        for r in self.rows:
            p=Path(r['path'])
            if 'npz_sha256' in r and hashlib.sha256(p.read_bytes()).hexdigest()!=r['npz_sha256']:
                raise ValueError('recording digest mismatch')
            with np.load(p) as d: md,y=d['md'].astype(np.float32),d['labels'].astype(np.int64)
            if md.shape!=(len(y),128) or len(y)<max_length or not np.isfinite(md).all():
                raise ValueError('bad MD or source shorter than maximum sampled length')
            if np.any((y!=-100)&((y<0)|(y>=9))): raise ValueError('invalid class')
            self.arrays.append((md,y))
        self.record_counts={r['recording_id']:0 for r in self.rows}
        self.subject_counts={r['subject_id']:0 for r in self.rows}
        self.class_frame_counts=np.zeros(9,dtype=np.int64)
        self.sampled_frames,self.ignored_frames,self.batches=0,0,0

    def sample(self,batch_size):
        if batch_size<1: raise ValueError('positive batch required')
        length=int(self.rng.integers(self.min_length,self.max_length+1))
        choices=self.rng.choice(len(self.rows),batch_size,replace=batch_size>len(self.rows))
        xs,ys,sources=[],[],[]
        for i in choices:
            r=self.rows[i];md,y=self.arrays[i];start=int(self.rng.integers(0,len(y)-length+1))
            ys.append(y[start:start+length]);xs.append(md[start:start+length])
            sources.append({'recording_id':r['recording_id'],'subject_id':r['subject_id'],'start_frame':start,'frames':length})
            self.record_counts[r['recording_id']]+=1;self.subject_counts[r['subject_id']]+=1
        labels=np.stack(ys); valid=labels!=-100
        self.class_frame_counts+=np.bincount(labels[valid],minlength=9)
        self.sampled_frames+=labels.size;self.ignored_frames+=int((~valid).sum());self.batches+=1
        return torch.from_numpy(np.stack(xs)),torch.from_numpy(labels),sources

    def state_dict(self):
        return copy.deepcopy({'rng':self.rng.bit_generator.state,'record_counts':self.record_counts,
            'subject_counts':self.subject_counts,'class_frame_counts':self.class_frame_counts.tolist(),
            'sampled_frames':self.sampled_frames,'ignored_frames':self.ignored_frames,'batches':self.batches,
            'source_order':[r['recording_id'] for r in self.rows],'min_length':self.min_length,'max_length':self.max_length})

    def load_state_dict(self,state):
        if (state['source_order']!=[r['recording_id'] for r in self.rows]
                or (state['min_length'],state['max_length'])!=(self.min_length,self.max_length)):
            raise ValueError('sampler source/order/length protocol changed')
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.record_counts=copy.deepcopy(state['record_counts']);self.subject_counts=copy.deepcopy(state['subject_counts'])
        self.class_frame_counts=np.array(state['class_frame_counts'],dtype=np.int64)
        for k in ('sampled_frames','ignored_frames','batches'):setattr(self,k,state[k])

    def expected_distribution(self):
        n=len(self.rows)
        return {'recording_probabilities':{r['recording_id']:1/n for r in self.rows},
                'subject_probabilities':{s:sum(r['subject_id']==s for r in self.rows)/n for s in self.subject_counts},
                'policy':'recording_uniform_without_replacement_within_batch_when_possible',
                'length':'one_uniform_integer_length_per_batch','class_resampling':False}


def make_training_state(kind,device):
    if kind not in ('split','baseline'):raise ValueError('unknown model kind')
    net=(SplitBiGRU() if kind=='split' else BiGRUBaseline()).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    return net,optimizer,scheduler


def optimize_step(net,optimizer,scheduler,batch,*,device):
    x,y,sources=batch;x=x.to(device);y=y.to(device)
    if not (y!=-100).any():raise ValueError('no supervised frames in training batch')
    net.train();optimizer.zero_grad(set_to_none=True)
    if isinstance(net,SplitBiGRU):
        out=net.forward_sequence(x);fl,fp=combo_loss(out['forward'],y);cl,cp=combo_loss(out['corrected'],y)
        loss=fl+cl;branches={'forward_loss':float(fl.detach()),'corrected_loss':float(cl.detach()),
                           'forward_cce':float(fp['cce'].detach()),'forward_dice':float(fp['dice'].detach()),
                           'corrected_cce':float(cp['cce'].detach()),'corrected_dice':float(cp['dice'].detach())}
        groups={'cnn':net.cnn,'forward_grus':net.forward_grus,'backward_grus':net.backward_grus,'head':net.head}
    else:
        loss,parts=combo_loss(net(x),y);branches={'baseline_loss':float(loss.detach()),
            'baseline_cce':float(parts['cce'].detach()),'baseline_dice':float(parts['dice'].detach())}
        groups={'cnn':net.cnn,'bidirectional_gru':net.gru,'head':net.head}
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward()
    gradients={k:sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) for k,m in groups.items()}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('bad gradients')
    optimizer.step();scheduler.step()
    return {'loss':float(loss.detach()),**branches,'valid_frames':int((y!=-100).sum()),
            'frames_per_sample':x.shape[1],'sources':sources,'gradient_group_sums':gradients,
            'learning_rate':optimizer.param_groups[0]['lr']}


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_checkpoint(path,net,optimizer,scheduler,sampler,*,step,extra):
    path=Path(path);temp=path.with_suffix('.writing')
    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'sampler':sampler.state_dict(),'rng':capture_rng(),'step':step,'extra':extra},temp)
    os.replace(temp,path)


def load_checkpoint(path,net,optimizer,scheduler,sampler):
    state=torch.load(path,map_location='cpu',weights_only=False)
    net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])
    return state


def bn_snapshot(net):
    return {n:{'mean':m.running_mean.detach().cpu().tolist(),'variance':m.running_var.detach().cpu().tolist(),
               'batches':int(m.num_batches_tracked)} for n,m in net.cnn.named_modules() if isinstance(m,torch.nn.BatchNorm1d)}


@torch.no_grad()
def bn_feature_diagnostic(net,x):
    device=next(net.parameters()).device;x=x.to(device)
    cnn=copy.deepcopy(net.cnn);cnn.eval();eval_features=cnn(x)
    cnn.train();train_features=cnn(x)
    return {'eval_mode_uses_running_statistics':True,'training_statistics_span_batch_and_time':True,
            'train_mean':float(train_features.mean()),'train_std':float(train_features.std()),
            'eval_mean':float(eval_features.mean()),'eval_std':float(eval_features.std()),
            'train_eval_difference_l2':float((train_features-eval_features).norm()),
            'finite':bool(torch.isfinite(train_features).all() and torch.isfinite(eval_features).all()),
            'actual_model_not_mutated':True}
