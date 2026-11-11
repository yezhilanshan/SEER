"""Fixed-clock finite episodes; references remain outside the runtime boundary."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import numpy as np
import torch
from ..data import PHRASES
from ..memory import MemoryItem
from ..frontend import ContinuousMemoryStream


def render_reference(sequence, ongoing):
    if not sequence: return 'No annotated action has been observed yet.'
    return 'The person '+', then '.join(PHRASES[y]+(' (ongoing)' if active else '') for y,active in zip(sequence,ongoing))+'.'


def window_reference(instances, start, stop):
    if not 0<=start<stop: raise ValueError('invalid time window')
    sequence=[]; ongoing=[]; left=[]; intervals=[]
    for row in sorted(instances,key=lambda x:x['start']):
        if row['start']>=stop or row['stop']<=start: continue
        sequence.append(row['label']); ongoing.append(row['stop']>stop); left.append(row['start']<start)
        intervals.append([max(row['start'],start)-start,min(row['stop'],stop)-start])
    known=sum(b-a for a,b in intervals)
    return dict(sequence=sequence,ongoing=ongoing,left_censored=left,intervals=intervals,
                known_frames=known,unknown_frames=stop-start-known,text=render_reference(sequence,ongoing))


def make_panel(records, lengths=(4,8,16,32), fps=15):
    heldout={subject:max(r['file'] for r in records if r['subject']==subject)
             for subject in {r['subject'] for r in records if r['role']=='train'}}
    result={'fit':[],'dev':[],'validation':[]}
    for row in sorted(records,key=lambda r:r['file']):
        if row['role']=='validation' and row['subject']=='subject7': cohort='validation'
        elif row['role']=='train' and row['subject'] in {f'subject{i}' for i in range(1,7)}:
            cohort='dev' if row['file']==heldout[row['subject']] else 'fit'
        else: raise ValueError('forbidden test role/subject')
        for seconds in lengths:
            width=round(seconds*fps)
            for start in range(0,row['frames']-width+1,width):
                stop=start+width
                result[cohort].append(dict(uid=f"{row['file']}:{start}:{stop}",file=row['file'],subject=row['subject'],
                    cohort=cohort,cells_path=row['cells_path'],start=start,stop=stop,seconds=seconds,
                    reference=window_reference(row['instances'],start,stop)))
    return result


def natural_order_pairs(examples):
    bags=defaultdict(list)
    for e in sorted(examples,key=lambda x:x['uid']):
        if len(set(e['reference']['sequence']))<2: continue
        bags[(e['seconds'],tuple(sorted(e['reference']['sequence'])))].append(e)
    used=set(); pairs=[]
    for bag in sorted(bags):
        for a in bags[bag]:
            if a['uid'] in used: continue
            b=next((b for b in bags[bag] if b['uid'] not in used and b['uid']!=a['uid']
                    and b['reference']['sequence']!=a['reference']['sequence']),None)
            if b:
                used.update((a['uid'],b['uid'])); pairs.append([a['uid'],b['uid']])
    return pairs


def choose_pilot(examples, count=32, seed=42):
    if count<12 or any(e['cohort']!='fit' for e in examples): raise ValueError('pilot must be fit-only and large enough')
    ranked=sorted(examples,key=lambda e:hashlib.sha256(f"{seed}:{e['uid']}".encode()).hexdigest())
    selected=[]; seen=set(); missing=set(range(12))
    def add(e):
        if e['uid'] not in seen: selected.append(e); seen.add(e['uid'])
    while missing:
        best=max(ranked,key=lambda e:len(missing&set(e['reference']['sequence'])))
        gain=missing&set(best['reference']['sequence'])
        if not gain: raise ValueError('fit pilot cannot cover all 12 classes')
        add(best); missing-=gain
    repeated=next((e for e in ranked if len(e['reference']['sequence'])>len(set(e['reference']['sequence']))),None)
    if repeated is None: raise ValueError('pilot diagnostic requires repeated instances')
    add(repeated)
    pairs=natural_order_pairs(ranked)
    if not pairs: raise ValueError('pilot diagnostic requires a natural order pair')
    by_uid={e['uid']:e for e in ranked}
    for uid in pairs[0]: add(by_uid[uid])
    ongoing=next((e for e in ranked if any(e['reference']['ongoing'])),None)
    if ongoing is None: raise ValueError('pilot diagnostic requires ongoing coverage')
    add(ongoing)
    lengths=sorted({e['seconds'] for e in ranked}); index=0
    while len(selected)<count:
        seconds=lengths[index%len(lengths)]; index+=1
        candidate=next((e for e in ranked if e['seconds']==seconds and e['uid'] not in seen),None)
        if candidate: add(candidate)
        elif len(seen)>=len(ranked): raise ValueError('insufficient unique pilot windows')
    if len(selected)>count: raise ValueError('pilot coverage needs a larger pre-registered count')
    return selected


class RawAgeMemory:
    def __init__(self, items, observed_frames):
        self.levels=[[x for x in items if x.level==g] for g in range(3)]
        self.observed_frames=observed_frames

    def candidates(self, group):
        if group not in (0,1,2): raise ValueError('invalid group')
        return sorted([x for level in self.levels[:group+1] for x in level],key=lambda x:x.start_frame)


def raw_memory(cells, fps=15, max_seconds=32):
    if cells.ndim!=3 or cells.shape[1:]!=(4,64) or not 0<len(cells)<=max_seconds*fps or not torch.isfinite(cells).all():
        raise ValueError('finite bounded cells[T,4,64] required')
    cells=cells.detach().cpu().float().clone(); items=[]
    for start in range(0,len(cells),2):
        stop=min(start+2,len(cells)); values=torch.zeros(2,4,64); values[:stop-start]=cells[start:stop]
        weights=torch.tensor([1.,float(stop-start==2)])
        age=(len(cells)-stop)/fps
        level=2 if age>=12 else 1 if age>=4 else 0
        items.append(MemoryItem(values,weights,start,stop,level))
    return RawAgeMemory(items,len(cells))


def read_cells(episode):
    # Only these fields cross the runtime boundary. No labels, reference, or future frames.
    path,start,stop=episode['cells_path'],episode['start'],episode['stop']
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    cells=torch.from_numpy(np.array(array[start:stop],copy=True))
    if len(cells)!=stop-start: raise ValueError('incomplete episode')
    return cells


def memory_view(cells, view='raw', fps=15):
    if view=='raw': return raw_memory(cells,fps)
    if view!='compressed': raise ValueError('unknown memory view')
    stream=ContinuousMemoryStream(); stream.push(cells); return stream.snapshot()
