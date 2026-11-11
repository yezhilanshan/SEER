"""Complete recordings for inference; interval annotations only build offline targets."""
import json
from pathlib import Path
import numpy as np
from continuous_mmwave.harrad.preprocess import sha256
from continuous_mmwave.harrad.data import CLASSES

PHRASES=('stood up','sat down','left the room','entered the room','put on clothing','took off clothing',
         'drummed','shook their body','swiped left','swiped right','gave a thumbs down','gave a thumbs up')


def query_cutoffs(frames, period_frames, include_tail=True):
    if frames<=0 or period_frames<=0: raise ValueError('positive frame counts required')
    points=list(range(period_frames,frames+1,period_frames))
    if include_tail and (not points or points[-1]!=frames): points.append(frames)
    return points


def prefix_reference(instances, cutoff):
    if cutoff<=0: raise ValueError('positive cutoff required')
    sequence=[]; ongoing=[]; phrases=[]; known=0; previous=0
    for row in sorted(instances,key=lambda x:x['start']):
        start,end,label=row['start'],row['stop'],row['label']
        if not previous<=start<end or label not in range(len(CLASSES)): raise ValueError('invalid annotations')
        previous=end
        if start>=cutoff: continue
        sequence.append(label); active=end>cutoff; ongoing.append(active)
        known+=min(end,cutoff)-start
        phrases.append(PHRASES[label]+(' (ongoing)' if active else ''))
    text='The person '+', then '.join(phrases)+'.' if phrases else 'No annotated action has been observed yet.'
    return dict(sequence=sequence,ongoing=ongoing,text=text,known_frames=known,unknown_frames=cutoff-known)


def load_records(manifest_path, expected_sha256=None):
    manifest_path=Path(manifest_path)
    if expected_sha256 and sha256(manifest_path)!=expected_sha256.lower(): raise ValueError('manifest hash mismatch')
    rows=json.loads(manifest_path.read_text(encoding='utf-8-sig')); seen=set()
    if not rows: raise ValueError('empty manifest')
    for row in rows:
        role=row['role']; subject=row['subject']
        if role not in ('train','validation') or subject not in {f'subject{i}' for i in range(1,8)}:
            raise ValueError('forbidden test role/subject')
        if role!=('validation' if subject=='subject7' else 'train'): raise ValueError('subject split mismatch')
        if row['file'] in seen: raise ValueError('duplicate recording')
        seen.add(row['file']); path=Path(row['cells_path'])
        if not path.is_absolute(): path=manifest_path.parent/path
        expected=row['artifacts']['cells.npy']['sha256'].lower()
        if sha256(path)!=expected: raise ValueError('cells hash mismatch')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=(row['frames'],4,64) or values.dtype!=np.float32: raise ValueError('invalid cells shape/dtype')
        for start in range(0,len(values),1024):
            if not np.isfinite(values[start:start+1024]).all(): raise ValueError('nonfinite cells')
        if any(x['stop']>row['frames'] for x in row['instances']): raise ValueError('annotation beyond recording')
        prefix_reference(row['instances'],row['frames'])
        row['cells_path']=str(path.resolve())
    return rows


def runtime_record(row):
    """Explicitly remove all annotation fields at the runtime boundary."""
    return {k:row[k] for k in ('file','frames','cells_path')}


def rebuild_memory(record, cutoff, config):
    import torch
    from .frontend import ContinuousMemoryStream
    if not 0<cutoff<=record['frames']: raise ValueError('invalid observation cutoff')
    values=np.load(record['cells_path'],mmap_mode='r',allow_pickle=False)
    stream=ContinuousMemoryStream(config.capacities,config.group_size)
    for start in range(0,cutoff,256):
        stream.push(torch.from_numpy(np.array(values[start:min(start+256,cutoff)],copy=True)))
    return stream.snapshot()


class PrefixExamples:
    """Lazy exact-prefix reconstruction; annotations never enter rebuild_memory."""
    def __init__(self, records, tokenizer, prompt_ids, config):
        self.records=records; self.tokenizer=tokenizer; self.prompt_ids=prompt_ids; self.config=config
        self.index=[(i,t) for i,r in enumerate(records) for t in query_cutoffs(r['frames'],config.query_frames)]

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        import torch
        i,cutoff=self.index[index]; row=self.records[i]
        reference=prefix_reference(row['instances'],cutoff)
        ids=self.tokenizer.encode(reference['text'],add_special_tokens=False)+[self.tokenizer.eos_token_id]
        if len(ids)>self.config.max_answer_tokens: raise ValueError('answer budget exceeded; refuse to truncate supervision')
        return dict(memory=rebuild_memory(runtime_record(row),cutoff,self.config),prompt_ids=self.prompt_ids,
                    answer_ids=torch.tensor([ids],dtype=torch.long),uid=f"{row['file']}:{cutoff}",reference=reference)
