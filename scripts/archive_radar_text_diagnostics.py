"""Verify immutable inputs and export diagnostics without raw radar or full LLM weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--code',required=True)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    root,code,archive=Path(args.output),Path(args.code),Path(args.archive)
    status=json.loads((root/'status.json').read_text())
    assert status['state']=='complete'
    assert status['completed']==['extract','probes','smoothing','misses','prior','perturbations']
    manifest=json.loads((code/'source-manifest.json').read_text())
    assert all(sha(code/name)==digest for name,digest in manifest.items())
    pilot=Path(status['pilot'])
    assert all(sha(pilot/stage/'best.pt')==digest for stage,digest in status['checkpoint_sha256'].items())
    original=code.parent/'radar-text-code-20261010-v2/native-preflight.json'
    preflight=json.loads(original.read_text())
    assert all(sha(Path(name))==digest for name,digest in preflight['source_sha256'].items())
    index=json.loads((root/'feature-index.json').read_text())
    cache_manifest={}
    frame_reports=json.loads((root/'frame-classification.json').read_text())
    for split,records in index.items():
        values={key:[] for key in ('labels','timestamps','steady','short','predictions','logits')}
        offsets=[0]
        for row in records:
            path=root/'features'/row['cache']
            cache_manifest[row['cache']]=dict(bytes=path.stat().st_size,sha256=sha(path))
            with np.load(path,allow_pickle=False) as data:
                for key in values:
                    values[key].append(data[key])
                offsets.append(offsets[-1]+len(data['labels']))
        flat={key:np.concatenate(value) for key,value in values.items()}
        for mode in ('all_valid','steady'):
            mask=flat['labels']>=0
            if mode=='steady':mask &= flat['steady']
            matrix=np.bincount(flat['labels'][mask]*14+flat['predictions'][mask],minlength=196).reshape(14,14)
            assert matrix.tolist()==frame_reports[split][mode]['confusion_matrix']
        np.savez(root/f'frame-predictions-{split}.npz',offsets=np.array(offsets),**flat)
    (root/'feature-cache-manifest.json').write_text(json.dumps(cache_manifest,indent=2))
    verification=dict(state='verified',source_files=len(manifest),feature_caches=len(cache_manifest),
                       original_checkpoint_sha256=status['checkpoint_sha256'],
                       original_data_and_llm_sha256=preflight['source_sha256'],frame_confusion_recomputed=True,
                       note='Feature caches retained on server; exported frame labels/logits/predictions have no raw RD/MD.')
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    entries=[(path,'diagnostics/'+path.relative_to(root).as_posix()) for path in sorted(root.rglob('*')) if path.is_file() and 'features' not in path.relative_to(root).parts]
    entries += [(code/name,'provenance/'+name) for name in ('source-manifest.json','native-tests.log')]
    entries += [(code.parent/'radar-text-diagnostics-20261011-v1.tar.gz','source/radar-text-diagnostics-20261011-v1.tar.gz')]
    entries += [(path,'launcher/'+path.name) for path in root.with_name(root.name+'.logs').iterdir() if path.is_file()]
    entries += [(pilot/name,'original-pilot/'+name) for name in ('validation-evaluation.json','test-evaluation.json','stage1/training.json','align/training.json')]
    hashes={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path,name in entries}
    result_manifest=archive.with_name(archive.name+'.manifest.json')
    with result_manifest.open('x') as stream:json.dump(hashes,stream,indent=2)
    with tarfile.open(archive,'x:gz') as tar:
        for path,name in entries:
            assert not path.is_symlink()
            tar.add(path,arcname=name,recursive=False)
        tar.add(result_manifest,arcname='result-manifest.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries),verification=verification),indent=2))


if __name__=='__main__':
    main()
