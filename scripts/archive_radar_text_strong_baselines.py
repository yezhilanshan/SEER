"""Seal matched-window results while retaining fresh features and radar on server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--code',required=True)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    root,code,archive=map(Path,(args.output,args.code,args.archive))
    status=json.loads((root/'status.json').read_text())
    assert status['state']=='complete' and status['completed']==['validation','test']
    provenance=json.loads((root/'provenance.json').read_text())
    assert provenance['trainable_parameters']==0
    assert all(sha(name)==expected for name,expected in provenance['original_input_sha256'].items())
    assert all(sha(code/name)==expected for name,expected in provenance['source_sha256'].items())
    features=json.loads((root/'feature-cache-manifest.json').read_text())
    assert all(sha(root/row['path'])==row['sha256'] for row in features.values())
    for split in ('validation','test'):
        windows=json.loads((root/f'windows-{split}.json').read_text())
        predictions=json.loads((root/f'predictions-{split}.json').read_text())
        expected=[row['sample_id'] for row in windows]
        assert len(set(expected))==len(expected)
        assert [row['sample_id'] for row in predictions['B3']]==expected
        for method in ('B0','B1','B2'):
            for rows in predictions[method].values():assert [row['sample_id'] for row in rows]==expected
    entries=[(p,'experiment/'+p.relative_to(root).as_posix()) for p in sorted(root.rglob('*'))
             if p.is_file() and 'features' not in p.relative_to(root).parts]
    entries += [(code/name,'provenance/'+name) for name in ('source-manifest.json','native-tests.log')]
    entries += [(root.with_name(root.name+'.logs')/'worker.log','launcher/worker.log')]
    capsule=code.parent/'radar-text-strong-baselines-20261011-v1.tar.gz'
    entries.append((capsule,'source/'+capsule.name))
    hashes={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,name in entries}
    manifest=archive.with_name(archive.name+'.manifest.json')
    with manifest.open('x') as stream:json.dump(hashes,stream,indent=2)
    with tarfile.open(archive,'x:gz') as tar:
        for p,name in entries:
            assert not p.is_symlink()
            tar.add(p,arcname=name,recursive=False)
        tar.add(manifest,arcname='result-manifest.json',recursive=False)
    info=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries))
    archive.with_name(archive.name+'.info.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info))


if __name__=='__main__':
    main()
