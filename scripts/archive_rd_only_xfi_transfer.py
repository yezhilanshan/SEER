"""Seal RD transfer outputs; retain large frame-feature caches on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('output','code','archive','capsule','native','smoke'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root,code,archive=map(Path,(a.output,a.code,a.archive))
    load=lambda p:json.loads(Path(p).read_text())
    assert load(root/'status.json')['state']=='complete'
    verify=load(root/'verification.json')
    assert all(verify[k] for k in ('backbones_bitwise_unchanged','BN_eval','no_backbone_gradients','selection_locked_before_test','paired_input_hashes'))
    assert not verify['llm_loaded'];prov=load(root/'provenance.json')
    assert all(sha(k)==v for k,v in prov['original_input_sha256'].items())
    assert all(sha(code/k)==v for k,v in prov['source_sha256'].items())
    for name,entry in load(root/'feature-cache-manifest.json').items():assert sha(root/f'features/{name}.npz')==entry['sha256']
    for name,entry in load(root/'training.json').items():
        for mode,chosen in entry['selected'].items():assert sha(root/f'readouts/{name}-{mode}.pt')==chosen['sha256']
    entries=[(p,'experiment/'+p.relative_to(root).as_posix()) for p in sorted(root.rglob('*')) if p.is_file() and 'features' not in p.relative_to(root).parts]
    entries.extend([(code/'source-manifest.json','provenance/source-manifest.json'),(Path(a.native),'provenance/native-tests.txt'),
        (Path(a.smoke),'provenance/gpu-smoke.txt'),(Path(a.capsule),'source/'+Path(a.capsule).name),
        (root.with_name(root.name+'.worker.log'),'launcher/worker.log')])
    entries.extend((code/'scripts'/name,'audit-source/'+name) for name in ('archive_rd_only_xfi_transfer.py','verify_rd_only_xfi_cache.py'))
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,name in entries}
    mp=archive.with_name(archive.name+'.manifest.json');mp.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(archive,'x:gz') as t:
        for path,name in entries:
            assert not path.is_symlink();t.add(path,arcname=name,recursive=False)
        t.add(mp,arcname='result-manifest.json',recursive=False)
    info=dict(files=len(entries),bytes=archive.stat().st_size,sha256=sha(archive));archive.with_name(archive.name+'.info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))

if __name__=='__main__':main()
