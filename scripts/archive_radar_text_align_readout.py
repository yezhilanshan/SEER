"""Seal P0 results, keeping full radar and newly extracted features on server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for k in ('output','code','archive'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();root,code,archive=map(Path,(args.output,args.code,args.archive))
    status=json.loads((root/'status.json').read_text());assert status['state']=='complete'
    verify=json.loads((root/'verification.json').read_text())
    assert verify['encoder_state_bitwise_unchanged'] and verify['encoder_has_no_gradients'] and not verify['llm_loaded']
    prov=json.loads((root/'provenance.json').read_text())
    assert all(sha(k)==v for k,v in prov['original_input_sha256'].items())
    assert all(sha(code/k)==v for k,v in prov['source_sha256'].items())
    features=json.loads((root/'feature-cache-manifest.json').read_text())
    assert all(sha(root/k)==v['sha256'] for k,v in features.items())
    entries=[(p,'experiment/'+p.relative_to(root).as_posix()) for p in sorted(root.rglob('*')) if p.is_file() and 'features' not in p.relative_to(root).parts]
    entries += [(code/k,'provenance/'+k) for k in ('source-manifest.json','native-tests.log')]
    entries += [(root.with_name(root.name+'.logs')/'worker.log','launcher/worker.log')]
    capsule=code.parent/(code.name.replace('-code-','-')+'.tar.gz')
    entries += [(capsule,'source/'+capsule.name)]
    failed=root.with_name(root.name.replace('_v2','_v1'))
    if failed!=root and (failed/'status.json').exists():
        entries += [(failed/'status.json','provenance/failed-v1-status.json'),
                    (failed.with_name(failed.name+'.logs')/'worker.log','provenance/failed-v1-worker.log')]
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,name in entries}
    mp=archive.with_name(archive.name+'.manifest.json');mp.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(archive,'x:gz') as tar:
        for p,name in entries:
            assert not p.is_symlink();tar.add(p,arcname=name,recursive=False)
        tar.add(mp,arcname='result-manifest.json',recursive=False)
    info=dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries))
    archive.with_name(archive.name+'.info.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info))


if __name__=='__main__':main()
