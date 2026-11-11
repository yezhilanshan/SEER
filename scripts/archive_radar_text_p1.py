"""Seal completed P1 outputs; native radar/H/Z/P caches stay on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for k in ('output','code','archive'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();root,code,archive=map(Path,(args.output,args.code,args.archive))
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    verify=json.loads((root/'verification.json').read_text())
    assert verify['frozen_encoder_projector_bitwise_unchanged'] and verify['no_encoder_projector_gradients']
    assert not verify['llm_loaded'] and verify['selection_locked_before_test'] and verify['paired_B3_input_hashes']
    prov=json.loads((root/'provenance.json').read_text())
    assert all(sha(k)==v for k,v in prov['original_input_sha256'].items())
    assert all(sha(code/k)==v for k,v in prov['source_sha256'].items())
    for entry in json.loads((root/'feature-cache-manifest.json').read_text()).values():
        assert sha(root/entry['path'])==entry['sha256']
    for name,entry in json.loads((root/'probe-training.json').read_text()).items():
        assert sha(root/'probes'/f'{name}.pt')==entry['checkpoint_sha256']
    entries=[(p,'experiment/'+p.relative_to(root).as_posix()) for p in sorted(root.rglob('*'))
             if p.is_file() and 'features' not in p.relative_to(root).parts]
    entries += [(code/k,'provenance/'+k) for k in ('source-manifest.json','native-tests.log')]
    entries += [(root.with_name(root.name+'.logs')/'worker.log','launcher/worker.log')]
    capsule=code.parent/(code.name.replace('-code-','-')+'.tar.gz')
    entries += [(capsule,'source/'+capsule.name)]
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,name in entries}
    mp=archive.with_name(archive.name+'.manifest.json');mp.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(archive,'x:gz') as tar:
        for p,name in entries:
            assert not p.is_symlink();tar.add(p,arcname=name,recursive=False)
        tar.add(mp,arcname='result-manifest.json',recursive=False)
    info=dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(entries))
    archive.with_name(archive.name+'.info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))


if __name__=='__main__':main()
