"""Export metadata/audits only; all radar arrays remain on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

p=argparse.ArgumentParser()
for name in ('data','code','audit','native','log','capsule','archive'):p.add_argument('--'+name,required=True,type=Path)
p.add_argument('--supplement',action='append',default=[],type=Path)
args=p.parse_args();assert not args.archive.exists();assert json.loads(args.audit.read_text())['passed']
entries=[(f,'dataset/'+f.relative_to(args.data).as_posix()) for f in sorted(args.data.rglob('*')) if f.is_file() and f.suffix!='.npz']
entries.extend((f,name) for f,name in ((args.audit,'independent-audit.json'),(args.native,'native-tests.log'),(args.log,'conversion.log'),(args.capsule,'source/'+args.capsule.name),(args.code/'source-manifest.json','source/source-manifest.json')))
entries.extend((args.code/'scripts'/name,'audit-source/'+name) for name in ('audit_parrad_expanded.py','archive_parrad_expanded.py'))
entries.extend((f,'supplementary/'+f.name) for f in args.supplement)
manifest={name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f,name in entries}
mp=args.archive.with_name(args.archive.name+'.manifest.json');assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2))
with tarfile.open(args.archive,'x:gz') as t:
    for f,name in entries:assert not f.is_symlink();t.add(f,arcname=name,recursive=False)
    t.add(mp,arcname='export-manifest.json',recursive=False)
print(json.dumps(dict(files=len(entries),bytes=args.archive.stat().st_size,sha256=sha(args.archive))))
