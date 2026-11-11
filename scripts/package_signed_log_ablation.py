"""Seal a new execution capsule without changing historical training artifacts."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
out=root/'.audit-staging/signed-log-ablation-20261014-v1'
names={p.relative_to(root).as_posix() for p in (root/'src').rglob('*.py')}
names.update(['scripts/'+name for name in ['train_rd_resnet18_scratch.py','run_parrad_signed_log_ablation.py',
    'report_parrad_signed_log_ablation.py','plot_parrad_signed_log_ablation.py']])
names.update(['configs/rd-resnet18-scratch-expanded-20261011-v1.json','tests/test_signed_log_ablation.py',
    'tests/test_rd_resnet_scratch.py','tests/test_rd_only_xfi.py'])
manifest={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}
lines={}
for name in sorted(names):
    if 'signed_log' in name or name.endswith(('rd_scratch.py','rd_transfer.py','train_rd_resnet18_scratch.py')):
        lines[name]=[dict(line=i,definition=line.strip()) for i,line in enumerate((root/name).read_text(encoding='utf-8').splitlines(),1) if line.lstrip().startswith(('def ','class '))]
mp=out/'source-manifest.json';mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lp=out/'source-lines.json';lp.write_text(json.dumps(lines,indent=2),encoding='utf-8')
archive=out/'signed-log-source-20261014-v2.tar.gz'
assert not archive.exists(),'source capsules are immutable; version next export'
with tarfile.open(archive,'w:gz') as tar:
    for name in sorted(names):tar.add(root/name,arcname=name)
    tar.add(mp,arcname='source-manifest.json');tar.add(lp,arcname='source-lines.json')
print(json.dumps(dict(path=str(archive),files=len(manifest),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
