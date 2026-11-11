"""Safe local export and immutable source capsule for RD transfer worker."""
import hashlib
import json
from pathlib import Path
import tarfile
import torch
from continuous_mmwave.radar_text.rd_transfer import load_xfi_state

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

root=Path(__file__).resolve().parents[1];staging=root/'.audit-staging'
raw=root/'backbone_models/mmwave_ResNet18.pt';expected='03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56'
assert sha(raw)==expected
converted=staging/'xfi-mmwave-tensors-20261011-v1.pt'
if not converted.exists():torch.save(dict(state_dict=load_xfi_state(raw),original_sha256=expected),converted)
payload=torch.load(converted,map_location='cpu',weights_only=True)
assert payload['original_sha256']==expected and all(torch.equal(v,payload['state_dict'][k]) for k,v in load_xfi_state(raw).items())
old=json.loads((staging/'radar-text-p1-20261011-v1-manifest.json').read_text())
names=set(old)|{'src/continuous_mmwave/radar_text/rd_transfer.py','tests/test_rd_only_xfi.py',
    'scripts/run_rd_only_xfi_transfer.py','scripts/package_rd_only_xfi.py',
    'docs/superpowers/plans/2026-10-11-rd-only-xfi-transfer.md','docs/xfi-mmwave-rd-transfer-assessment-2026-10-11.md'}
manifest={p:sha(root/p) for p in sorted(names)}
mp=staging/'rd-only-xfi-source-20261011-v1-manifest.json';mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
capsule=staging/'rd-only-xfi-source-20261011-v1.tar.gz'
assert not capsule.exists(),'source capsule is immutable; version next export'
with tarfile.open(capsule,'w:gz') as archive:
    for name in manifest:archive.add(root/name,arcname=name)
    archive.add(mp,arcname='source-manifest.json')
assert sha(raw)==expected
print(json.dumps(dict(source=str(capsule),source_sha256=sha(capsule),files=len(manifest),converted=str(converted),converted_sha256=sha(converted),original_unchanged=True)))
