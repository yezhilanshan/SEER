"""Seal current review/report artifacts without modifying historical measurements."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--results',required=True);args=p.parse_args()
workspace=Path(__file__).resolve().parents[1];root=Path(args.results).resolve();destination=root/'final-manifest.json'
assert not destination.exists(),'do not rewrite a historical final seal'
for name in ('independent-audit.json','cache-audit.json'):assert json.loads((root/name).read_text())['passed']
old=json.loads((root/'result-manifest.json').read_text())
assert all(sha(root/k)==v['sha256'] and (root/k).stat().st_size==v['bytes'] for k,v in old.items())
files=['src/continuous_mmwave/radar_text/rd_transfer.py','tests/test_rd_only_xfi.py',
    'README.md','docs/offline-radar-to-text.md','docs/rd-only-xfi-transfer-2026-10-11.md',
    'docs/xfi-mmwave-rd-transfer-assessment-2026-10-11.md','docs/superpowers/plans/2026-10-11-rd-only-xfi-transfer.md']
files.extend('scripts/'+name for name in ('run_rd_only_xfi_transfer.py','package_rd_only_xfi.py','archive_rd_only_xfi_transfer.py',
    'verify_rd_only_xfi_transfer.py','verify_rd_only_xfi_cache.py','report_rd_only_xfi_transfer.py','unpack_rd_only_xfi_transfer.py','seal_rd_only_xfi_transfer.py'))
original=workspace/'backbone_models/mmwave_ResNet18.pt';expected='03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56'
assert sha(original)==expected
data=dict(sealed_at=datetime.now().astimezone().isoformat(),original_xfi_sha256=expected,
    artifacts={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()},
    workspace={name:sha(workspace/name) for name in files},historical_measurement_manifest_unchanged=True,
    scope='frozen RD frame backbone transfer only; no real Qwen training/evaluation; one seed, previously observed subject06')
destination.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(artifact_files=len(data['artifacts']),workspace_files=len(files),manifest_sha256=sha(destination),original_unchanged=True)))
