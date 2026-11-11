"""Verify bounded cleanup and the surviving evidence without running training."""
import hashlib
import json
from pathlib import Path
import time

root=Path('D:/lab/SEER').resolve(); out=root/'outputs/workspace-cleanup-20261018-v1'
plan=json.loads((out/'deletion-plan.json').read_text())
execution=json.loads((out/'cleanup-execution.json').read_text(encoding='utf-8-sig'))
assert len(execution['Deleted'])==len(plan['targets'])
for target in plan['targets']: assert not (root/target['path']).exists(),target['path']
audit=json.loads((out/'archive-audit-final.json').read_text())
deleted={str((root/t['path']).resolve()) for t in plan['targets']}
copies={}
for row in audit['archives']:
    if str((root/row['archive']).resolve()) not in deleted or not row['fully_duplicated']: continue
    for member in row['members']:
        path=(root/member['retained_copy']).resolve()
        assert path.is_relative_to(root) and path.is_file() and path.stat().st_size==member['bytes']
        copies[path]=member['sha256']
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024): h.update(data)
    return h.hexdigest()
for path,expected in copies.items(): assert digest(path)==expected,path
experiment=root/'outputs/harrad-joint-prediction-20261017-v2'
snapshot=json.loads((experiment/'finished-v2/status-snapshot.json').read_text())
checked=0
for name,row in snapshot['jobs'].items():
    for filename,expected in row['files'].items():
        assert digest(experiment/'finished-v2/joint_prediction_20261017_v2/jobs'/name/filename)==expected
        checked+=1
assert (root/'docs/v261-semantic-diagnostic-results-2026-10-16.md').is_file()
assert (root/'outputs/v261-semantic-diagnostic-20261016-v1/curve-table.json').is_file()
assert (root/'outputs/v261-semantic-diagnostic-20261016-v1/matched-strata.json').is_file()
assert (root/'outputs/v261-joint-pilot-20261015-v4/compact-evidence').is_dir()
inventory=json.loads((out/'inventory.json').read_text())
remaining=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
summary={'passed':True,'verified_unix':time.time(),'deleted_targets':len(plan['targets']),
         'freed_bytes':int(execution['DeletedBytes']),'freed_gib':int(execution['DeletedBytes'])/1024**3,
         'initial_workspace_gib':inventory['initial_bytes']/1024**3,'remaining_workspace_gib':remaining/1024**3,
         'retained_duplicate_member_files_verified':len(copies),'harrad_job_files_verified':checked,
         'remaining_unique_archives':plan['kept_nonduplicate_archives']}
(out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
