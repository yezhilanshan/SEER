"""Finish nested-source checks, then write an explicit bounded deletion plan."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

root=Path('D:/lab/SEER').resolve(); out=root/'outputs/workspace-cleanup-20261018-v1'
audit=json.loads((out/'archive-audit.json').read_text())
files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
outer={str((root/r['archive']).resolve()) for r in audit['archives']}
index={}
for p in files:
    if str(p.resolve()) not in outer: index.setdefault((p.name,p.stat().st_size),[]).append(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(1024*1024): h.update(data)
    return h.hexdigest()
for report in audit['archives']:
    if report.get('error'): continue
    pending={r['name']:r for r in report['unmatched'] if r['name'].endswith(('.zip','.tar.gz')) and index.get((Path(r['name']).name,r['bytes']))}
    if pending:
        matched=[]
        with tarfile.open(root/report['archive'],'r|gz') as tf:
            for row in tf:
                if row.name not in pending: continue
                with tf.extractfile(row) as f: blob=f.read()
                digest=hashlib.sha256(blob).hexdigest()
                candidates=[p for p in index[(Path(row.name).name,row.size)] if sha(p)==digest]
                if candidates:
                    report['members'].append({'name':row.name,'bytes':row.size,'sha256':digest,'retained_copy':str(candidates[0].relative_to(root))})
                    matched.append(row.name)
                del pending[row.name]
                if not pending: break
        report['unmatched']=[r for r in report['unmatched'] if r['name'] not in matched]
        report['fully_duplicated']=bool(report['members']) and not report['unmatched'] and not report['special']
(out/'archive-audit-final.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

archives=[r for r in audit['archives'] if r['fully_duplicated']]
protected={str((root/m['retained_copy']).resolve()).casefold() for r in archives for m in r['members']}
targets=[{'path':r['archive'],'kind':'duplicate_archive','bytes':r['bytes'],'sha256':r['archive_sha256']} for r in archives]
partial='outputs/v261-joint-pilot-20261015-v4/evidence.tar.gz'
record=next(r for r in audit['archives'] if (root/r['archive']).resolve()==(root/partial).resolve())
text=(root/'docs/v261-joint-results-2026-10-16.md').read_text(encoding='utf-8')
assert record.get('error') and '为已停止下载的部分文件' in text
targets.append({'path':partial,'kind':'incomplete_download','bytes':record['bytes'],'sha256':record['archive_sha256'],
                'evidence':'docs/v261-joint-results-2026-10-16.md: stopped partial download; archive read failed; compact evidence is retained'})

candidates=[]
for p in root.rglob('*'):
    if not p.is_dir() or p.is_symlink(): continue
    name=p.name
    if name in ('__pycache__','.pytest_cache'):
        candidates.append((p,'generated_cache'))
    elif (re.match(r'^pytest.*(green|final|release)',name) or re.match(r'^radar-text-(acceptance-tests|final-tests|verification-final|verification-core)$',name)):
        if not re.search(r'(red|fail|error)',str(p.relative_to(root)),re.I):
            leaves=[f for f in p.rglob('*') if f.is_file()]
            # A passed-test root must contain only named pytest fixture folders
            # and optional pytest state; preserve anything resembling a real run.
            if leaves and all(f.relative_to(p).parts[0].startswith(('test_','.')) for f in leaves):
                candidates.append((p,'passed_test_fixtures'))
candidates.sort(key=lambda x:len(x[0].parts)); chosen=[]
for p,kind in candidates:
    if any(p.is_relative_to(q) for q,_ in chosen): continue
    if any(Path(q).is_relative_to(p) for q in protected): continue
    chosen.append((p,kind))
for p,kind in chosen:
    leaves=[f for f in p.rglob('*') if f.is_file()]
    targets.append({'path':str(p.relative_to(root)),'kind':kind,'bytes':sum(f.stat().st_size for f in leaves),
                    'files':[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size} for f in leaves]})
for target in targets:
    p=(root/target['path']).resolve()
    assert p!=root and p.is_relative_to(root)
    assert not any(Path(q).is_relative_to(p) for q in protected)

plan={'workspace':str(root),'targets':targets,'planned_bytes':sum(t['bytes'] for t in targets),
      'retained_archive_member_copies':len(protected),'kept_nonduplicate_archives':[r['archive'] for r in audit['archives'] if not r['fully_duplicated'] and (root/r['archive']).resolve()!=(root/partial).resolve()]}
(out/'deletion-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'target_count':len(targets),'planned_gib':plan['planned_bytes']/1024**3,
                  'by_kind':{kind:sum(t['bytes'] for t in targets if t['kind']==kind)/1024**3 for kind in sorted({t['kind'] for t in targets})},
                  'archives_to_delete':[t['path'] for t in targets if t['kind']=='duplicate_archive']},indent=2))
