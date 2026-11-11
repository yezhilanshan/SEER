"""Inventory duplicate delivery archives without deleting or extracting anything."""
import hashlib
import json
from pathlib import Path
import tarfile
import time

root=Path('D:/lab/SEER').resolve()
output=root/'outputs/workspace-cleanup-20261018-v1'
output.mkdir(exist_ok=False)
files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
index={}
for p in files:
    index.setdefault((p.name,p.stat().st_size),[]).append(p)
hashes={}
def digest_file(p):
    key=str(p)
    if key not in hashes:
        h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(1024*1024): h.update(data)
        hashes[key]=h.hexdigest()
    return hashes[key]

archives=[p for p in files if p.name.endswith('.tar.gz') and p.stat().st_size>10*1024**2]
reports=[]
for archive in sorted(archives,key=lambda p:p.stat().st_size,reverse=True):
    report={'archive':str(archive.relative_to(root)),'bytes':archive.stat().st_size,
            'archive_sha256':digest_file(archive),'members':[],'unmatched':[],'special':[]}
    try:
        with tarfile.open(archive,'r|gz') as tf:
            for row in tf:
                if row.isdir(): continue
                if not row.isfile(): report['special'].append(row.name); continue
                candidates=index.get((Path(row.name).name,row.size),[])
                candidates=[p for p in candidates if p!=archive and not p.name.endswith(('.tar.gz','.zip'))]
                suffix=row.name.replace('\\','/').removeprefix('./')
                candidates.sort(key=lambda p:(not p.as_posix().endswith('/'+suffix),len(str(p))))
                if not candidates:
                    report['unmatched'].append({'name':row.name,'bytes':row.size,'reason':'no physical candidate'})
                    continue
                h=hashlib.sha256()
                with tf.extractfile(row) as f:
                    while data:=f.read(1024*1024): h.update(data)
                digest=h.hexdigest(); matches=[p for p in candidates if digest_file(p)==digest]
                if matches:
                    report['members'].append({'name':row.name,'bytes':row.size,'sha256':digest,
                        'retained_copy':str(matches[0].relative_to(root))})
                else: report['unmatched'].append({'name':row.name,'bytes':row.size,'reason':'content differs'})
        report['fully_duplicated']=bool(report['members']) and not report['unmatched'] and not report['special']
    except Exception as exc:
        report['fully_duplicated']=False; report['error']=repr(exc)
    reports.append(report)
    (output/'archive-audit.json').write_text(json.dumps({'created_unix':time.time(),'archives':reports},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('members','unmatched','special')},ensure_ascii=True),flush=True)

summary={'workspace':str(root),'initial_bytes':sum(p.stat().st_size for p in files),
         'initial_files':len(files),'archives_checked':len(reports),
         'duplicate_archives':[r['archive'] for r in reports if r['fully_duplicated']],
         'reclaimable_archive_bytes':sum(r['bytes'] for r in reports if r['fully_duplicated'])}
(output/'inventory.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True),flush=True)
