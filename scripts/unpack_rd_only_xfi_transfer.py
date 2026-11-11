"""Verify and safely unpack the immutable server result archive once."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p=argparse.ArgumentParser();p.add_argument('--results',required=True);args=p.parse_args();root=Path(args.results).resolve()
archive=root/'rd-only-xfi-results-20261011-v1.tar.gz';info=json.loads(archive.with_name(archive.name+'.info.json').read_text())
assert archive.stat().st_size==info['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==info['sha256']
assert not (root/'experiment').exists(),'do not overwrite a previously extracted experiment'
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        destination=(root/member.name).resolve()
        assert root in destination.parents and member.isfile() and not destination.exists()
    tar.extractall(root)
shutil.copyfile(root/'rd-only-xfi-cache-audit-20261011-v1.json',root/'cache-audit.json')
print(json.dumps(dict(archive_sha256=info['sha256'],files=info['files'],verified_and_unpacked=True)))
