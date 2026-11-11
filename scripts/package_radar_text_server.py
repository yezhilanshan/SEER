"""Create a source-only capsule with hashes; never include data, model weights or credentials."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    if not re.fullmatch(r"v[1-9][0-9]*", args.version):
        raise ValueError("version must have the form v1, v2, ...")
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/continuous_mmwave").rglob("*.py"))
    paths += [root / name for name in (
        "pyproject.toml", "configs/radar-text-actions-parrad.json", "configs/radar-text-server-pilot.yaml",
        "scripts/radar_text_remote_preflight.py", "scripts/run_radar_text_server_pilot.py",
        "scripts/make_synthetic_parrad_data.py",
        "tests/test_radar_text.py", "tests/test_parrad_data.py", "tests/test_parrad_convert.py", "tests/test_parrad_split.py",
    )]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = root / f".audit-staging/radar-text-server-20261010-{args.version}.tar.gz"
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix())
        payload = json.dumps(hashes, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo("source-manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size, members=len(hashes) + 1,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
