import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from romsl.audit import run_audit


def main():
    parser = argparse.ArgumentParser(description='ROMSL P0 MM-Fi file and geometry audit')
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs' / 'mvp.yaml')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, joints, geometry = [
        json.loads(file.read_text(encoding='utf-8'))
        for file in (args.config, args.config.with_name('joints.yaml'), args.config.with_name('extrinsics_e01.yaml'))
    ]
    audit = run_audit(args.data_root, args.output, config, joints, geometry)
    print(json.dumps({key: audit[key] for key in (
        'totals', 'file_audit', 'geometry_gate', 'doppler_gate', 'gates', 'training_blockers',
    )}, indent=2))
    print(args.output / 'reports' / 'audit' / 'data_audit.json')
    return int(audit['file_audit'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
