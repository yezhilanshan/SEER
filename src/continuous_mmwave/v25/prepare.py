"""Exclusive dataset-v2 preparation; holdout labels and arrays stay sealed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from .checkpoint import file_sha
from .data import DATASET_V2, SENSORS, build_training_manifest, build_evidence_targets, fit_normalization


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(root, output):
    from ..parrad_convert import convert_recording, read_command_annotations
    import h5py
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = root / 'canonical/phase1'
    lm = original / 'label_map.json'
    shutil.copyfile(lm, output / 'label_map.json')
    known = {r['command'] for r in json.loads(lm.read_text())['classes']}
    if len(known) != 14:
        raise ValueError('original 14-class vocabulary required')
    old_rows = [json.loads(line) for line in (original / 'train.jsonl').read_text().splitlines() if line.strip()]
    for row in old_rows:
        row['path'] = str((original / row['path']).resolve())
        row.pop('split', None)
    rows, sources, deferred = [], [], []
    for role, subjects in DATASET_V2.items():
        for subject, sessions in subjects.items():
            for session in sessions:
                if subject in ('subject_02', 'subject_03', 'subject_04'):
                    matches = [r for r in old_rows if r['subject_id'] == subject and r['session_id'] == session]
                    if sorted(r['sensor_id'] for r in matches) != sorted(SENSORS):
                        raise ValueError(f'missing or duplicate old sensor pair: {subject}/{session}')
                    rows.extend(matches)
                    continue
                directory = root / 'raw' / subject / 'captured_data' / session
                h5, csv = directory / 'processed_data.h5', directory / 'timestamp_speech.csv'
                if not h5.is_file() or not csv.is_file():
                    raise ValueError(f'missing fixed recording {directory}')
                source = dict(subject_id=subject, session_id=session, role=role, files=[])
                for path in (h5, csv):
                    digest = file_sha(path)
                    expected = None
                    # Standard sha256 sidecars are provenance assertions from the download,
                    # not necessarily an independently authenticated publisher signature.
                    for sidecar in (Path(str(path) + '.sha256'), path.with_suffix(path.suffix + '.sha')):
                        if sidecar.is_file():
                            candidate = sidecar.read_text().strip().split()[0]
                            if len(candidate) == 64 and all(c in '0123456789abcdef' for c in candidate.lower()):
                                expected = candidate.lower()
                                if digest != expected:
                                    raise ValueError(f'download sidecar mismatch: {path}')
                    source['files'].append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest,
                                                download_sidecar_sha256=expected,
                                                independently_authenticated=False))
                with h5py.File(h5, 'r') as handle:
                    source['headers'] = {sensor: {name: dict(shape=list(handle[f'radar/{sensor}/{name}'].shape),
                        dtype=str(handle[f'radar/{sensor}/{name}'].dtype)) for name in ('mDoppler', 'rdn', 'cube')}
                        for sensor in SENSORS}
                if role.endswith('holdout'):
                    for sensor in SENSORS:
                        deferred.append(dict(subject_id=subject, session_id=session, sensor_id=sensor,
                            role=role, source_h5=str(h5), source_commands=str(csv),
                            environment_id='source-unknown', array_access_locked=True,
                            conversion_status='deferred-until-terminal-lock'))
                else:
                    annotations = read_command_annotations(csv)
                    unknown = {r.command for r in annotations} - known
                    if unknown:
                        raise ValueError(f'unknown commands at {directory}: {sorted(unknown)}')
                    destination = output / 'converted' / subject / session
                    destination.mkdir(parents=True)
                    shutil.copyfile(lm, destination / 'label_map.json')
                    manifest = convert_recording(h5, csv, destination, subject_id=subject, session_id=session,
                        environment_id='source-unknown', frame_period_s=.09, md_only=True)
                    for line in manifest.read_text().splitlines():
                        item = json.loads(line)
                        item['path'] = str((destination / item['path']).resolve())
                        item['recording_start_rule'] = 'first-command-as-original-converter'
                        rows.append(item)
                sources.append(source)
                print(json.dumps(dict(phase='data-preparation', subject=subject, session=session,
                    role=role, converted=not role.endswith('holdout'))), flush=True)
    result = build_training_manifest(rows)
    for role in ('session_holdout', 'subject_holdout'):
        result['splits'][role].extend(r for r in deferred if r['role'] == role)
    counts = {role: len(values) for role, values in result['splits'].items()}
    expected = dict(fit=18, dev_regression=6, dev_new=2, dev=8, session_holdout=8, subject_holdout=8)
    if counts != expected or result['duplicates']:
        raise ValueError(f'fixed view counts mismatch: {counts}')
    fit = []
    for row in result['splits']['fit']:
        with np.load(row['path'], allow_pickle=False) as recording:
            fit.append(dict(role='fit', sensor_id=row['sensor_id'], md=recording['md']))
    normalization = fit_normalization(fit)
    coverage = []
    for row in result['splits']['fit'] + result['splits']['dev']:
        with np.load(row['path'], allow_pickle=False) as recording:
            _, mask = build_evidence_targets(recording['md'], normalization=normalization[row['sensor_id']])
        proportions = mask.mean(axis=0)
        coverage.append(dict(subject_id=row['subject_id'], session_id=row['session_id'], sensor_id=row['sensor_id'],
                             role=row['role'], valid_coverage=proportions.tolist()))
        if (proportions >= .8).sum() < 2:
            raise ValueError(f'insufficient physical target coverage: {coverage[-1]}')
    write_json(output / 'normalization.json', normalization)
    for role, items in result['splits'].items():
        with (output / f'{role}.jsonl').open('x', encoding='utf-8') as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    evidence = dict(schema_version=25, dataset_revision=2, counts=counts, sources=sources,
        original_train_manifest_sha256=file_sha(original / 'train.jsonl'), label_map_sha256=file_sha(lm),
        normalization_sha256=file_sha(output / 'normalization.json'), coverage=coverage,
        holdout_array_access=False, official_validation_test_access=False,
        manifests={role: file_sha(output / f'{role}.jsonl') for role in result['splits']})
    write_json(output / 'dataset-provenance.json', evidence)
    print(json.dumps(dict(phase='data-ready', counts=counts, actual_optimizer_steps=0)), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        prepare(args.root, args.output)
    except BaseException:
        # Keep partial preparation and its failure; never silently resume or delete.
        output = Path(args.output)
        if output.is_dir() and not (output / 'preparation-failure.txt').exists():
            (output / 'preparation-failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
