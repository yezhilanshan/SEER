"""Create leakage-safe PARrad manifests from one canonical manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from continuous_mmwave.data_parrad import assert_disjoint_splits


def _subjects(values: Sequence[str]) -> set[str]:
    result = {value.strip() for value in values if value.strip()}
    if not result:
        raise ValueError("every split requires at least one subject")
    return result


def _write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def split_manifest_by_subject(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> dict[str, Path]:
    """Partition every source row exactly once and preserve recording path resolution."""
    source = Path(source_manifest).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("source manifest must contain at least one recording")

    requested = {
        "train": _subjects(train_subjects),
        "validation": _subjects(validation_subjects),
        "test": _subjects(test_subjects),
    }
    split_names = tuple(requested)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            overlap = requested[first_name] & requested[second_name]
            if overlap:
                raise ValueError(f"subject overlap between {first_name} and {second_name}: {sorted(overlap)}")
    available = {str(row["subject_id"]) for row in rows}
    assigned = set().union(*requested.values())
    if available != assigned:
        raise ValueError(
            f"split subjects must exactly cover source subjects; missing={sorted(available - assigned)}, "
            f"unknown={sorted(assigned - available)}"
        )

    split_rows: dict[str, list[dict[str, object]]] = {name: [] for name in split_names}
    for source_row in rows:
        subject = str(source_row["subject_id"])
        split_name = next(name for name, subjects in requested.items() if subject in subjects)
        row = dict(source_row)
        recording = Path(str(row["path"]))
        if not recording.is_absolute():
            recording = (source.parent / recording).resolve()
        row["path"] = Path(os.path.relpath(recording, output)).as_posix()
        split_rows[split_name].append(row)

    assert_disjoint_splits(**split_rows)
    paths = {name: output / f"{name}.jsonl" for name in split_names}
    for name, path in paths.items():
        _write_manifest(path, split_rows[name])
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-subjects", required=True, nargs="+")
    parser.add_argument("--validation-subjects", required=True, nargs="+")
    parser.add_argument("--test-subjects", required=True, nargs="+")
    args = parser.parse_args(argv)
    paths = split_manifest_by_subject(
        args.source_manifest,
        args.output_dir,
        train_subjects=args.train_subjects,
        validation_subjects=args.validation_subjects,
        test_subjects=args.test_subjects,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
