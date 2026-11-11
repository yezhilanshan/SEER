"""Fixed-capacity probes for train-only frozen Stage P3 layer traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from continuous_mmwave.parrad_stage3_candidate_audit import score_binary


PROBE_SETTINGS = {
    "projection_seeds": [3407, 7211, 12893], "projection_dim": 128,
    "solver": "torch.optim.LBFGS", "line_search_fn": "strong_wolfe", "lambda": 1e-3,
    "max_iter": 200, "tolerance_grad": 1e-7, "tolerance_change": 1e-9,
    "decision_threshold": 0.5, "class_weight": "inverse frequency on probe-training subjects only",
    "standardization": "probe-training subjects only", "dtype": "float64", "cpu_threads": 1,
}
SUBJECTS = ("subject_02", "subject_03", "subject_04")
_PROJECTIONS: dict[tuple[int, int], torch.Tensor] = {}


def _project(vectors: np.ndarray, seed: int) -> np.ndarray:
    dimension = vectors.shape[1]
    key = (dimension, seed)
    if key not in _PROJECTIONS:
        generator = torch.Generator().manual_seed(seed)
        _PROJECTIONS[key] = (torch.randint(0, 2, (dimension, 128), generator=generator).float() * 2 - 1) / math.sqrt(128)
    return (torch.from_numpy(np.asarray(vectors, dtype=np.float32)) @ _PROJECTIONS[key]).numpy()


def geometry(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    current, reference = np.asarray(current, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape or current.ndim != 2 or not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("paired features must have equal finite matrix shapes")
    norm_c = np.linalg.norm(current, axis=1)
    norm_r = np.linalg.norm(reference, axis=1)
    denom = norm_c * norm_r
    cosine = np.sum(current * reference, axis=1) / np.maximum(denom, np.finfo(np.float64).tiny)
    cosine = np.clip(cosine, -1, 1)
    return np.column_stack((1 - cosine, np.linalg.norm(current - reference, axis=1)))


def relational_features(current: np.ndarray, reference: np.ndarray, seed: int) -> np.ndarray:
    distances = geometry(current, reference)
    return np.column_stack((_project(np.asarray(current) - np.asarray(reference), seed), distances))


def subject_fold_masks(rows: Sequence[dict[str, Any]], held_subject: str) -> tuple[np.ndarray, np.ndarray]:
    if held_subject not in SUBJECTS or any(row.get("source_split") != "train" or row.get("subject_id") not in SUBJECTS for row in rows):
        raise ValueError("probe folds accept the three approved train subjects only")
    held = np.array([row["subject_id"] == held_subject for row in rows], dtype=bool)
    return ~held, held


def binary_metrics(targets: Sequence[bool], scores: Sequence[float]) -> dict[str, Any]:
    result = score_binary([bool(value) for value in targets], [float(value) for value in scores])
    labels = np.asarray(targets, dtype=bool)
    predicted = np.asarray(scores) >= PROBE_SETTINGS["decision_threshold"]
    result["balanced_accuracy"] = float(((predicted[labels]).mean() + (~predicted[~labels]).mean()) / 2) if labels.any() and (~labels).any() else None
    result["decision_threshold"] = PROBE_SETTINGS["decision_threshold"]
    return result


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, held_x: np.ndarray) -> dict[str, Any]:
    """Training transformations and optimizer never consume held-out labels."""
    train_x, train_y, held_x = np.asarray(train_x, np.float64), np.asarray(train_y, np.float64), np.asarray(held_x, np.float64)
    if train_x.ndim != 2 or held_x.ndim != 2 or held_x.shape[1] != train_x.shape[1] or len(train_y) != len(train_x):
        raise ValueError("invalid probe matrix/label shapes")
    if not len(train_y) or not np.isfinite(train_x).all() or not np.isfinite(held_x).all() or set(np.unique(train_y)) != {0., 1.}:
        raise ValueError("probe training requires finite features and both classes")
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std = np.maximum(std, 1e-6)
    x = torch.from_numpy((train_x - mean) / std)
    y = torch.from_numpy(train_y)
    held = torch.from_numpy((held_x - mean) / std)
    counts = np.bincount(train_y.astype(int), minlength=2)
    weights = len(train_y) / (2 * counts.astype(np.float64))
    sample_weights = torch.from_numpy(weights[train_y.astype(int)])
    coefficients = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients], max_iter=PROBE_SETTINGS["max_iter"],
        tolerance_grad=PROBE_SETTINGS["tolerance_grad"], tolerance_change=PROBE_SETTINGS["tolerance_change"],
        line_search_fn=PROBE_SETTINGS["line_search_fn"],
    )

    def objective():
        logits = x @ coefficients[:-1] + coefficients[-1]
        return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * sample_weights).mean() + 0.5 * PROBE_SETTINGS["lambda"] * coefficients[:-1].square().sum()

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure()
    gradient_inf = float(coefficients.grad.detach().abs().max())
    state = optimizer.state[coefficients]
    iterations = int(state.get("n_iter", 0))
    loss_change = abs(float(loss.detach()) - float(state.get("prev_loss", float("inf"))))
    converged = gradient_inf <= PROBE_SETTINGS["tolerance_grad"] or (iterations < PROBE_SETTINGS["max_iter"] and loss_change <= PROBE_SETTINGS["tolerance_change"])
    with torch.no_grad():
        scores = torch.sigmoid(held @ coefficients[:-1] + coefficients[-1]).numpy()
        train_scores = torch.sigmoid(x @ coefficients[:-1] + coefficients[-1]).numpy()
    if not np.isfinite(scores).all() or not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("nonfinite fitted probe")
    return {
        "scores": scores, "train_scores": train_scores, "mean": mean, "std": std,
        "class_weights": weights, "coefficients": coefficients.detach().numpy(),
        "parameter_count": len(coefficients), "coefficient_norm": float(coefficients[:-1].detach().norm()),
        "iterations": iterations, "gradient_inf": gradient_inf, "objective": float(loss.detach()),
        "loss_change": loss_change, "converged": converged,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source: Path, output: Path, *, raw_sensitivity: bool = False) -> dict[str, Any]:
    from continuous_mmwave.parrad_stage3_frozen_layers import LAYER_NAMES, validate_record_cohort

    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train" or manifest.get("status") != "complete" or manifest.get("test_evaluation_allowed") is not False or manifest.get("probe_settings") != PROBE_SETTINGS:
        raise ValueError("probe source must be a complete train-only extraction with fixed settings")
    if manifest.get("probe_code_sha256") != _sha(Path(__file__)):
        raise ValueError("probe code changed after extraction was declared")
    records = manifest["records"]
    validate_record_cohort(records)
    rows: list[dict[str, Any]] = []
    slices: list[tuple[Path, int, int]] = []
    candidates: list[dict[str, Any]] = []
    completed = {record["key"]: record for record in manifest["completed_records"]}
    for record in records:
        directory = source / record["key"]
        if record["key"] not in completed:
            raise ValueError("missing completed recording")
        for name, expected in completed[record["key"]]["artifact_sha256"].items():
            if _sha(directory / name) != expected:
                raise ValueError("source shard SHA mismatch")
        part = json.loads((directory / "rows.json").read_text(encoding="utf-8"))
        candidates.extend(json.loads((directory / "candidates.json").read_text(encoding="utf-8")))
        start = len(rows)
        rows.extend(part)
        slices.append((directory, start, len(rows)))
    subject_fold_masks(rows, SUBJECTS[0])
    identity = [(row["candidate_id"], row["cohort"], row["age_updates"]) for row in rows]
    if len(set(identity)) != len(identity):
        raise ValueError("duplicate Candidate/cohort/age rows")
    output.mkdir(parents=True, exist_ok=False)
    (output / "fits").mkdir()
    (output / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    torch.set_num_threads(PROBE_SETTINGS["cpu_threads"])
    results: list[dict[str, Any]] = []

    def fit_group(features: np.ndarray, layer: str, protocol: str, seed: int | None, cohort: str, task: str, age: int, feature_indices: np.ndarray | None = None):
        eligible = np.array([
            row["cohort"] == cohort and row["age_updates"] == age and row[f"{task}_target"] is not None
            and (cohort != "forced" or row["full_horizon"])
            for row in rows
        ])
        indices = np.flatnonzero(eligible)
        selected_rows = [rows[index] for index in indices]
        targets = np.array([row[f"{task}_target"] for row in selected_rows], dtype=bool)
        feature_positions = indices if feature_indices is None else np.searchsorted(feature_indices, indices)
        if feature_indices is not None and not np.array_equal(feature_indices[feature_positions], indices):
            raise ValueError("secondary feature row mapping mismatch")
        for subject in SUBJECTS:
            train, held = subject_fold_masks(selected_rows, subject)
            entry = {"layer": layer, "protocol": protocol, "seed": seed, "cohort": cohort, "task": task,
                     "age_updates": age, "held_subject": subject, "train_count": int(train.sum()), "held_count": int(held.sum())}
            if not train.any() or not held.any() or len(np.unique(targets[train])) < 2:
                entry.update({"status": "undefined_training_classes", "held_metrics": binary_metrics(targets[held].tolist(), [float(targets[train].mean()) if train.any() else 0.5] * int(held.sum()))})
                for name in ("auroc", "average_precision", "balanced_accuracy"):
                    entry["held_metrics"][name] = None
                results.append(entry)
                continue
            fitted = fit_probe(features[feature_positions[train]], targets[train].astype(float), features[feature_positions[held]])
            scores = fitted["scores"]
            entry.update({"status": "fitted", "held_metrics": binary_metrics(targets[held].tolist(), scores.tolist()),
                          "train_metrics": binary_metrics(targets[train].tolist(), fitted["train_scores"].tolist()),
                          **{name: fitted[name] for name in ("parameter_count", "coefficient_norm", "iterations", "gradient_inf", "objective", "loss_change", "converged")}})
            entry["strata"] = {}
            held_rows = [selected_rows[index] for index in np.flatnonzero(held)]
            for field in ("sensor_id", "session_id"):
                entry["strata"][field] = {
                    str(value): binary_metrics(targets[held][mask].tolist(), scores[mask].tolist())
                    for value in sorted({row.get(field, "unknown") for row in held_rows})
                    for mask in [np.array([row.get(field, "unknown") == value for row in held_rows])]
                }
            suffix = f"{protocol}-{layer}-{cohort}-{task}-{age}-{seed}-{subject}"
            np.savez_compressed(output / "fits" / f"{suffix}.npz", held_indices=indices[held], scores=scores,
                                coefficients=fitted["coefficients"], mean=fitted["mean"], std=fitted["std"], class_weights=fitted["class_weights"])
            entry["artifact"] = f"fits/{suffix}.npz"
            prevalence = float(targets[train].mean())
            entry["prevalence_age_baseline"] = binary_metrics(targets[held].tolist(), [prevalence] * int(held.sum()))
            results.append(entry)

    for layer in LAYER_NAMES:
        projected: dict[int, list[np.ndarray]] = {seed: [] for seed in PROBE_SETTINGS["projection_seeds"]}
        current_only: dict[int, list[np.ndarray]] = {seed: [] for seed in PROBE_SETTINGS["projection_seeds"]}
        distances: list[np.ndarray] = []
        raw_latest: list[np.ndarray] = []
        raw_indices: list[np.ndarray] = []
        for directory, start, stop in slices:
            if start == stop:
                continue
            with np.load(directory / "layers.npz", allow_pickle=False) as shard:
                current, reference = shard[f"current_{layer}"], shard[f"reference_{layer}"]
                if len(current) != stop - start or current.shape != reference.shape:
                    raise ValueError("layer shard row alignment mismatch")
                geom = geometry(current, reference)
                distances.append(geom)
                for seed in PROBE_SETTINGS["projection_seeds"]:
                    projected[seed].append(np.column_stack((_project(current - reference, seed), geom)))
                    current_only[seed].append(np.column_stack((_project(current, seed), np.zeros((len(current), 2)))))
                if raw_sensitivity:
                    latest = np.array([row["age_updates"] == 13 for row in rows[start:stop]])
                    raw_latest.append(np.column_stack((current[latest] - reference[latest], geom[latest])))
                    raw_indices.append(np.flatnonzero(latest) + start)
        geom_features = np.concatenate(distances)
        for seed in PROBE_SETTINGS["projection_seeds"]:
            features = np.concatenate(projected[seed])
            secondary = np.concatenate(current_only[seed])
            if features.shape != (len(rows), 130):
                raise ValueError("capacity-matched features must have 130 inputs")
            for cohort in ("natural", "forced"):
                for age in manifest["ages"]:
                    for task in ("proposal", "update"):
                        fit_group(features, layer, "projected_relational", seed, cohort, task, age)
                    fit_group(secondary, layer, "current_only_secondary", seed, cohort, "proposal", age)
        for cohort in ("natural", "forced"):
            for age in manifest["ages"]:
                for task in ("proposal", "update"):
                    fit_group(geom_features, layer, "geometry_only", None, cohort, task, age)
                    eligible = [index for index, row in enumerate(rows) if row["cohort"] == cohort and row["age_updates"] == age and row[f"{task}_target"] is not None and (cohort != "forced" or row["full_horizon"])]
                    for subject in SUBJECTS:
                        indices = [index for index in eligible if rows[index]["subject_id"] == subject]
                        targets = [rows[index][f"{task}_target"] for index in indices]
                        for column, name in enumerate(("cosine_fixed_direction", "l2_fixed_direction")):
                            results.append({"layer": layer, "protocol": name, "seed": None, "cohort": cohort, "task": task,
                                            "age_updates": age, "held_subject": subject, "held_count": len(indices),
                                            "held_metrics": score_binary(targets, geom_features[indices, column].tolist())})
        if raw_sensitivity:
            raw = np.concatenate(raw_latest)
            raw_row_indices = np.concatenate(raw_indices)
            for cohort in ("natural", "forced"):
                for task in ("proposal", "update"):
                    fit_group(raw, layer, "raw_relational_secondary_age13", None, cohort, task, 13, raw_row_indices)
        with (output / "probe-results.jsonl").open("w", encoding="utf-8") as handle:
            for entry in results:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps({"completed_layer": layer, "fits_and_metrics": len(results)}), flush=True)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in results:
        grouped[tuple(entry[name] for name in ("layer", "protocol", "cohort", "task", "age_updates", "seed"))].append(entry)
    macros = []
    for key, entries in grouped.items():
        values = [entry["held_metrics"]["auroc"] for entry in entries if entry["held_metrics"]["auroc"] is not None]
        aps = [entry["held_metrics"]["average_precision"] for entry in entries if entry["held_metrics"]["average_precision"] is not None]
        macros.append({**dict(zip(("layer", "protocol", "cohort", "task", "age_updates", "seed"), key)),
                       "defined_subjects": len(values), "subject_macro_auroc": statistics.fmean(values) if values else None,
                       "subject_auroc_range": [min(values), max(values)] if values else None,
                       "subject_macro_ap": statistics.fmean(aps) if aps else None})
    eligibility = []
    for cohort in ("natural", "forced"):
        for age in manifest["ages"]:
            selected = [row for row in rows if row["cohort"] == cohort and row["age_updates"] == age and (cohort != "forced" or row["full_horizon"])]
            for task in ("proposal", "update"):
                values = [row[f"{task}_target"] for row in selected if row[f"{task}_target"] is not None]
                eligibility.append({"cohort": cohort, "age_updates": age, "task": task, "available_rows": len(selected),
                                    "eligible": len(values), "positives": sum(values), "negatives": len(values) - sum(values),
                                    "missing_or_excluded": len(candidates) - len(values)})
    seed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in macros:
        if entry["seed"] is not None:
            seed_groups[tuple(entry[name] for name in ("layer", "protocol", "cohort", "task", "age_updates"))].append(entry)
    seed_results = []
    for key, entries in seed_groups.items():
        values = [entry["subject_macro_auroc"] for entry in entries if entry["defined_subjects"] == 3 and entry["subject_macro_auroc"] is not None]
        seed_results.append({**dict(zip(("layer", "protocol", "cohort", "task", "age_updates"), key)),
                             "defined_seeds": len(values), "median_subject_macro_auroc": statistics.median(values) if values else None,
                             "seed_auroc_range": [min(values), max(values)] if values else None})
    summary = {"source_split": "train", "diagnostic_only": True, "test_evaluation_allowed": False,
               "source_manifest_sha256": _sha(manifest_path), "probe_code_sha256": _sha(Path(__file__)),
               "probe_settings": PROBE_SETTINGS, "torch_version": torch.__version__, "numpy_version": np.__version__,
               "raw_sensitivity_age": 13 if raw_sensitivity else None, "row_count": len(rows), "candidate_count": len(candidates),
               "eligibility": eligibility, "macro_results": macros, "seed_results": seed_results,
               "unconverged_fits": sum(entry.get("converged") is False for entry in results), "status": "complete"}
    (output / "probe-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-sensitivity", action="store_true")
    args = parser.parse_args()
    evaluate(args.source_dir, args.output_dir, raw_sensitivity=args.raw_sensitivity)


if __name__ == "__main__":
    main()
