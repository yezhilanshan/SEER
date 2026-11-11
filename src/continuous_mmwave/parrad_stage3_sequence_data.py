"""Checked, train-only endpoint indices for continuous frozen Candidate traces."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from continuous_mmwave.parrad_stage3_frozen_layers import SEQUENCE_SETTINGS, TRACE_TOLERANCE, validate_record_cohort
from continuous_mmwave.parrad_stage3_linear_probe import PROBE_SETTINGS


@dataclass(frozen=True)
class SequenceEndpoint:
    recording_key: str
    candidate_id: str
    cohort: str
    endpoint_row: int
    history_rows: tuple[int, ...]
    metadata: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequence_source(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("source_split") != "train" or manifest.get("status") != "complete"
        or manifest.get("diagnostic_only") is not True or manifest.get("test_evaluation_allowed") is not False
        or manifest.get("audit_kind") != "frozen_short_horizon_sequential"
        or manifest.get("trace_profile") != "continuous" or manifest.get("ages") != list(range(1, 14))
        or manifest.get("sequence_settings") != SEQUENCE_SETTINGS or manifest.get("probe_settings") != PROBE_SETTINGS):
        raise ValueError("source must be a complete continuous train-only extraction with fixed settings")
    records = manifest.get("records", [])
    validate_record_cohort(records)
    completed = manifest.get("completed_records", [])
    if len(completed) != 24 or {r["key"] for r in completed} != {r["key"] for r in records}:
        raise ValueError("incomplete or duplicate recording artifacts")
    code = Path(__file__).parent
    expected_code = {
        "extractor_sha256": code / "parrad_stage3_frozen_layers.py",
        "event_former_sha256": code / "models/event_former.py",
        "probe_code_sha256": code / "parrad_stage3_linear_probe.py",
    }
    for name, file in expected_code.items():
        if manifest.get(name) != sha256(file):
            raise ValueError(f"changed source code: {name}")
    names = ("parrad_stage3_sequence_data.py", "parrad_stage3_sequence_features.py", "parrad_stage3_sequence_probe.py")
    if manifest.get("sequence_code_sha256") != {name: sha256(code / name) for name in names}:
        raise ValueError("changed sequential audit code")
    for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("stage2_checkpoint", "stage2_checkpoint_sha256"),
                              ("config", "config_sha256"), ("threshold_artifact", "threshold_artifact_sha256")):
        if not manifest.get(field) or sha256(Path(manifest[field])) != manifest.get(hash_field):
            raise ValueError(f"changed source artifact: {field}")
    if not manifest.get("cache_fingerprint"):
        raise ValueError("missing cache fingerprint")
    rows_by_record, candidates_by_record = {}, {}
    by_key = {r["key"]: r for r in records}
    for result in completed:
        key = result["key"]
        files = result.get("artifact_sha256", {})
        if set(files) != {"rows.json", "candidates.json", "layers.npz", "integrity.json"}:
            raise ValueError("missing artifact hashes")
        for name, digest in files.items():
            if sha256(source / key / name) != digest:
                raise ValueError(f"changed recording artifact: {key}/{name}")
        integrity = json.loads((source / key / "integrity.json").read_text())
        if integrity != result["integrity"] or any(not 0 <= integrity.get(name, float("inf")) <= TRACE_TOLERANCE
                                                  for name in ("max_event_error", "max_key_error")):
            raise ValueError("invalid trace/controller integrity")
        if integrity.get("natural_updates") != by_key[key]["updates"]:
            raise ValueError("incomplete recording replay")
        rows = json.loads((source / key / "rows.json").read_text(encoding="utf-8"))
        candidates = json.loads((source / key / "candidates.json").read_text(encoding="utf-8"))
        for row in rows:
            if row.get("source_split") != "train" or row.get("recording_key") != key or any(
                row.get(field) != by_key[key]["metadata"].get(field) for field in ("subject_id", "session_id", "sensor_id")
            ):
                raise ValueError("row split or recording metadata mismatch")
        # Validate all branches, including excluded rows, before feature access.
        select_endpoints(rows, candidates, cohort="natural", endpoint_kind="fixed_age", age=4, history_lengths=(1,3), task="update")
        rows_by_record[key], candidates_by_record[key] = rows, candidates
    return {"manifest": manifest, "rows_by_record": rows_by_record, "candidates_by_record": candidates_by_record,
            "manifest_sha256": sha256(source / "manifest.json")}


def select_endpoints(rows: Sequence[dict], candidates: Sequence[dict], *, cohort: str, endpoint_kind: str,
                     age: int | None, history_lengths: tuple[int, ...], task: str) -> tuple[list[SequenceEndpoint], dict[str, int]]:
    if (cohort not in ("natural", "forced") or endpoint_kind not in ("fixed_age", "terminal")
        or (endpoint_kind == "terminal" and cohort != "natural") or task not in ("update", "proposal")
        or type(age) is not int or not 1 <= age <= 13 or not history_lengths
        or any(k not in (1,3,8) for k in history_lengths) or len(set(history_lengths)) != len(history_lengths)):
        raise ValueError("invalid sequential endpoint specification")
    candidate_map = {c["candidate_id"]: c for c in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("duplicate Candidate metadata")
    grouped = defaultdict(dict)
    references = {}
    for index, row in enumerate(rows):
        identity = row["candidate_id"]
        a = row["age_updates"]
        if row.get("source_split") != "train" or row["cohort"] not in ("natural", "forced") or identity not in candidate_map:
            raise ValueError("unapproved Candidate row")
        if type(a) is not int or not 1 <= a <= 13 or row["current_update"] != row["proposal_update"] + a - 1:
            raise ValueError("inconsistent age/update index")
        if abs(row["age_seconds"] - (a-1)*.18) > 1e-6 or row["active_ref_update"] >= row["proposal_update"]:
            raise ValueError("inconsistent age/reference timing")
        signature = tuple(row[field] for field in ("recording_key", "proposal_update", "active_ref_update", "active_ref_state_id"))
        if identity in references and references[identity] != signature:
            raise ValueError("changed frozen reference within Candidate")
        references[identity] = signature
        c = candidate_map[identity]
        if any(c[field] != row[field] for field in ("proposal_update", "active_ref_update", "active_ref_state_id")):
            raise ValueError("row differs from frozen Candidate reference")
        branch = grouped[identity, row["cohort"]]
        if a in branch:
            raise ValueError("duplicate Candidate/cohort/age")
        if row["cohort"] == "natural" and c["terminal_age"] is not None and a > c["terminal_age"]:
            raise ValueError("natural trace continued after terminal decision")
        branch[a] = index
    counts = Counter()
    selected = []
    for identity, c in candidate_map.items():
        branch = grouped.get((identity, cohort), {})
        endpoint_age = age
        if endpoint_kind == "terminal":
            if c["terminal_decision"] not in ("confirm", "reject"):
                counts["recording_end"] += 1
                continue
            endpoint_age = c["terminal_age"]
            if endpoint_age is None or endpoint_age not in branch:
                raise ValueError("missing terminal Candidate trace")
            if rows[branch[endpoint_age]]["decision"] != c["terminal_decision"]:
                raise ValueError("terminal decision mismatch")
            if endpoint_age < age:
                counts["too_young"] += 1
                continue
        if endpoint_age not in branch:
            counts["missing_endpoint"] += 1
            continue
        k = max(history_lengths)
        if endpoint_age < k:
            counts["too_young"] += 1
            continue
        ages = range(endpoint_age-k+1, endpoint_age+1)
        if any(a not in branch for a in ages):
            counts["incomplete_history"] += 1
            continue
        row = rows[branch[endpoint_age]]
        target = row[task + "_target"]
        if target is None:
            counts[row.get(task + "_exclusion") or "excluded_target"] += 1
            continue
        if type(target) is not bool:
            raise ValueError("target must be bool or excluded")
        selected.append(SequenceEndpoint(row["recording_key"], identity, cohort, branch[endpoint_age],
                                        tuple(branch[a] for a in ages), {**row, "terminal_age": c["terminal_age"],
                                                                             "terminal_decision": c["terminal_decision"]}))
    counts["eligible"] = len(selected)
    return selected, dict(counts)


def verify_smoke_overlap(previous: Path, smoke: Path) -> dict:
    """Read-only gate before full replay; never load cache or validation/test."""
    import numpy as np
    old = json.loads((previous/"manifest.json").read_text())
    new = json.loads((smoke/"manifest.json").read_text())
    for manifest in (old,new):
        if manifest.get("status")!="complete" or manifest.get("source_split")!="train" or manifest.get("test_evaluation_allowed") is not False:
            raise ValueError("smoke/previous must be complete train-only extractions")
    validate_record_cohort(old["records"])
    if len(new["records"])!=1 or len(new["completed_records"])!=1 or new.get("trace_profile")!="continuous":
        raise ValueError("smoke must contain one continuous recording")
    key = new["records"][0]["key"]
    if key not in {r["key"] for r in old["records"]} or new["completed_records"][0]["key"]!=key:
        raise ValueError("smoke recording outside prior train cohort")
    for field in ("checkpoint_sha256","stage2_checkpoint_sha256","threshold_artifact_sha256","config_sha256","cache_fingerprint"):
        if new[field]!=old[field]:
            raise ValueError(f"smoke sources changed: {field}")
    result = new["completed_records"][0]
    old_result = next(r for r in old["completed_records"] if r["key"]==key)
    for root,entry in ((smoke,result),(previous,old_result)):
        for name,digest in entry["artifact_sha256"].items():
            if sha256(root/key/name)!=digest:
                raise ValueError("smoke/prior artifact hash changed")
    read = lambda root,name: json.loads((root/key/f"{name}.json").read_text())
    if read(smoke,"candidates")!=read(previous,"candidates"):
        raise ValueError("continuous replay changed natural Candidate decisions/references")
    integrity = read(smoke,"integrity")
    if integrity!=read(previous,"integrity") or any(integrity[n]>TRACE_TOLERANCE for n in ("max_event_error","max_key_error")):
        raise ValueError("continuous replay changed controller integrity")
    old_rows,rows = read(previous,"rows"),read(smoke,"rows")
    row_key = lambda row:(row["candidate_id"],row["cohort"],row["age_updates"])
    lookup = {row_key(r):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):
        raise ValueError("duplicate continuous smoke rows")
    matched=[]
    for row in old_rows:
        if row_key(row) not in lookup or rows[lookup[row_key(row)]]!=row:
            raise ValueError("continuous replay changed overlapping trace metadata")
        matched.append(lookup[row_key(row)])
    for c in read(smoke,"candidates"):
        if c["terminal_age"] is not None:
            terminal=(c["candidate_id"],"natural",c["terminal_age"])
            if terminal not in lookup or rows[lookup[terminal]]["decision"]!=c["terminal_decision"]:
                raise ValueError("missing pre-cleanup terminal smoke trace")
    error=0.
    with np.load(smoke/key/"layers.npz",allow_pickle=False) as current, np.load(previous/key/"layers.npz",allow_pickle=False) as prior:
        if set(current.files)!=set(prior.files):
            raise ValueError("trace array fields changed")
        for name in current.files:
            values= current[name]
            if len(values)!=len(rows) or not np.isfinite(values).all():
                raise ValueError("smoke arrays must be complete and finite")
            older=prior[name]
            if values.shape[1:]!=older.shape[1:] or len(older)!=len(old_rows):
                raise ValueError("trace tensor shapes changed")
            if len(older):
                delta=float(np.max(np.abs(values[matched]-older)))
                error=max(error,delta)
                if delta!=0:
                    raise ValueError(f"continuous/sparse overlap differs: {name}, max_error={delta}")
    return {"status":"accepted","recording_key":key,"rows":len(rows),"overlap_rows":len(matched),
            "max_absolute_error":error,"controller_integrity":integrity}
