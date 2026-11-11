"""Frozen short-horizon readout audit; never fits or promotes production weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from continuous_mmwave.parrad_stage3_frozen_layers import SEQUENCE_SETTINGS
from continuous_mmwave.parrad_stage3_linear_probe import PROBE_SETTINGS, SUBJECTS, binary_metrics, fit_probe, subject_fold_masks
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256
from continuous_mmwave.parrad_stage3_sequence_features import SOURCE_SETS, features_from_pack, iter_history_blocks, pack_history_block

METRICS = ("auroc", "average_precision", "balanced_accuracy")


def experiment_grid() -> tuple[dict,...]:
    result = []
    for cohort, kind, settings in (("natural","fixed_age","natural_fixed"),
                                   ("natural","terminal","natural_terminal"),
                                   ("forced","fixed_age","forced_fixed")):
        for age, histories in SEQUENCE_SETTINGS[settings].items():
            for task in SEQUENCE_SETTINGS["tasks"]:
                result.append({"name": f"{cohort}-{kind}-{age}-{task}", "cohort": cohort,
                               "endpoint_kind": kind, "age": int(age), "histories": tuple(histories), "task": task})
    return tuple(result)


def paired_delta(left: dict, right: dict) -> dict:
    if left["endpoint_ids"] != right["endpoint_ids"] or left["targets"] != right["targets"]:
        raise ValueError("paired comparisons require identical endpoint IDs and labels")
    return {m: (right["metrics"].get(m)-left["metrics"].get(m)
                if left["metrics"].get(m) is not None and right["metrics"].get(m) is not None else None) for m in METRICS}


def stratified_metrics(rows, targets, scores) -> dict:
    result = {}
    for name, field in (("by_recording","recording_key"),("by_sensor","sensor_id"),
                        ("by_session","session_id"),("by_age","age_updates")):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[str(row[field])].append(i)
        result[name] = {key: binary_metrics([targets[i] for i in indices],[scores[i] for i in indices]) for key,indices in groups.items()}
    pairs = weighted = 0.
    for metric in result["by_recording"].values():
        support = metric["positives"]*(metric["count"]-metric["positives"])
        if metric["auroc"] is not None:
            pairs += support
            weighted += support*metric["auroc"]
    result["conditional_recording"] = {"pairs": int(pairs), "auroc": weighted/pairs if pairs else None}
    return result


def _prepare(source, endpoints, history):
    """One recording read per history; compact fixed projections precede all folds.

    Projection contributions are linear in each source's scale. Their geometry
    moments are sufficient for the same full-history cosine/L2, so folds never
    need repeated decompression of high-dimensional histories.
    """
    packs = {}
    seeds = PROBE_SETTINGS["projection_seeds"]
    for indices, current, reference in iter_history_blocks(source,endpoints,("F","E","z"),history):
        for sources in SOURCE_SETS:
            for seed in seeds:
                block = pack_history_block({s:current[s] for s in sources},{s:reference[s] for s in sources},sources=sources,seed=seed)
                key = (sources,seed)
                if key not in packs:
                    packs[key] = {"sources": sources,"history":history,"blocks":{
                        s:{field:np.empty((len(endpoints),*values.shape[1:]),dtype=values.dtype)
                           for field,values in block["blocks"][s].items()} for s in sources}}
                for s in sources:
                    for field,values in block["blocks"][s].items():
                        packs[key]["blocks"][s][field][indices] = values
    return packs


def _write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def _fit_record(output, identifier, descriptor, endpoints, targets, train, held, x, scales, *, fixed_scores=None):
    held_indices = np.flatnonzero(held)
    metadata = [endpoints[i].metadata for i in held_indices]
    ids = [f"{endpoints[i].recording_key}|{endpoints[i].candidate_id}|{endpoints[i].cohort}|{endpoints[i].metadata['current_update']}" for i in held_indices]
    labels = targets[held].astype(bool).tolist()
    fingerprint = hashlib.sha256(json.dumps([ids,labels]).encode()).hexdigest()
    record = {**descriptor,"fit_id":identifier,"endpoint_ids":ids,"targets":labels,"endpoint_fingerprint":fingerprint,
              "held_count":len(labels),"train_count":int(train.sum()),"held_positives":int(targets[held].sum()),
              "train_positives":int(targets[train].sum()),"source_scales":scales,
              "metrics":{m:None for m in METRICS},"converged":None,"parameter_count":0,
              "status":"undefined","reason":None}
    fitted = None
    if not len(labels):
        record["reason"] = "no_heldout_endpoints"
        return record
    if not scales and descriptor["source"] != "age":
        record["reason"] = "no_probe_training_source_scale"
        return record
    if fixed_scores is not None:
        scores = np.asarray(fixed_scores)[held]
        record.update(status="scored",reason=None)
    elif len(set(targets[train])) < 2:
        record["reason"] = "probe_training_lacks_both_classes"
        return record
    else:
        fitted = fit_probe(x[train],targets[train],x[held])
        scores = fitted["scores"]
        record.update(status="fitted",converged=fitted["converged"],parameter_count=fitted["parameter_count"],
                      iterations=fitted["iterations"],gradient_inf=fitted["gradient_inf"],objective=fitted["objective"],
                      coefficient_norm=fitted["coefficient_norm"],
                      loss_change=fitted["loss_change"] if math.isfinite(fitted["loss_change"]) else None,
                      train_metrics=binary_metrics(targets[train].tolist(),fitted["train_scores"].tolist()))
    record["metrics"] = binary_metrics(labels,scores.tolist())
    record["strata"] = stratified_metrics(metadata,labels,scores.tolist())
    artifacts = {"held_scores":scores,"held_targets":np.asarray(labels),"endpoint_ids":np.asarray(ids),
                 "held_subjects":np.asarray([r["subject_id"] for r in metadata]),
                 "training_subjects":np.asarray([s for s in SUBJECTS if s!=descriptor["held_subject"]]),
                 "source_names":np.asarray(list(scales)),"source_scales":np.asarray(list(scales.values()))}
    if fitted:
        artifacts.update({k:fitted[k] for k in ("coefficients","mean","std","class_weights","train_scores")})
        artifacts["train_targets"] = targets[train]
    path = output/"fits"/f"{identifier}.npz"
    np.savez_compressed(path,**artifacts)
    record["artifact"] = str(path.relative_to(output))
    record["artifact_sha256"] = sha256(path)
    return record


def _aggregate(records, *, pairing=False):
    grouped = defaultdict(list)
    fields = ("group","comparison") if pairing else ("group","source","history","protocol")
    for r in records:
        grouped[tuple(r[field] for field in fields)].append(r)
    output = []
    for identity,values in grouped.items():
        seed_metrics = []
        for seed in sorted({r["seed"] for r in values},key=str):
            selected = [r for r in values if r["seed"]==seed]
            metrics = {}
            for metric in METRICS:
                defined = [r["delta" if pairing else "metrics"].get(metric) for r in selected]
                defined = [v for v in defined if v is not None]
                metrics[metric] = {"subject_macro":statistics.mean(defined) if defined else None,
                                   "subjects_defined":len(defined),"subject_range":[min(defined),max(defined)] if defined else None}
            seed_metrics.append({"seed":seed,"metrics":metrics})
        summary = {**dict(zip(fields,identity)),"per_seed":seed_metrics}
        for metric in METRICS:
            numbers = [r["metrics"][metric]["subject_macro"] for r in seed_metrics]
            numbers = [n for n in numbers if n is not None]
            summary[metric] = {"seed_median":statistics.median(numbers) if numbers else None,
                               "seed_range":[min(numbers),max(numbers)] if numbers else None}
        if not pairing:
            summary["nonconverged_fits"] = sum(r["converged"] is False for r in values)
        output.append(summary)
    return output


def evaluate(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    checked = load_sequence_source(source)
    torch.set_num_threads(PROBE_SETTINGS["cpu_threads"])
    grid = experiment_grid()
    output.mkdir(parents=True,exist_ok=False)
    (output/"fits").mkdir()
    manifest = {"status":"evaluating","source_split":"train","test_evaluation_allowed":False,
                "source_dir":str(source),"source_manifest_sha256":checked["manifest_sha256"],
                "probe_settings":PROBE_SETTINGS,"sequence_settings":SEQUENCE_SETTINGS,"grid":grid,
                "completed_groups":[],"started_unix":time.time()}
    _write(output/"sequence-manifest.json",manifest)
    records,paired,counts = [],[],[]
    identifier = 0
    def emit(record, stream):
        compact = {k:v for k,v in record.items() if k not in ("endpoint_ids","targets")}
        stream.write(json.dumps(compact,ensure_ascii=False,allow_nan=False)+"\n")
        stream.flush()
        records.append(compact)
    try:
        with (output/"sequence-results.jsonl").open("x",encoding="utf-8") as stream:
            for group in grid:
                started = time.monotonic()
                endpoints=[]; excluded=Counter()
                for key,rows in checked["rows_by_record"].items():
                    chosen,reasons = select_endpoints(rows,checked["candidates_by_record"][key],cohort=group["cohort"],
                        endpoint_kind=group["endpoint_kind"],age=group["age"],history_lengths=group["histories"],task=group["task"])
                    endpoints.extend(chosen); excluded.update(reasons)
                rows = [e.metadata for e in endpoints]
                targets = np.asarray([r[group["task"]+"_target"] for r in rows],dtype=float)
                counts.append({"group":group["name"],"eligible":len(endpoints),"positives":int(targets.sum()),
                               "negatives":len(endpoints)-int(targets.sum()),"exclusions":dict(excluded),
                               "by_subject":{s:{"count":sum(r["subject_id"]==s for r in rows),
                                   "positives":sum(r["subject_id"]==s and r[group["task"]+"_target"] for r in rows)} for s in SUBJECTS}})
                folds = {s:subject_fold_masks(rows,s) for s in SUBJECTS}
                lookup={}
                for history in group["histories"]:
                    packs = _prepare(source,endpoints,history)
                    for sources in SOURCE_SETS:
                        source_name = "+".join(sources)
                        for seed in PROBE_SETTINGS["projection_seeds"]:
                            pack = packs.get((sources,seed))
                            for held_subject,(train,held) in folds.items():
                                scales = {s:max(math.sqrt(float(pack["blocks"][s]["residual_norm2"][train].mean())/history),1e-6)
                                          for s in sources} if pack is not None and train.any() else {}
                                x = features_from_pack(pack,scales) if scales else np.empty((len(endpoints),130))
                                descriptor={"group":group["name"],"source":source_name,"history":history,"protocol":"relational",
                                            "seed":seed,"held_subject":held_subject}
                                identifier+=1
                                record=_fit_record(output,f"{identifier:06d}",descriptor,endpoints,targets,train,held,x,scales)
                                if record["status"]=="fitted" and record["parameter_count"]!=131:
                                    raise RuntimeError("capacity mismatch")
                                emit(record,stream)
                                lookup[source_name,history,seed,held_subject]=record
                                if seed==PROBE_SETTINGS["projection_seeds"][0]:
                                    for protocol,fixed in (("geometry",None),("fixed_cosine",x[:,-2]),("fixed_l2",x[:,-1])):
                                        identifier+=1
                                        control=_fit_record(output,f"{identifier:06d}",{**descriptor,"protocol":protocol,"seed":None},
                                                            endpoints,targets,train,held,x[:,-2:],scales,fixed_scores=fixed)
                                        emit(control,stream)
                    del packs
                if group["endpoint_kind"]=="terminal":
                    age_features=np.asarray([r["age_seconds"] for r in rows]).reshape(-1,1)
                    for held_subject,(train,held) in folds.items():
                        identifier+=1
                        control=_fit_record(output,f"{identifier:06d}",{"group":group["name"],"source":"age", "history":0,
                              "protocol":"age_only","seed":None,"held_subject":held_subject},endpoints,targets,train,held,age_features,{})
                        emit(control,stream)
                for seed in PROBE_SETTINGS["projection_seeds"]:
                    for subject in SUBJECTS:
                        comparisons=[]
                        for s in ("F","E","z","F+E","F+z"):
                            comparisons.extend((f"{s}:history{h}-history1",(s,1),(s,h)) for h in group["histories"] if h!=1)
                        for h in group["histories"]:
                            comparisons.extend((f"{hybrid}-{constituent}:history{h}",(constituent,h),(hybrid,h))
                                               for hybrid,parts in (("F+E",("F","E")),("F+z",("F","z"))) for constituent in parts)
                        for name,left,right in comparisons:
                            a,b=lookup[(*left,seed,subject)],lookup[(*right,seed,subject)]
                            paired.append({"group":group["name"],"comparison":name,"seed":seed,"held_subject":subject,
                                           "endpoint_fingerprint":a["endpoint_fingerprint"],"delta":paired_delta(a,b),
                                           "both_converged":a["converged"] is True and b["converged"] is True})
                manifest["completed_groups"].append({"name":group["name"],"seconds":time.monotonic()-started,"eligible":len(endpoints)})
                _write(output/"endpoint-counts.json",counts)
                _write(output/"sequence-manifest.json",manifest)
                print(json.dumps(manifest["completed_groups"][-1]),flush=True)
        # Check extraction artifacts/code/checkpoints once more, not merely JSON settings.
        final = load_sequence_source(source)
        if final["manifest_sha256"]!=checked["manifest_sha256"]:
            raise RuntimeError("source manifest changed during probing")
        expected = sum(sum(len(SOURCE_SETS)*len(PROBE_SETTINGS["projection_seeds"])*len(SUBJECTS)
                           +len(SOURCE_SETS)*3*len(SUBJECTS) for _ in g["histories"])
                       +(len(SUBJECTS) if g["endpoint_kind"]=="terminal" else 0) for g in grid)
        if len(records)!=expected or len(manifest["completed_groups"])!=len(grid):
            raise RuntimeError("incomplete experiment grid")
        decision_histogram=Counter((c["terminal_decision"],c["terminal_age"]) for cs in checked["candidates_by_record"].values() for c in cs)
        summary={"status":"complete","source_split":"train","test_evaluation_allowed":False,
                 "source_manifest_sha256":checked["manifest_sha256"],"expected_records":expected,"record_count":len(records),
                 "natural_decisions":[{"decision":d,"age":a,"count":n} for (d,a),n in decision_histogram.items()],
                 "counts":counts,"aggregates":_aggregate(records),"paired":paired,"paired_aggregates":_aggregate(paired,pairing=True),
                 "nonconverged_fits":[r["fit_id"] for r in records if r["converged"] is False],
                 "undefined_records":[r["fit_id"] for r in records if r["status"]=="undefined"],
                 "interpretation":"Probe-only subject holdout; natural survivor and forced cohorts must not be mixed. No automatic promotion."}
        _write(output/"sequence-summary.json",summary)
        manifest.update(status="complete",completed_unix=time.time(),record_count=len(records),
                        summary_sha256=sha256(output/"sequence-summary.json"),results_sha256=sha256(output/"sequence-results.jsonl"))
        _write(output/"sequence-manifest.json",manifest)
        return summary
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        _write(output/"sequence-manifest.json",manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    summary=evaluate(args.source_dir,args.output_dir)
    print(json.dumps({"status":summary["status"],"records":summary["record_count"],"nonconverged":len(summary["nonconverged_fits"])}))


if __name__=="__main__":
    main()
