"""Blockwise relational histories; compact projections are reusable across folds."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import math

import numpy as np
import torch

from continuous_mmwave.parrad_stage3_sequence_data import SequenceEndpoint

SOURCE_SETS = (("F",), ("E",), ("z",), ("F", "E"), ("F", "z"))


@lru_cache(maxsize=2)
def _matrix(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0,2,(dimension,128),generator=generator).float()*2-1)/math.sqrt(128)


def _validate(current, reference, sources):
    if sources not in SOURCE_SETS or set(current) != set(sources) or set(reference) != set(sources):
        raise ValueError("invalid history sources")
    shape = None
    for source in sources:
        c, r = np.asarray(current[source]), np.asarray(reference[source])
        if c.ndim != 3 or c.shape != r.shape or c.shape[1] not in (1,3,8) or not c.shape[2]:
            raise ValueError("inconsistent paired history shapes")
        if not np.isfinite(c).all() or not np.isfinite(r).all():
            raise ValueError("history values must be finite")
        if shape is not None and c.shape[:2] != shape:
            raise ValueError("inconsistent history positions")
        shape = c.shape[:2]
    return shape


def iter_history_blocks(source: Path, endpoints: Sequence[SequenceEndpoint], sources: tuple[str,...], history: int, *, block_size: int=64):
    if sources not in SOURCE_SETS and sources != ("F","E","z"):
        raise ValueError("invalid sources")
    if history not in (1,3,8) or block_size <= 0:
        raise ValueError("invalid history/block size")
    grouped = defaultdict(list)
    for i, endpoint in enumerate(endpoints):
        if len(endpoint.history_rows) < history or endpoint.history_rows[-1] != endpoint.endpoint_row:
            raise ValueError("incomplete or noncausal endpoint history")
        grouped[endpoint.recording_key].append(i)
    for key, indices in grouped.items():
        if Path(key).name != key or "/" in key or "\\" in key:
            raise ValueError("unsafe recording key")
        with np.load(source / key / "layers.npz", allow_pickle=False) as archive:
            arrays = {side: {s: archive[f"{side}_{s}"] for s in sources} for side in ("current", "reference")}
        for start in range(0,len(indices),block_size):
            selected = np.asarray(indices[start:start+block_size])
            positions = np.asarray([endpoints[i].history_rows[-history:] for i in selected])
            if positions.min() < 0:
                raise ValueError("negative history row index")
            try:
                current, reference = [{s: arrays[side][s][positions] for s in sources} for side in ("current","reference")]
            except IndexError as exc:
                raise ValueError("history row outside tensor shard") from exc
            for s in sources:
                if not np.isfinite(current[s]).all() or not np.isfinite(reference[s]).all():
                    raise ValueError("history values must be finite")
                if not np.array_equal(reference[s], np.repeat(reference[s][:,:1],history,axis=1)):
                    raise ValueError("frozen reference tensor changed within history")
            yield selected, current, reference


def fit_source_scales(blocks) -> dict[str,float]:
    totals = defaultdict(float)
    counts = defaultdict(int)
    for _, current, reference in blocks:
        for s, c in current.items():
            c, r = np.asarray(c,np.float64), np.asarray(reference[s],np.float64)
            if c.ndim != 3 or c.shape != r.shape or not np.isfinite(c).all() or not np.isfinite(r).all():
                raise ValueError("scale fitting requires finite paired histories")
            totals[s] += float(np.square(c-r).sum())
            counts[s] += c.shape[0]*c.shape[1]
    if not counts or any(n == 0 for n in counts.values()):
        raise ValueError("no probe-training histories for source scales")
    return {s:max(math.sqrt(totals[s]/counts[s]),1e-6) for s in counts}


def pack_history_block(current: dict[str,np.ndarray], reference: dict[str,np.ndarray], *, sources: tuple[str,...], seed: int) -> dict:
    n, history = _validate(current,reference,sources)
    dimensions = [current[s].shape[2] for s in sources]
    total = sum(dimensions)
    matrix = _matrix(history*total,seed).reshape(history,total,128)
    pack = {"sources": sources, "history": history, "blocks": {}}
    offset = 0
    for s, d in zip(sources,dimensions):
        c, r = np.asarray(current[s],np.float64), np.asarray(reference[s],np.float64)
        diff = np.asarray(c-r,np.float32).reshape(n,history*d)
        projection = (torch.from_numpy(diff) @ matrix[:,offset:offset+d].reshape(history*d,128)).numpy()
        pack["blocks"][s] = {
            "projection": projection, "current_norm2": np.square(c).sum(axis=(1,2)),
            "reference_norm2": np.square(r).sum(axis=(1,2)), "dot": (c*r).sum(axis=(1,2)),
            "residual_norm2": np.square(c-r).sum(axis=(1,2)),
        }
        offset += d
    return pack


def features_from_pack(pack: dict, scales: dict[str,float]) -> np.ndarray:
    if set(scales) != set(pack["sources"]) or any(not math.isfinite(s) or s <= 0 for s in scales.values()):
        raise ValueError("source scales must be positive and finite")
    projection = None
    norm_c = norm_r = dot = residual = 0.
    for s in pack["sources"]:
        block, scale = pack["blocks"][s], scales[s]
        value = block["projection"]/scale
        projection = value if projection is None else projection+value
        norm_c += block["current_norm2"]/scale**2
        norm_r += block["reference_norm2"]/scale**2
        dot += block["dot"]/scale**2
        residual += block["residual_norm2"]/scale**2
    cosine = dot/np.maximum(np.sqrt(norm_c)*np.sqrt(norm_r),np.finfo(np.float64).tiny)
    result = np.column_stack((projection,1-np.clip(cosine,-1,1),np.sqrt(np.maximum(residual,0))))
    if result.shape[1] != 130 or not np.isfinite(result).all():
        raise ValueError("projected history must have130 finite columns")
    return result


def project_history_block(current: dict[str,np.ndarray], reference: dict[str,np.ndarray], *, sources: tuple[str,...], scales: dict[str,float], seed: int) -> np.ndarray:
    return features_from_pack(pack_history_block(current,reference,sources=sources,seed=seed),scales)
