"""Frozen validation-only matched/mismatched radar diagnostic; never an optimization step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from continuous_mmwave.radar_text.data import collate_windows
from continuous_mmwave.radar_text.metrics import score_caption
from continuous_mmwave.radar_text.runner import datasets, move_batch, read_settings, restore_caption_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    settings = read_settings(args.config)
    sets, phrases = datasets(settings, ("validation",))
    dataset = sets["validation"]
    indices = np.linspace(0, len(dataset) - 1, min(32, len(dataset)), dtype=int).tolist()
    samples = [dataset[index] for index in indices]
    model, _ = restore_caption_model(args.checkpoint, "cuda")
    rows = []
    for i, sample in enumerate(samples):
        wrong = next((samples[(i + offset) % len(samples)] for offset in range(1, len(samples))
                      if samples[(i + offset) % len(samples)]["actions"] != sample["actions"]), None)
        if wrong is None:
            continue
        correct_batch = move_batch(collate_windows([sample]), "cuda")
        wrong_batch = move_batch(collate_windows([wrong]), "cuda")
        # Both losses supervise the SAME target. Only the radar observation changes.
        target = [sample["caption"]]
        matched = float(model(correct_batch["rd"], correct_batch["md"], correct_batch["lengths"], target).loss)
        mismatched = float(model(wrong_batch["rd"], wrong_batch["md"], wrong_batch["lengths"], target).loss)
        row = dict(sample_id=sample["sample_id"], wrong_sample_id=wrong["sample_id"],
                   target=sample["caption"], wrong_reference=wrong["caption"],
                   matched_loss=matched, mismatched_loss=mismatched, delta=mismatched - matched)
        if i < 8:
            correct_text = model.generate(correct_batch["rd"], correct_batch["md"],
                                          correct_batch["lengths"], max_new_tokens=96)[0]
            wrong_text = model.generate(wrong_batch["rd"], wrong_batch["md"],
                                        wrong_batch["lengths"], max_new_tokens=96)[0]
            row.update(matched_generation=correct_text, mismatched_generation=wrong_text,
                       matched_score=score_caption(correct_text, sample["actions"], phrases),
                       wrong_score_against_target=score_caption(wrong_text, sample["actions"], phrases),
                       wrong_score_against_own_radar=score_caption(wrong_text, wrong["actions"], phrases))
        rows.append(row)
    if not rows:
        raise ValueError("validation has no distinct action sequences for the mismatch control")
    report = dict(split="validation", optimizer_updates=0, samples=len(rows),
                  selection="32 evenly spaced validation windows; next distinct action sequence is the wrong radar",
                  matched_loss=float(np.mean([row["matched_loss"] for row in rows])),
                  mismatched_loss=float(np.mean([row["mismatched_loss"] for row in rows])),
                  mean_loss_delta=float(np.mean([row["delta"] for row in rows])),
                  matched_better_fraction=float(np.mean([row["delta"] > 0 for row in rows])),
                  note="A positive paired delta suggests conditioning; it does not prove correct fine-grained grounding.",
                  results=rows)
    generated = [row for row in rows if "matched_generation" in row]
    report["generation_changed_fraction"] = float(np.mean([
        row["matched_generation"] != row["mismatched_generation"] for row in generated
    ]))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
