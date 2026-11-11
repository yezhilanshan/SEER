"""Validate existing server data and a real pretrained Qwen without optimizer updates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from continuous_mmwave.radar_text.data import collate_windows
from continuous_mmwave.radar_text.runner import build_model, datasets, read_settings, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    settings = read_settings(args.config)
    seed_everything(17)
    sets, phrases = datasets(settings, ("train", "validation"))
    report = dict(optimizer_updates=0, tiny=False, splits={}, gpu=torch.cuda.get_device_name(0),
                  compute_capability=torch.cuda.get_device_capability(0), precision="float32")
    expected = dict(train={"subject_02", "subject_03", "subject_04"},
                    validation={"subject_05"}, test={"subject_06"})
    report["split_subjects"] = {}
    for split, subjects in expected.items():
        rows = [json.loads(line) for line in Path(settings["data"][f"{split}_manifest"]).read_text().splitlines()
                if line.strip()]
        actual = {row["subject_id"] for row in rows}
        assert actual == subjects, f"{split} differs from the fixed pilot split: {actual}"
        report["split_subjects"][split] = sorted(actual)
    for split, dataset in sets.items():
        counts = Counter(len(window[4]) for window in dataset.windows)
        report["splits"][split] = dict(windows=len(dataset), action_counts=dict(counts),
                                      skipped=dataset.skipped,
                                      subjects=sorted({r["subject_id"] for r in dataset.recordings.rows}))
    label_map_path = Path(settings["data"]["train_manifest"]).parent / "label_map.json"
    label_map = json.loads(label_map_path.read_text())
    assert {int(c["id"]) for c in label_map["classes"]} == set(phrases), "phrase ID mismatch"
    report["label_map"] = label_map
    report["phrases"] = phrases
    model_path = Path(settings["llm"]["name"])
    report["source_sha256"] = {}
    for path in [label_map_path, model_path / "config.json", model_path / "model.safetensors"] + [
        Path(settings["data"][f"{split}_manifest"]) for split in ("train", "validation", "test")
    ]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        report["source_sha256"][str(path)] = digest.hexdigest()
    sample = sets["train"][0]
    batch = collate_windows([sample])
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    start = time.monotonic()
    model = build_model(settings, "align", len(phrases)).cuda()
    report["llm_hidden"] = model.llm.config.hidden_size
    report["llm_layers"] = model.llm.config.num_hidden_layers
    report["llm_trainable"] = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    assert report["llm_trainable"] == 0
    output = model(batch["rd"], batch["md"], batch["lengths"], batch["captions"])
    output.loss.backward()
    report["initial_loss"] = float(output.loss.detach())
    assert torch.isfinite(output.loss)
    report["gradient_norms"] = {}
    for name, module in dict(rd=model.encoder.rd_cnn, md=model.encoder.md_cnn,
                             bigru=model.encoder.gru, projector=model.projector).items():
        norm = sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None) ** 0.5
        report["gradient_norms"][name] = norm
        assert norm > 0, name
    assert all(p.grad is None for p in model.llm.parameters())
    del output
    model.zero_grad(set_to_none=True)
    gc.collect()
    model.eval()
    report["untrained_generation"] = model.generate(batch["rd"], batch["md"], batch["lengths"], max_new_tokens=64)[0]
    report["reference"] = sample["caption"]
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    report["model_check_seconds"] = time.monotonic() - start
    report["passed"] = True
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
