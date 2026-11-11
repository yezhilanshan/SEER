"""Run one isolated fixed-budget real-data pilot, preserving stage logs and failures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    root = Path(args.output).resolve()
    root.mkdir(exist_ok=False, parents=True)
    env = {**os.environ, "PYTHONPATH": str(code / "src"), "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1"}
    config = yaml.safe_load((code / "configs/radar-text-server-pilot.yaml").read_text())
    config["data"]["phrases"] = str(code / "configs/radar-text-actions-parrad.json")
    config["training"]["output_dir"] = str(root / "align")
    stage1 = {**config, "training": {**config["training"], "batch_size": 4,
                                   "gradient_accumulation": 1, "encoder_lr": 1e-3,
                                   "output_dir": str(root / "stage1")}}
    stage1_path, align_path = root / "stage1.yaml", root / "align.yaml"
    stage1_path.write_text(yaml.safe_dump(stage1))
    align_path.write_text(yaml.safe_dump(config))
    prefix = [sys.executable, "-u", "-m", "continuous_mmwave.radar_text"]
    steps = [
        ("stage1", prefix + ["train", "--stage", "stage1", "--config", str(stage1_path)]),
        ("align", prefix + ["train", "--stage", "align", "--config", str(align_path),
                            "--checkpoint", str(root / "stage1/best.pt")]),
    ]
    for split in ("validation", "test"):
        steps.append((f"evaluate-{split}", prefix + [
            "evaluate", "--config", str(align_path), "--checkpoint", str(root / "align/best.pt"),
            "--split", split, "--baseline-checkpoint", str(root / "stage1/best.pt"),
            "--output", str(root / f"{split}-evaluation.json"),
        ]))
    metadata = dict(started=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                    code=str(code), output=str(root), stage1_epochs=3, align_epochs=3,
                    tiny=False, llm=config["llm"]["name"], completed=[])
    def write_status(state, **extra):
        metadata.update(state=state, **extra)
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(root / "status.json")
        print(json.dumps(metadata), flush=True)
    write_status("starting")
    try:
        for name, command in steps:
            write_status("running", active_stage=name)
            with (root / f"{name}.log").open("xb") as log:
                process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                write_status("running", child_pid=process.pid)
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f"{name} failed with exit code {returncode}; inspect {root / (name + '.log')}")
            metadata["completed"].append(name)
        write_status("complete", finished=datetime.now().astimezone().isoformat())
    except BaseException as error:
        write_status("failed", error=str(error), finished=datetime.now().astimezone().isoformat())
        raise


if __name__ == "__main__":
    main()
