#!/usr/bin/env bash
# CPU 端到端冒烟验证：合成数据 -> Stage 1（单步）-> Stage 2（单步）。
# --max-steps 1 仅验证前向/反向与 checkpoint 流程，不代表任何训练效果。
set -euo pipefail
PYTHONPATH=src python scripts/make_synthetic_data.py
PYTHONPATH=src python -m continuous_mmwave.train stage1 --config configs/tiny-stage1.yaml --max-steps 1
PYTHONPATH=src python -m continuous_mmwave.train stage2 --config configs/tiny-stage2.yaml --max-steps 1
echo "Smoke training completed"
