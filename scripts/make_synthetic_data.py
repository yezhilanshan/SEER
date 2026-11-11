"""Create two shape-correct events for smoke testing only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    """生成两个形状正确的事件与 manifest，仅用于验证接口/梯度/checkpoint。"""
    root = Path("data/synthetic")
    root.mkdir(parents=True, exist_ok=True)
    # 固定种子保证冒烟测试可复现。
    rng = np.random.default_rng(7)
    rows = []
    for index, caption in enumerate(("person sits down", "person stands up")):
        path = root / f"event-{index}.npz"
        # 4 帧正好对应一次 radar update，point_counts 递减用于验证变长点云填充。
        counts = np.array([6, 5, 4, 6], dtype=np.int64)
        np.savez_compressed(
            path,
            rd=rng.normal(size=(4, 2, 64, 64)).astype("float32"),
            points=rng.normal(size=(4, 6, 6)).astype("float32"),
            point_counts=counts,
            skeleton=rng.normal(size=(4, 5, 3)).astype("float32"),
        )
        rows.append({"path": path.name, "event_id": f"synthetic-{index}", "caption": caption})
    # 路径写成相对 manifest 目录的形式，验证 ManifestDataset 的相对路径解析。
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
