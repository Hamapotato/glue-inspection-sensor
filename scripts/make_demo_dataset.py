"""生成演示用合成数据集（训练/验证），并写出标签文件。

与 `host/crates/cli gen` 的生成风格保持一致：1280×960 灰度、背景 18 + 噪声、
水平胶条（宽 5/10/15 mm @ 0.0833 mm/px）、断胶为两段、反光为亮斑。
产物目录（默认 outputs/demo_dataset，已被 .gitignore 忽略）：

  train/*.png  val/*.png  train_labels.json  val_labels.json  dataset.json

用法：
  .venv\\Scripts\\python.exe scripts\\make_demo_dataset.py
  .venv\\Scripts\\python.exe scripts\\make_demo_dataset.py --train 300 --val 60 --with-reflection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_image(
    kind: str, w: int, h: int, rng: np.random.Generator, mm_per_px: float
) -> tuple[np.ndarray, list[list[float]], list[list[float]]]:
    """返回 (灰度图, 胶条框列表, 反光框列表)。"""
    bg = 18
    img = np.full((h, w), bg, dtype=np.float64)
    strip_w = int(round(10.0 / mm_per_px))  # 标准 10 mm
    if kind == "narrow":
        strip_w = int(round(5.0 / mm_per_px))
    elif kind == "wide":
        strip_w = int(round(15.0 / mm_per_px))
    strip_w = max(8, strip_w)

    if kind == "no_roi":
        # 无胶条：点阵背景（有对比度、无长条目标），用于教会模型"没有胶条时不要输出框"
        img[:] = 60
        for yy in range(0, h, 16):
            for xx in range(0, w, 16):
                img[yy : yy + 6, xx : xx + 6] = 200
        img += rng.normal(0.0, 8.0, size=(h, w))
        return np.clip(img, 0, 255).astype(np.uint8), [], []

    y0 = h // 2 - strip_w // 2
    x0, x1 = 40, w - 40
    boxes: list[list[float]] = []
    if kind == "broken":
        mid = (x0 + x1) // 2
        gap = 30
        for a, b in [(x0, mid - gap), (mid + gap, x1)]:
            img[y0 : y0 + strip_w, a:b] = 225
            boxes.append([float(a), float(y0), float(b), float(y0 + strip_w)])
    else:
        img[y0 : y0 + strip_w, x0:x1] = 225
        boxes.append([float(x0), float(y0), float(x1), float(y0 + strip_w)])

    reflection_boxes: list[list[float]] = []
    if kind == "reflection":
        cy, cx = h // 2, (x0 + x1) // 2
        ry, rx = max(6, strip_w // 2), max(10, w // 18)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        reflection_boxes.append([float(cx - rx), float(cy - ry), float(cx + rx), float(cy + ry)])

    img += rng.normal(0.0, 8.0, size=(h, w))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, boxes, reflection_boxes


def build_split(
    out_dir: Path, split: str, kinds: list[str], count: int, w: int, h: int, mm_per_px: float, seed: int
) -> list[dict]:
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(count):
        kind = kinds[i % len(kinds)]
        img, boxes, refl = make_image(kind, w, h, rng, mm_per_px)
        name = f"{i:04d}_{kind}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"PNG 编码失败: {name}")
        buf.tofile(str(d / name))  # 避免 cv2.imwrite 在非 ASCII 路径下静默失败
        records.append(
            {"file": f"{split}/{name}", "kind": kind, "boxes": boxes, "reflection_boxes": refl}
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo_dataset"))
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--mm-per-px", type=float, default=0.0833)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-reflection", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    kinds = ["normal", "narrow", "wide", "broken", "no_roi"]
    if args.with_reflection:
        kinds.append("reflection")
    train = build_split(out, "train", kinds, args.train, args.width, args.height, args.mm_per_px, args.seed)
    val = build_split(out, "val", kinds, args.val, args.width, args.height, args.mm_per_px, args.seed + 1000)

    (out / "train_labels.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "val_labels.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "train": len(train),
        "val": len(val),
        "size": [args.width, args.height],
        "mm_per_px": args.mm_per_px,
        "kinds": kinds,
        "note": "合成演示数据，仅用于流程展示，不代表真实精度",
    }
    (out / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(train)} 训练 / {len(val)} 验证 图像 -> {out}")
    print(f"类别: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
