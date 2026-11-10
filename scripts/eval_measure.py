"""批量量测：对目录中的图像跑完整流水线，输出 CSV 与汇总。

用法：
    py -3 scripts/eval_measure.py --images data/raw/xxx --calib calib.json --out outputs/eval.csv

说明：本脚本只做"跑通 + 汇总"，真正的指标对照（M1–M6）需要真实标注，
见 docs/algo/p0-spec.md。CSV 字段可直接用于后续报告。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

FIELDS = [
    "image",
    "verdict",
    "severity",
    "width_mm",
    "width_px",
    "area_px",
    "length_px",
    "n_components",
    "reflection",
    "total_ms",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="批量量测")
    parser.add_argument("--images", required=True, help="图像目录")
    parser.add_argument("--config", default=None)
    parser.add_argument("--calib", default=None, help="标定 JSON（Calibration.save 产物）")
    parser.add_argument("--out", default="outputs/eval_measure.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    calib = Calibration.load(args.calib) if args.calib else None
    inspector = GlueInspector(cfg, calib=calib)

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"目录中没有图像：{args.images}")
        return 1

    rows = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        result = inspector.run(image)
        rows.append(
            {
                "image": path.name,
                "verdict": result.verdict.code,
                "severity": result.verdict.severity,
                "width_mm": round(result.measure.width_mm, 4),
                "width_px": round(result.measure.width_px, 3),
                "area_px": round(result.measure.area_px, 1),
                "length_px": round(result.measure.length_px, 2),
                "n_components": result.measure.n_components,
                "reflection": int(result.reflection.has_reflection),
                "total_ms": result.timings_ms.get("total", 0.0),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"处理 {len(rows)} 张，结果 -> {out}")
    print("判定分布：", counts)
    if inspector.degradations:
        print("降级：", inspector.degradations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
