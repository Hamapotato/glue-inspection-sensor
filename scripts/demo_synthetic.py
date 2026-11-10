"""合成样本冒烟：把五类用例跑过整条流水线，输出 JSON 与叠加图。

用法：
    py -3 scripts/demo_synthetic.py --out outputs/demo

注意：合成数据的指标**不用于验收**（AGENTS.md R1 / p0-spec §6）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402
from algo.synth import generate_case  # noqa: E402
from algo.visualize import draw_result  # noqa: E402

KINDS = ["normal", "narrow", "wide", "broken", "reflection"]


def main() -> int:
    parser = argparse.ArgumentParser(description="合成样本冒烟测试")
    parser.add_argument("--out", default="outputs/demo", help="输出目录")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/default.yaml")
    parser.add_argument("--mm-per-px", type=float, default=0.25, help="[假设] 合成数据的像素尺度")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 冒烟时启用占位反光检测，否则 REFLECTION_SUSPECT 分支不会被触发
    cfg["reflect"]["type"] = "heuristic"

    calib = Calibration(mm_per_px=args.mm_per_px)
    inspector = GlueInspector(cfg, calib=calib)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for kind in KINDS:
        case = generate_case(kind, seed=args.seed)
        result = inspector.run(case.image)
        overlay = draw_result(case.image, result)
        cv2.imwrite(str(out_dir / f"{kind}.png"), overlay)
        record = {"kind": kind, **result.as_dict()}
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入 {out_dir}（{len(summary)} 个用例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
