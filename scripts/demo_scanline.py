"""运行旋转扫描矩形检测主线。

该脚本用合成掩膜模拟分割模型输出，用于验证：

    分割掩膜 -> 二值后处理 -> 旋转扫描矩形 -> 边缘点 -> 中心线 -> 判定

注意：这是链路冒烟，不代表真实相机或训练模型指标。
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
from algo.segmentation import MaskSegmenter  # noqa: E402
from algo.synthetic_dataset import SCENARIOS, generate_sample  # noqa: E402
from algo.visualize import draw_result  # noqa: E402

SCENARIO_TITLES = {
    "glue": "正常",
    "break": "断胶",
    "narrow": "过窄",
    "wide": "过宽",
    "width_unstable": "宽窄不一致",
    "curve": "弯曲",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="旋转扫描矩形检测主线冒烟")
    parser.add_argument("--out", default="outputs/scanline-demo")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mm-per-px",
        type=float,
        default=0.25,
        help="[假设] 合成数据像素尺度，仅用于链路验证",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    calib = Calibration(mm_per_px=args.mm_per_px)

    summary: list[dict[str, object]] = []
    for index, scenario in enumerate(
        name for name in SCENARIOS if name != "background"
    ):
        settings = cfg["synthetic_dataset"]
        settings["scenario_weights"] = {
            name: 1.0 if name == scenario else 0.0 for name in SCENARIOS
        }
        sample = generate_sample(settings, args.seed + index)
        inspector = GlueInspector(
            cfg,
            segmenter=MaskSegmenter(sample.glue_mask, source="synthetic_model_mask"),
            calib=calib,
        )
        result = inspector.run(sample.image)
        overlay = draw_result(sample.image, result, mask=sample.glue_mask)
        cv2.imwrite(str(output_dir / f"{scenario}.png"), overlay)
        record = {
            "scenario": scenario,
            "title": SCENARIO_TITLES[scenario],
            **result.as_dict(),
        }
        summary.append(record)
        print(
            f"{scenario}: {result.verdict.code} "
            f"width={result.measure.width_mm:.2f}mm "
            f"curve={result.measure.curve_deviation_mm:.2f}mm "
            f"break_windows={result.measure.break_windows}"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入 {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
