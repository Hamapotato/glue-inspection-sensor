"""生成旋转扫描矩形检测的四类缺陷总览图。

用法：
    py -3 scripts/make_scanline_overview.py

预览图读取 ``visiontrain-glue-binary-v5`` 合成数据，只用于核对扫描矩形、
绿色边缘点、红色中心线和判定结果；合成数据不等于真实相机验收结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402
from algo.segmentation import MaskSegmenter  # noqa: E402
from algo.visualize import draw_result  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATASET_DIR = Path("outputs/synthetic/visiontrain-glue-binary-v5")
OUTPUT_PATH = Path("outputs/preview/scanline_v5_overview.png")
MM_PER_PX = 0.25  # [假设] 合成数据尺度，仅用于链路验证

CASES = (
    ("syn_20261120_000004", "正常胶条", "OK"),
    ("syn_20261120_000003", "断胶", "BROKEN"),
    ("syn_20261120_000000", "胶条过窄", "TOO_NARROW"),
    ("syn_20261120_000005", "胶条过宽", "TOO_WIDE"),
    ("syn_20261120_000002", "宽窄不一致", "WIDTH_UNSTABLE"),
    ("syn_20261120_000001", "胶条弯曲", "BENT"),
)


def main() -> int:
    cfg = load_config()

    figure, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=150)
    for axis, (image_id, title, expected_code) in zip(axes.ravel(), CASES):
        image_path = DATASET_DIR / "images" / f"{image_id}.png"
        mask_path = DATASET_DIR / "masks" / "glue" / f"{image_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"缺少预览数据：{image_path} 或 {mask_path}")

        inspector = GlueInspector(
            cfg,
            segmenter=MaskSegmenter(mask, source="synthetic_model_mask"),
            calib=Calibration(mm_per_px=MM_PER_PX),
        )
        result = inspector.run(image)
        if result.verdict.code != expected_code:
            raise RuntimeError(
                f"{image_id} 期望 {expected_code}，实际 {result.verdict.code}"
            )
        overlay = draw_result(image, result, mask=mask)
        measure = result.measure
        if result.roi is None:
            raise RuntimeError(f"{image_id} 没有检测 RoI，无法生成局部预览")
        margin = 24
        height, width = overlay.shape[:2]
        x1 = max(0, int(result.roi.x1) - margin)
        y1 = max(0, int(result.roi.y1) - margin)
        x2 = min(width, int(result.roi.x2) + margin)
        y2 = min(height, int(result.roi.y2) + margin)
        crop = overlay[y1:y2, x1:x2]
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"{title} → {result.verdict.code}\n"
            f"宽 {measure.width_mm:.2f} mm    "
            f"宽度极差 {measure.max_width_mm - measure.min_width_mm:.2f} mm    "
            f"弯曲 {measure.curve_deviation_mm:.2f} mm",
            fontsize=11.5,
        )
        axis.axis("off")
        print(
            f"{image_id} {result.verdict.code} "
            f"width={measure.width_mm:.2f}mm "
            f"range={measure.max_width_mm - measure.min_width_mm:.2f}mm "
            f"curve={measure.curve_deviation_mm:.2f}mm "
            f"break_windows={measure.break_windows}"
        )

    figure.suptitle(
        "沿涂胶方向的旋转扫描矩形检测（合成数据，非真实相机验收）",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "图例：蓝色矩形 = 有效扫描窗；红色矩形 = 断胶窗；"
        "绿色圆点 = 左右边缘点；红色曲线 = 左右边缘点中点连线",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"预览图已写入：{OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
