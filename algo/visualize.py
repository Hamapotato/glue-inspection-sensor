"""可视化工具（调试与报告配图）。

只负责画图，不做任何判定或计算；调试产物写入 ``outputs/``（已在 .gitignore 中忽略）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import FrameResult

_COLORS = {
    "OK": (0, 200, 0),
    "BROKEN": (0, 0, 255),
    "TOO_WIDE": (0, 0, 255),
    "TOO_NARROW": (0, 0, 255),
    "WARN_WIDE": (0, 165, 255),
    "WARN_NARROW": (0, 165, 255),
    "REFLECTION_SUSPECT": (0, 255, 255),
}


def draw_result(
    image: np.ndarray,
    result: FrameResult,
    mask: np.ndarray | None = None,
    skeleton: np.ndarray | None = None,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """把检测结果叠加到图像上，返回 BGR 图。

    mask / skeleton 位于裁剪后的坐标系，``offset`` 为裁剪左上角在原图中的坐标。
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    ox, oy = int(offset[0]), int(offset[1])

    if mask is not None:
        m = np.asarray(mask) > 0
        h, w = m.shape
        roi = canvas[oy:oy + h, ox:ox + w]
        overlay = roi.copy()
        overlay[m] = (0.4 * overlay[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        canvas[oy:oy + h, ox:ox + w] = overlay

    if skeleton is not None:
        ys, xs = np.nonzero(np.asarray(skeleton) > 0)
        canvas[ys + oy, xs + ox] = (0, 0, 255)

    if result.roi is not None:
        b = result.roi
        cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)

    for b in result.reflection.boxes:
        cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 2)

    color = _COLORS.get(result.verdict.code, (255, 255, 255))
    lines = [
        f"{result.verdict.code} ({result.verdict.severity})",
        f"width={result.measure.width_mm:.2f}mm  area={result.measure.area_px:.0f}px  n_roi={result.verdict.details.get('n_rois', 0)}",
        f"total={result.timings_ms.get('total', 0.0):.1f}ms",
    ]
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return canvas
