"""旋转扫描矩形结果可视化。"""

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
    "WIDTH_UNSTABLE": (0, 0, 255),
    "BENT": (0, 0, 255),
}


def draw_result(
    image: np.ndarray,
    result: FrameResult,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """绘制分割掩膜、扫描窗、边缘点和实际中心线。"""

    canvas = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image.copy()
    )

    if mask is not None:
        binary = np.asarray(mask) > 0
        overlay = canvas.copy()
        overlay[binary] = (
            0.35 * overlay[binary] + 0.65 * np.array([0, 255, 0])
        ).astype(np.uint8)
        canvas = overlay

    if result.roi is not None:
        bbox = result.roi
        cv2.rectangle(
            canvas,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )

    for window in result.measure.scan_windows:
        center = (int(round(window.cx)), int(round(window.cy)))
        rectangle = (
            center,
            (
                max(float(window.length_px), 2.0),
                max(float(window.window_width_px), 2.0),
            ),
            window.angle_deg,
        )
        corners = cv2.boxPoints(rectangle)
        color = (0, 0, 255) if window.break_detected else (255, 120, 0)
        cv2.polylines(
            canvas,
            [np.round(corners).astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        for edge in (window.left, window.right):
            if edge is None:
                continue
            cv2.circle(
                canvas,
                (int(round(edge[0])), int(round(edge[1]))),
                4,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )
        if window.midpoint is not None:
            cv2.circle(
                canvas,
                (
                    int(round(window.midpoint[0])),
                    int(round(window.midpoint[1])),
                ),
                2,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    for segment in result.measure.centerline_segments:
        if len(segment) < 2:
            continue
        points = np.asarray(
            [(int(round(x)), int(round(y))) for x, y in segment],
            dtype=np.int32,
        )
        cv2.polylines(
            canvas,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    color = _COLORS.get(result.verdict.code, (255, 255, 255))
    width_range = max(
        0.0,
        result.measure.max_width_mm - result.measure.min_width_mm,
    )
    lines = [
        f"{result.verdict.code} ({result.verdict.severity})",
        f"width={result.measure.width_mm:.2f}mm  range={width_range:.2f}mm",
        (
            f"curve={result.measure.curve_deviation_mm:.2f}mm  "
            f"break_windows={result.measure.break_windows}"
        ),
        (
            f"segment={result.segmentation.get('source', 'unknown')}  "
            f"total={result.timings_ms.get('total', 0.0):.1f}ms"
        ),
    ]
    for index, text in enumerate(lines):
        y = 22 + index * 22
        cv2.putText(
            canvas,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
