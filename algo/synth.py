"""合成样本生成器（冒烟测试与链路验证用）。

用途
----
论文的数据集未公开，本模块生成**可控的合成胶条图像**，用于：

- 验证去噪 → 分割 → 面积/骨架 → 宽度的链路是否按预期工作；
- 构造正常 / 过窄 / 过宽 / 断胶 / 反光 五类用例，跑通判定分支；
- 在没有真实数据时提供回归测试的固定输入。

**限制**：合成数据只用于冒烟与单元测试，不能用于验收或指标对照
（见 docs/algo/p0-spec.md §6 与 AGENTS.md R1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SyntheticCase:
    """一个合成用例。"""

    image: np.ndarray
    mask: np.ndarray
    kind: str
    width_px: float
    length_px: float
    box: tuple[float, float, float, float]
    reflection_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def _strip_polygon(
    x0: float,
    x1: float,
    cy: float,
    amp: float,
    width: float,
    n: int = 120,
) -> np.ndarray:
    """生成一条带轻微弧度的胶条多边形（左右两侧各一条曲线）。"""
    ts = np.linspace(0.0, 1.0, n)
    xs = x0 + ts * (x1 - x0)
    ys = cy + amp * np.sin(2.0 * np.pi * ts)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    nx, ny = -dy / norm, dx / norm
    half = width / 2.0
    left = np.stack([xs + nx * half, ys + ny * half], axis=1)
    right = np.stack([xs - nx * half, ys - ny * half], axis=1)
    return np.concatenate([left, right[::-1]], axis=0).round().astype(np.int32)


def generate_case(
    kind: str = "normal",
    size: tuple[int, int] = (320, 480),
    seed: int = 0,
    width_px: float = 40.0,
    bg_value: int = 40,
    fg_value: int = 200,
    noise_sigma: float = 6.0,
) -> SyntheticCase:
    """生成一个合成用例。

    kind: normal / narrow / wide / broken / reflection
    size: (h, w)
    """
    if kind not in {"normal", "narrow", "wide", "broken", "reflection"}:
        raise ValueError(f"未知用例类型 {kind}")

    rng = np.random.default_rng(seed)
    h, w = int(size[0]), int(size[1])
    image = np.full((h, w), bg_value, dtype=np.float64)
    image += rng.normal(0.0, noise_sigma, size=(h, w))

    cy = h / 2.0
    margin = 60
    x0, x1 = margin, w - margin
    amp = 12.0

    actual_width = width_px * {
        "narrow": 0.55,
        "wide": 1.45,
        "reflection": 0.55,  # 反光用例同时做成过窄，用于验证"反光复判"分支
    }.get(kind, 1.0)

    mask = np.zeros((h, w), dtype=np.uint8)
    if kind == "broken":
        gap = (x1 - x0) * 0.25
        mid = (x0 + x1) / 2.0
        parts = [(x0, mid - gap / 2), (mid + gap / 2, x1)]
    else:
        parts = [(x0, x1)]

    for a, b in parts:
        poly = _strip_polygon(a, b, cy, amp, actual_width)
        cv2.fillPoly(mask, [poly], 255)

    strip = mask > 0
    image[strip] = fg_value
    image += rng.normal(0.0, noise_sigma * 0.5, size=(h, w))
    image = np.clip(image, 0, 255).astype(np.uint8)

    reflection_boxes: list[tuple[float, float, float, float]] = []
    if kind == "reflection":
        cx = int((x0 + x1) / 2.0)
        ry, rx = int(actual_width * 0.5), int(w * 0.06)
        cv2.ellipse(image, (cx, int(cy)), (rx, ry), 0, 0, 360, 255, -1)
        reflection_boxes.append((cx - rx, cy - ry, cx + rx, cy + ry))

    ys, xs = np.nonzero(mask)
    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    if parts:
        length_px = float(sum(b - a for a, b in parts))
    else:  # pragma: no cover
        length_px = 0.0

    return SyntheticCase(
        image=image,
        mask=mask,
        kind=kind,
        width_px=actual_width,
        length_px=length_px,
        box=box,
        reflection_boxes=reflection_boxes,
    )
