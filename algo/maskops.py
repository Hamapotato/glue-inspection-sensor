"""胶条分割掩膜的基础几何操作。

只提供主线需要的连通域统计与骨架细化，不包含旧路线的面积法平均宽度。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class Component:
    """一个二值图像连通域。"""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def as_binary(mask: np.ndarray) -> np.ndarray:
    """把单通道或三通道输入转换为二维布尔掩膜。"""

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"掩膜必须是二维图像，实际形状为 {array.shape}")
    return array > 0


def connected_components(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 1,
) -> tuple[np.ndarray, list[Component]]:
    """标记连通域，返回标签图和面积不小于 ``min_area`` 的连通域。"""

    binary = as_binary(mask)
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    offsets = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4
    components: list[Component] = []
    ys, xs = np.nonzero(binary)

    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if labels[seed_y, seed_x] != 0:
            continue
        label = len(components) + 1
        queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        labels[seed_y, seed_x] = label
        area = 0
        min_x = max_x = seed_x
        min_y = max_y = seed_y
        sum_x = 0
        sum_y = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for dy, dx in offsets:
                neighbor_y, neighbor_x = y + dy, x + dx
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and binary[neighbor_y, neighbor_x]
                    and labels[neighbor_y, neighbor_x] == 0
                ):
                    labels[neighbor_y, neighbor_x] = label
                    queue.append((neighbor_y, neighbor_x))

        if area >= min_area:
            components.append(
                Component(
                    label=label,
                    area=area,
                    bbox=(min_x, min_y, max_x + 1, max_y + 1),
                    centroid=(sum_x / area, sum_y / area),
                )
            )
        else:
            labels[labels == label] = 0

    return labels, components


def zhang_suen_thinning(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen 骨架细化，返回单像素骨架。"""

    image = as_binary(mask).astype(np.uint8)
    if image.sum() == 0:
        return image.astype(bool)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            neighborhood_sum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            sequence = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = np.zeros_like(image)
            for index in range(8):
                transitions += (
                    (sequence[index] == 0) & (sequence[index + 1] == 1)
                ).astype(np.uint8)
            if step == 0:
                condition = (
                    (neighborhood_sum >= 2)
                    & (neighborhood_sum <= 6)
                    & (transitions == 1)
                    & (p2 * p4 * p6 == 0)
                    & (p4 * p6 * p8 == 0)
                )
            else:
                condition = (
                    (neighborhood_sum >= 2)
                    & (neighborhood_sum <= 6)
                    & (transitions == 1)
                    & (p2 * p4 * p8 == 0)
                    & (p2 * p6 * p8 == 0)
                )
            remove = condition & (image == 1)
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break

    return image.astype(bool)
