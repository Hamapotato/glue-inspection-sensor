"""胶条几何量测（论文 4.4）。

实现要点
--------
- 面积：BFS 从置信点种子出发遍历连通域（论文 4.4.1，图 4-15~4-20）；
  同时提供外轮廓法 ``contour_area`` 作为对照（论文表 4-3、4-4）。
- 骨架：Zhang-Suen 细化（论文 4.4.2，文献 [53]）。
- 宽度：``width_px = area_px / length_px``（论文 4.4.3）。

注意：论文的"置信点"来自深度学习网络（图 4-20）。B1 未上线前，种子由分割结果的
连通域质心给出，属于复现假设 A5，见 docs/algo/p0-spec.md §5。
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .types import MeasureResult

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class Component:
    """一个连通域。"""

    label: int
    area: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2（右下开区间）
    centroid: tuple[float, float]  # x, y


def _as_binary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def connected_components(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 1,
) -> tuple[np.ndarray, list[Component]]:
    """BFS 标记连通域（论文 4.4.1 的广度优先搜索）。

    返回 (label_map, components)。label 从 1 开始，0 为背景。
    """
    binary = _as_binary(mask)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    offsets = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4
    comps: list[Component] = []
    ys, xs = np.nonzero(binary)

    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if labels[sy, sx] != 0:
            continue
        label = len(comps) + 1
        q: deque[tuple[int, int]] = deque([(sy, sx)])
        labels[sy, sx] = label
        area = 0
        min_x = max_x = sx
        min_y = max_y = sy
        sum_x = 0
        sum_y = 0
        while q:
            cy, cx = q.popleft()
            area += 1
            sum_x += cx
            sum_y += cy
            if cx < min_x:
                min_x = cx
            elif cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            elif cy > max_y:
                max_y = cy
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = label
                    q.append((ny, nx))
        if area >= min_area:
            comps.append(
                Component(
                    label=label,
                    area=area,
                    bbox=(min_x, min_y, max_x + 1, max_y + 1),
                    centroid=(sum_x / area, sum_y / area),
                )
            )
        else:
            labels[labels == label] = 0
    return labels, comps


def bfs_region(
    mask: np.ndarray,
    seeds: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """从种子点出发的 BFS 区域（论文 4.4.1 的置信点扩散）。

    seeds: (N,2) 的 (x, y) 整数坐标；返回同尺寸布尔掩膜。
    """
    binary = _as_binary(mask)
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    offsets = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4
    q: deque[tuple[int, int]] = deque()
    for sx, sy in np.asarray(seeds).reshape(-1, 2):
        x, y = int(round(float(sx))), int(round(float(sy)))
        if 0 <= x < w and 0 <= y < h and binary[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in offsets:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def contour_area(mask: np.ndarray) -> float:
    """外轮廓法面积（论文 4.4.1 的对照方法，表 4-3/4-4）。"""
    binary = _as_binary(mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(sum(cv2.contourArea(c) for c in contours))


def zhang_suen_thinning(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen 骨架细化（论文 4.4.2，文献 [53]）。"""
    img = _as_binary(mask).astype(np.uint8)
    if img.sum() == 0:
        return img.astype(bool)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            p2 = p[0:-2, 1:-1]
            p3 = p[0:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, 0:-2]
            p8 = p[1:-1, 0:-2]
            p9 = p[0:-2, 0:-2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            a = np.zeros_like(img)
            for i in range(8):
                a += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                cond = (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = cond & (img == 1)
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def skeleton_length(skeleton: np.ndarray) -> float:
    """骨架长度（像素）：取骨架图的最长路径（直径）。

    早期版本用"4 邻接记 1、8 邻接记 √2"的边计数，斜向台阶会使长度虚高（实测可高估
    30% 以上），进而使 ``width = area / length`` 系统性偏小；改为在 8 邻接图上做两次
    Dijkstra 求直径，可忽略短分支（毛刺），对弯曲胶条更稳。
    """
    s = _as_binary(skeleton)
    ys, xs = np.nonzero(s)
    if ys.size == 0:
        return 0.0
    points = set(zip(ys.tolist(), xs.tolist()))

    def neighbors(p: tuple[int, int]):
        y, x = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in points:
                    yield q, (np.sqrt(2.0) if dy and dx else 1.0)

    def farthest(start: tuple[int, int]) -> tuple[tuple[int, int], dict]:
        dist: dict[tuple[int, int], float] = {start: 0.0}
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        far = start
        best = 0.0
        while heap:
            d, p = heapq.heappop(heap)
            if d > dist.get(p, float("inf")) + 1e-12:
                continue
            if d > best:
                best = d
                far = p
            for q, w in neighbors(p):
                nd = d + w
                if nd < dist.get(q, float("inf")) - 1e-12:
                    dist[q] = nd
                    heapq.heappush(heap, (nd, q))
        return far, dist

    start = (ys[0], xs[0])
    end_a, _ = farthest(start)
    end_b, dist = farthest(end_a)
    return float(dist.get(end_b, 0.0))


def width_from_area_length(area_px: float, length_px: float) -> float:
    """平均宽度 = 面积 / 骨架长度（论文 4.4.3）。"""
    if length_px <= 0:
        return 0.0
    return float(area_px) / float(length_px)


def measure_mask(
    mask: np.ndarray,
    seeds: np.ndarray | None = None,
    connectivity: int = 8,
    min_area_px: int = 1,
    mm_per_px: float | None = None,
) -> MeasureResult:
    """对一张已分割的掩膜完成面积/长度/宽度量测。

    - seeds 为 None 时取最大连通域质心作为 BFS 种子；
    - 存在多个连通域时，面积取最大者，``n_components`` 记录总数（断胶判据用）。
    """
    binary = _as_binary(mask)
    _, comps = connected_components(binary, connectivity=connectivity, min_area=min_area_px)
    if not comps:
        return MeasureResult()

    comps.sort(key=lambda c: -c.area)
    main = comps[0]
    if seeds is None:
        seed_arr = np.array([[main.centroid[0], main.centroid[1]]], dtype=np.float64)
    else:
        seed_arr = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    region = bfs_region(binary, seed_arr, connectivity=connectivity)
    if region.sum() == 0:
        region = binary  # 兜底：种子未落在前景时退回整幅掩膜
    area_px = float(region.sum())
    skel = zhang_suen_thinning(region)
    length_px = skeleton_length(skel)
    width_px = width_from_area_length(area_px, length_px)
    x1, y1, x2, y2 = main.bbox
    from .types import Box  # 局部导入，避免循环

    return MeasureResult(
        area_px=area_px,
        length_px=length_px,
        width_px=width_px,
        width_mm=width_px * float(mm_per_px) if mm_per_px else 0.0,
        n_components=len(comps),
        bbox=Box(float(x1), float(y1), float(x2), float(y2)),
    )
