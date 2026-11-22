"""沿胶条方向生成旋转扫描矩形并测量左右边缘。

处理流程：

1. 对二值分割结果做骨架化，提取每条胶条的中心线；
2. 按整体涂胶方向排序各段中心线，断胶缺口通过相邻端点桥接；
3. 沿中心线等距采样，在每个位置生成与局部切线垂直的扫描矩形；
4. 从扫描中心向左右法向找胶条边缘，记录左右边缘点和二者中点；
5. 缺失任一边缘的连续扫描窗判为断胶，所有有效中点组成实际中心线。

该模块只处理像素坐标；毫米换算通过 ``mm_per_px`` 统一完成。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from .maskops import connected_components, zhang_suen_thinning
from .types import MeasureResult, ScanWindow


@dataclass
class ScanResult:
    """单帧扫描矩形与边缘点结果。"""

    windows: list[ScanWindow] = field(default_factory=list)
    centerline_segments: list[list[tuple[float, float]]] = field(default_factory=list)
    widths_px: list[float] = field(default_factory=list)
    width_px: float = 0.0
    width_mm: float = 0.0
    min_width_px: float = 0.0
    max_width_px: float = 0.0
    width_std_px: float = 0.0
    min_width_mm: float = 0.0
    max_width_mm: float = 0.0
    width_std_mm: float = 0.0
    curve_deviation_px: float = 0.0
    curve_deviation_mm: float = 0.0
    break_windows: int = 0
    missing_edge_windows: int = 0


def _as_binary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _principal_axis(points_xy: np.ndarray) -> np.ndarray:
    """返回胶条整体主方向，并统一指向 x 正方向。"""

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0:
        axis = -axis
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _farthest_node(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
    """在骨架邻接图上广度优先搜索最远节点并记录父节点。"""

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        farthest = node
        for neighbor in graph.get(node, []):
            if neighbor in parents:
                continue
            parents[neighbor] = node
            queue.append(neighbor)
    return farthest, parents


def _longest_skeleton_path(
    skeleton_points: set[tuple[int, int]],
) -> list[tuple[float, float]]:
    """提取一个连通骨架的最长路径，返回有序 (x, y) 点列。"""

    if not skeleton_points:
        return []
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in skeleton_points:
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in skeleton_points:
                    neighbors.append(neighbor)
        graph[(y, x)] = neighbors

    start = next(iter(skeleton_points))
    end_a, _ = _farthest_node(graph, start)
    end_b, parents = _farthest_node(graph, end_a)
    path_yx: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = end_b
    while cursor is not None:
        path_yx.append(cursor)
        cursor = parents[cursor]
    path_yx.reverse()
    return [(float(x), float(y)) for y, x in path_yx]


def _ordered_centerline(
    binary: np.ndarray,
    min_area_px: int,
    connectivity: int,
) -> list[tuple[float, float]]:
    """提取所有胶条段的中心线，并按整体涂胶方向串接。"""

    labels, components = connected_components(
        binary,
        connectivity=connectivity,
        min_area=min_area_px,
    )
    if not components:
        return []
    skeleton = zhang_suen_thinning(binary)
    ys, xs = np.nonzero(binary)
    overall_axis = _principal_axis(np.stack([xs, ys], axis=1).astype(np.float64))

    paths: list[list[tuple[float, float]]] = []
    for component in components:
        component_points = {
            (int(y), int(x))
            for y, x in np.argwhere(
                skeleton & (labels == component.label)
            )
        }
        path = _longest_skeleton_path(component_points)
        if len(path) < 2:
            continue
        start_projection = float(np.dot(path[0], overall_axis))
        end_projection = float(np.dot(path[-1], overall_axis))
        if start_projection > end_projection:
            path.reverse()
        paths.append(path)

    paths.sort(
        key=lambda path: float(
            np.median(np.asarray(path, dtype=np.float64) @ overall_axis)
        )
    )
    ordered: list[tuple[float, float]] = []
    for path in paths:
        if ordered and np.allclose(ordered[-1], path[0]):
            ordered.extend(path[1:])
        else:
            ordered.extend(path)
    return ordered


def _resample_polyline(
    points: list[tuple[float, float]],
    step_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按弧长重采样中心线，返回中心点和单位切线。"""

    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    polyline = np.asarray(points, dtype=np.float64)
    segment_vectors = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    keep = segment_lengths > 1e-9
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    starts = polyline[:-1][keep]
    vectors = segment_vectors[keep]
    lengths = segment_lengths[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    sample_distance = np.arange(0.0, total + 1e-9, step_px)
    if sample_distance[-1] < total:
        sample_distance = np.append(sample_distance, total)

    centers: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    for distance in sample_distance:
        segment_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment_index = min(segment_index, len(lengths) - 1)
        local = (distance - cumulative[segment_index]) / max(lengths[segment_index], 1e-12)
        centers.append(starts[segment_index] + vectors[segment_index] * local)
        tangents.append(vectors[segment_index] / max(lengths[segment_index], 1e-12))
    return np.asarray(centers), np.asarray(tangents)


def _find_edge(
    binary: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    max_ray_px: float,
    ray_step_px: float,
) -> tuple[float, float] | None:
    """从扫描中心沿一个法向寻找最外侧前景点。"""

    h, w = binary.shape
    last_inside: np.ndarray | None = None
    inside_run = False
    for distance in np.arange(0.0, max_ray_px + 1e-9, ray_step_px):
        point = origin + direction * distance
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if binary[y, x]:
            inside_run = True
            last_inside = point.copy()
        elif inside_run:
            break
    if last_inside is None:
        return None
    return float(last_inside[0]), float(last_inside[1])


def _smooth_points(
    points: list[tuple[float, float]],
    window: int,
) -> list[tuple[float, float]]:
    """用滑动平均降低骨架台阶噪声。"""

    if len(points) < 3 or window <= 1:
        return points
    window = min(int(window), len(points))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return points
    array = np.asarray(points, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(array, ((pad, pad), (0, 0)), mode="edge")
    smooth = np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(2)
        ],
        axis=1,
    )
    return [(float(x), float(y)) for x, y in smooth]


def _curve_deviation(points: list[tuple[float, float]]) -> float:
    """计算中心线相对首尾弦线的最大垂距。"""

    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    chord = array[-1] - array[0]
    norm = float(np.linalg.norm(chord))
    if norm <= 1e-12:
        return 0.0
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / norm
    return float(np.max(np.abs((array - array[0]) @ normal)))


def _mark_break_runs(
    windows: list[ScanWindow],
    min_missing_windows: int,
) -> int:
    """把连续缺失边缘的扫描窗标记为断胶。"""

    break_windows = 0
    index = 0
    while index < len(windows):
        if windows[index].valid:
            index += 1
            continue
        end = index
        while end < len(windows) and not windows[end].valid:
            end += 1
        if end - index >= min_missing_windows:
            for break_index in range(index, end):
                windows[break_index].break_detected = True
                break_windows += 1
        index = end
    return break_windows


def scan_glue_strip(
    mask: np.ndarray,
    cfg: Config,
    mm_per_px: float | None = None,
) -> ScanResult:
    """沿胶条方向生成扫描矩形、边缘点、中心线和缺陷统计。"""

    if not bool(cfg.get("measure.scan.enabled")):
        return ScanResult()
    binary = _as_binary(mask)
    if not binary.any():
        return ScanResult()

    step_px = float(cfg.get("measure.scan.step_px"))
    window_length_px = float(cfg.get("measure.scan.window_length_px"))
    max_ray_px = float(cfg.get("measure.scan.max_ray_px"))
    ray_step_px = float(cfg.get("measure.scan.ray_step_px"))
    min_area_px = int(cfg.get("measure.min_area_px"))
    connectivity = int(cfg.get("measure.mask_connectivity"))
    min_missing_windows = int(cfg.get("measure.scan.break_min_missing_windows"))
    smooth_window = int(cfg.get("measure.scan.centerline_smooth_window"))
    window_width_factor = float(cfg.get("measure.scan.window_width_factor"))

    centerline = _ordered_centerline(binary, min_area_px, connectivity)
    centers, tangents = _resample_polyline(centerline, step_px)
    if len(centers) >= 3:
        smooth_centers = _smooth_points(
            [(float(x), float(y)) for x, y in centers],
            smooth_window,
        )
        centers = np.asarray(smooth_centers, dtype=np.float64)
        derivatives = np.gradient(centers, axis=0)
        tangent_norm = np.linalg.norm(derivatives, axis=1, keepdims=True)
        tangent_norm[tangent_norm <= 1e-12] = 1.0
        tangents = derivatives / tangent_norm

    if len(centers) >= 3:
        distances = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
        )
        edge_margin = window_length_px
        trim = (distances >= edge_margin) & (
            distances <= distances[-1] - edge_margin
        )
        centers = centers[trim]
        tangents = tangents[trim]

    windows: list[ScanWindow] = []
    for index, (center, tangent) in enumerate(zip(centers, tangents)):
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        left = _find_edge(binary, center, normal, max_ray_px, ray_step_px)
        right = _find_edge(binary, center, -normal, max_ray_px, ray_step_px)
        valid = left is not None and right is not None
        width_px = 0.0
        midpoint = None
        if valid:
            left_array = np.asarray(left, dtype=np.float64)
            right_array = np.asarray(right, dtype=np.float64)
            width_px = float(np.linalg.norm(left_array - right_array))
            midpoint_array = (left_array + right_array) / 2.0
            midpoint = (float(midpoint_array[0]), float(midpoint_array[1]))
        windows.append(
            ScanWindow(
                index=index,
                cx=float(center[0]),
                cy=float(center[1]),
                angle_deg=float(np.degrees(np.arctan2(tangent[1], tangent[0]))),
                length_px=window_length_px,
                width_px=width_px,
                left=left,
                right=right,
                midpoint=midpoint,
                valid=valid,
            )
        )

    break_windows = _mark_break_runs(windows, min_missing_windows)
    missing_edge_windows = sum(1 for window in windows if not window.valid)
    widths = [window.width_px for window in windows if window.valid]
    reference_width_px = float(np.median(widths)) if widths else 0.0
    for window in windows:
        window.window_width_px = max(
            reference_width_px * window_width_factor,
            window.width_px * window_width_factor,
            window_length_px,
        )

    centerline_segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for window in windows:
        if window.valid and window.midpoint is not None:
            current.append(window.midpoint)
        elif current:
            centerline_segments.append(current)
            current = []
    if current:
        centerline_segments.append(current)
    centerline_segments = [
        _smooth_points(segment, smooth_window)
        for segment in centerline_segments
        if len(segment) >= 2
    ]

    width_px = float(np.median(widths)) if widths else 0.0
    min_width_px = float(np.min(widths)) if widths else 0.0
    max_width_px = float(np.max(widths)) if widths else 0.0
    width_std_px = float(np.std(widths)) if widths else 0.0
    curve_deviation_px = max(
        (_curve_deviation(segment) for segment in centerline_segments),
        default=0.0,
    )
    scale = float(mm_per_px) if mm_per_px else 0.0
    return ScanResult(
        windows=windows,
        centerline_segments=centerline_segments,
        widths_px=widths,
        width_px=width_px,
        width_mm=width_px * scale,
        min_width_px=min_width_px,
        max_width_px=max_width_px,
        width_std_px=width_std_px,
        min_width_mm=min_width_px * scale,
        max_width_mm=max_width_px * scale,
        width_std_mm=width_std_px * scale,
        curve_deviation_px=curve_deviation_px,
        curve_deviation_mm=curve_deviation_px * scale,
        break_windows=break_windows,
        missing_edge_windows=missing_edge_windows,
    )


def apply_scan_result(
    measure: MeasureResult,
    scan: ScanResult,
) -> MeasureResult:
    """把扫描结果写入公开量测结构，并优先采用扫描宽度。"""

    measure.scan_windows = scan.windows
    measure.centerline_segments = scan.centerline_segments
    measure.break_windows = scan.break_windows
    measure.missing_edge_windows = scan.missing_edge_windows
    measure.min_width_px = scan.min_width_px
    measure.max_width_px = scan.max_width_px
    measure.width_std_px = scan.width_std_px
    measure.min_width_mm = scan.min_width_mm
    measure.max_width_mm = scan.max_width_mm
    measure.width_std_mm = scan.width_std_mm
    measure.curve_deviation_px = scan.curve_deviation_px
    measure.curve_deviation_mm = scan.curve_deviation_mm
    if scan.width_px > 0:
        measure.width_px = scan.width_px
        measure.width_mm = scan.width_mm
    return measure


def translate_measure_coordinates(
    measure: MeasureResult,
    dx: float,
    dy: float,
) -> MeasureResult:
    """把裁剪坐标系中的扫描结果平移回原图坐标。"""

    if dx == 0 and dy == 0:
        return measure
    for window in measure.scan_windows:
        window.cx += dx
        window.cy += dy
        if window.left is not None:
            window.left = (window.left[0] + dx, window.left[1] + dy)
        if window.right is not None:
            window.right = (window.right[0] + dx, window.right[1] + dy)
        if window.midpoint is not None:
            window.midpoint = (
                window.midpoint[0] + dx,
                window.midpoint[1] + dy,
            )
    measure.centerline_segments = [
        [(x + dx, y + dy) for x, y in segment]
        for segment in measure.centerline_segments
    ]
    if measure.bbox is not None:
        measure.bbox.x1 += dx
        measure.bbox.y1 += dy
        measure.bbox.x2 += dx
        measure.bbox.y2 += dy
    return measure
