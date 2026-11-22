"""生成与 M06 示教轨迹一致的同轨迹验证数据集。

该数据集面向 VisionMaster ``边缘对模型缺陷检测``：

- 所有图像共用一条固定标准胶路；
- 只改变断胶、宽度或局部弯曲，避免把轨迹不同误当成缺陷；
- 同时提供正常控制图，用于检查正常样本误报。

合成数据仅用于流程验证和冒烟，不能作为真实产线验收数据。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .synthetic_dataset import _draw_scene

TASK = "m06_matched_glue_defect_validation"
SCENARIOS: tuple[str, ...] = (
    "reference_good",
    "normal",
    "break",
    "narrow",
    "wide",
    "bend",
)
SCENARIO_RESULTS: dict[str, tuple[str, int | None]] = {
    "reference_good": ("OK", None),
    "normal": ("OK", None),
    "break": ("BROKEN", 4),
    "narrow": ("TOO_NARROW", 1),
    "wide": ("TOO_WIDE", 1),
    "bend": ("BENT", 2),
}


@dataclass(frozen=True)
class M06ValidationSummary:
    """一次同轨迹 M06 验证集生成结果。"""

    output_dir: Path
    seed: int
    sample_count: int
    scenario_counts: dict[str, int]
    reference_image: str


@dataclass(frozen=True)
class _FixedPath:
    """固定标准胶路的中心线、切向法线和基准宽度。"""

    xs: np.ndarray
    ys: np.ndarray
    widths: np.ndarray


def _uniform(
    rng: np.random.Generator,
    bounds: list[float] | tuple[float, float],
) -> float:
    """从配置区间采样一个浮点数。"""

    if len(bounds) != 2:
        raise ValueError(f"区间配置必须包含两个值：{bounds}")
    low, high = float(bounds[0]), float(bounds[1])
    if high < low:
        raise ValueError(f"区间上限小于下限：{bounds}")
    return float(rng.uniform(low, high))


def _build_fixed_path(
    height: int,
    width: int,
    settings: Mapping[str, Any],
) -> _FixedPath:
    """按配置生成所有样本共用的标准胶路。"""

    path_cfg = settings["fixed_trajectory"]
    point_count = int(path_cfg["centerline_points"])
    if point_count < 3:
        raise ValueError("fixed_trajectory.centerline_points 必须至少为 3")

    ts = np.linspace(-1.0, 1.0, point_count)
    length_px = width * float(path_cfg["length_ratio"])
    local_x = ts * length_px / 2.0
    amplitude = float(path_cfg["amplitude_px"])
    frequency = float(path_cfg["wave_count"])
    phase = float(path_cfg["phase_rad"])
    slope = float(path_cfg["slope_ratio"])
    local_y = (
        amplitude * np.sin(math.pi * frequency * ts + phase)
        + slope * local_x
    )

    angle = math.radians(float(path_cfg["angle_deg"]))
    center_x = width * float(path_cfg["center_x_ratio"])
    center_y = height * float(path_cfg["center_y_ratio"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xs = center_x + cos_a * local_x - sin_a * local_y
    ys = center_y + sin_a * local_x + cos_a * local_y

    base_width = float(path_cfg["base_width_px"])
    width_variation = float(path_cfg["width_variation"])
    width_phase = float(path_cfg["width_phase_rad"])
    widths = base_width * (
        1.0
        + width_variation
        * np.sin(math.pi * frequency * ts + width_phase)
    )
    widths = np.maximum(widths, float(path_cfg["min_width_px"]))
    return _FixedPath(xs=xs, ys=ys, widths=widths)


def _unit_normals(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算路径局部单位法向量。"""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norm = np.hypot(dx, dy)
    norm[norm <= 1e-12] = 1.0
    return -dy / norm, dx / norm


def _render_strip_mask(
    height: int,
    width: int,
    xs: np.ndarray,
    ys: np.ndarray,
    strip_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按中心线和局部宽度绘制胶条掩膜。"""

    normals_x, normals_y = _unit_normals(xs, ys)
    left = np.stack(
        [
            xs + normals_x * strip_widths / 2.0,
            ys + normals_y * strip_widths / 2.0,
        ],
        axis=1,
    )
    right = np.stack(
        [
            xs - normals_x * strip_widths / 2.0,
            ys - normals_y * strip_widths / 2.0,
        ],
        axis=1,
    )
    polygon = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask, normals_x, normals_y


def _bbox(mask: np.ndarray) -> list[int] | None:
    """返回掩膜外接框 ``(x1, y1, x2, y2)``。"""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]


def _path_length(xs: np.ndarray, ys: np.ndarray) -> float:
    """返回折线总弧长。"""

    return float(np.linalg.norm(np.diff(np.stack([xs, ys], axis=1), axis=0), axis=1).sum())


def _apply_break(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    widths: np.ndarray,
    normals_x: np.ndarray,
    normals_y: np.ndarray,
    center_ratio: float,
    length_px: float,
    margin_px: float,
) -> dict[str, Any] | None:
    """在固定轨迹中间切除一段胶条。"""

    segments = np.linalg.norm(np.diff(np.stack([xs, ys], axis=1), axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(cumulative[-1])
    center_distance = (center_ratio + 1.0) / 2.0 * total
    start_distance = max(0.0, center_distance - length_px / 2.0)
    end_distance = min(total, center_distance + length_px / 2.0)
    indices = np.flatnonzero(
        (cumulative >= start_distance) & (cumulative <= end_distance)
    )
    if len(indices) < 2:
        return None

    half_width = widths[indices] / 2.0 + margin_px
    left = np.stack(
        [
            xs[indices] + normals_x[indices] * half_width,
            ys[indices] + normals_y[indices] * half_width,
        ],
        axis=1,
    )
    right = np.stack(
        [
            xs[indices] - normals_x[indices] * half_width,
            ys[indices] - normals_y[indices] * half_width,
        ],
        axis=1,
    )
    cut_mask = np.zeros_like(mask)
    polygon = np.concatenate([left, right[::-1]], axis=0)
    cv2.fillPoly(cut_mask, [np.round(polygon).astype(np.int32)], 255)
    cut_mask &= mask
    mask[cut_mask > 0] = 0
    box = _bbox(cut_mask)
    if box is None:
        return None
    return {
        "type": "break",
        "box": box,
        "area_px": int((cut_mask > 0).sum()),
        "length_px": round(end_distance - start_distance, 3),
        "center_ratio": round(center_ratio, 6),
    }


def _apply_bend(
    xs: np.ndarray,
    ys: np.ndarray,
    center_ratio: float,
    span_ratio: float,
    amplitude_px: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """在固定标准轨迹上叠加局部平滑弯曲。"""

    ts = np.linspace(-1.0, 1.0, len(xs))
    start = center_ratio - span_ratio / 2.0
    end = center_ratio + span_ratio / 2.0
    u = np.clip((ts - start) / max(end - start, 1e-12), 0.0, 1.0)
    weight = np.sin(math.pi * u) ** 2
    normals_x, normals_y = _unit_normals(xs, ys)
    bend_xs = xs + normals_x * amplitude_px * weight
    bend_ys = ys + normals_y * amplitude_px * weight
    return bend_xs, bend_ys, float(np.max(np.hypot(bend_xs - xs, bend_ys - ys)))


def _render_image(
    rng: np.random.Generator,
    mask: np.ndarray,
    height: int,
    width: int,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """在纯背景上渲染胶条，并叠加轻微成像噪声。"""

    image = _draw_scene(rng, height, width, settings["background"])
    appearance = settings["glue_appearance"]
    glue_value = _uniform(rng, appearance["value_range"])
    texture = rng.normal(
        0.0,
        float(appearance["texture_sigma"]),
        size=(height, width),
    )
    strip = mask > 0
    image[strip] = glue_value + texture[strip]

    blur_kernel = int(_uniform(rng, settings["camera"]["blur_kernel"]))
    blur_kernel = max(1, blur_kernel)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel > 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX=0)
    image += rng.normal(
        0.0,
        float(settings["camera"]["noise_sigma"]),
        size=(height, width),
    ).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def _connected_component_count(mask: np.ndarray, min_area_px: int) -> int:
    """统计达到最小面积的连通域数量。"""

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas >= min_area_px).sum())


def _write_preview(
    root: Path,
    selected: dict[str, Path],
) -> None:
    """生成模板与四类缺陷的缩略总览图。"""

    order = ("reference_good", "normal", "break", "narrow", "wide", "bend")
    tile_width, tile_height = 480, 360
    tiles: list[np.ndarray] = []
    for scenario in order:
        path = selected.get(scenario)
        if path is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        resized = cv2.resize(image, (tile_width, tile_height))
        tile = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        text = f"{scenario} | {path.name}"
        cv2.rectangle(tile, (0, 0), (tile_width - 1, 30), (0, 0, 0), -1)
        cv2.putText(
            tile,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    overview = np.vstack(rows)
    if not cv2.imwrite(str(root / "preview_overview.png"), overview):
        raise OSError("无法写入 preview_overview.png")


def generate_m06_validation_dataset(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> M06ValidationSummary:
    """生成固定轨迹的 M06 验证集。

    输出目录必须不存在或为空，避免覆盖历史验证数据。
    """

    settings = config.get("m06_validation")
    if not isinstance(settings, Mapping):
        raise ValueError("配置缺少 m06_validation 段")

    actual_seed = int(seed if seed is not None else settings["seed"])
    total_samples = int(settings["total_samples"])
    counts = {name: int(settings["counts"][name]) for name in SCENARIOS}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("m06_validation.counts 中每个场景必须大于 0")
    if sum(counts.values()) != total_samples:
        raise ValueError(
            f"场景数量之和 {sum(counts.values())} 与 total_samples={total_samples} 不一致"
        )

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新版本目录：{root}")
    for relative in ("images", "masks/glue", "labels"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    height, width = (int(value) for value in settings["image_size"])
    base_path = _build_fixed_path(height, width, settings)
    base_normals_x, base_normals_y = _unit_normals(base_path.xs, base_path.ys)
    base_path_length_px = _path_length(base_path.xs, base_path.ys)
    reference_id = "m06_000_reference_good_000"
    reference_image = f"images/{reference_id}.png"

    rows: list[dict[str, Any]] = []
    scenario_counts = {name: 0 for name in SCENARIOS}
    scenario_first_image: dict[str, Path] = {}
    global_index = 0

    for scenario in SCENARIOS:
        for variant_index in range(counts[scenario]):
            rng = np.random.default_rng(
                actual_seed + global_index * 10_007 + 1_000_003
            )
            xs = base_path.xs.copy()
            ys = base_path.ys.copy()
            widths = base_path.widths.copy()
            width_scale = 1.0
            bend_amplitude = 0.0
            defects: list[dict[str, Any]] = []

            if scenario == "normal":
                width_scale = _uniform(
                    rng,
                    settings["variants"]["normal_width_scale"],
                )
            elif scenario == "narrow":
                width_scale = _uniform(
                    rng,
                    settings["variants"]["narrow_width_scale"],
                )
            elif scenario == "wide":
                width_scale = _uniform(
                    rng,
                    settings["variants"]["wide_width_scale"],
                )
            elif scenario == "bend":
                center_ratio = _uniform(
                    rng,
                    settings["variants"]["bend_center_ratio"],
                )
                span_ratio = _uniform(
                    rng,
                    settings["variants"]["bend_span_ratio"],
                )
                requested_amplitude = _uniform(
                    rng,
                    settings["variants"]["bend_amplitude_px"],
                )
                xs, ys, bend_amplitude = _apply_bend(
                    xs,
                    ys,
                    center_ratio,
                    span_ratio,
                    requested_amplitude,
                )

            widths *= width_scale
            mask, normals_x, normals_y = _render_strip_mask(
                height,
                width,
                xs,
                ys,
                widths,
            )

            if scenario == "break":
                break_length = _uniform(
                    rng,
                    settings["variants"]["break_length_px"],
                )
                break_center = _uniform(
                    rng,
                    settings["variants"]["break_center_ratio"],
                )
                break_record = _apply_break(
                    mask,
                    xs,
                    ys,
                    widths,
                    normals_x,
                    normals_y,
                    break_center,
                    break_length,
                    float(settings["variants"]["break_margin_px"]),
                )
                if break_record is None:
                    raise RuntimeError("断胶区域生成失败")
                defects.append(break_record)
            elif scenario in {"narrow", "wide"}:
                defects.append(
                    {
                        "type": "width",
                        "width_scale": round(width_scale, 6),
                        "box": _bbox(mask),
                    }
                )
            elif scenario == "bend":
                defects.append(
                    {
                        "type": "position",
                        "max_deviation_px": round(bend_amplitude, 3),
                        "box": _bbox(mask),
                    }
                )

            image = _render_image(rng, mask, height, width, settings)
            image_id = (
                f"m06_{global_index:03d}_{scenario}_{variant_index:03d}"
            )
            image_path = root / "images" / f"{image_id}.png"
            mask_path = root / "masks" / "glue" / f"{image_id}.png"
            label_path = root / "labels" / f"{image_id}.json"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"无法写入图像：{image_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"无法写入掩膜：{mask_path}")

            expected_result, expected_defect_type = SCENARIO_RESULTS[scenario]
            component_count = _connected_component_count(
                mask,
                int(settings["min_component_area_px"]),
            )
            label_data = {
                "image": f"images/{image_id}.png",
                "mask": f"masks/glue/{image_id}.png",
                "image_size": {"height": height, "width": width},
                "source": "synthetic",
                "task": TASK,
                "trajectory_id": settings["trajectory_id"],
                "reference_image": reference_image,
                "is_reference": int(
                    scenario == "reference_good" and variant_index == 0
                ),
                "scenario": scenario,
                "expected_result": expected_result,
                "expected_vm_defect_type": expected_defect_type,
                "defects": defects,
                "component_count": component_count,
                "width_scale": round(width_scale, 6),
                "nominal_width_px": round(float(np.median(widths)), 6),
                "min_width_px": round(float(np.min(widths)), 6),
                "max_width_px": round(float(np.max(widths)), 6),
                "pixel_format": "Mono8",
                "acceptance_eligible": False,
                "warning": "合成数据仅用于 M06 流程验证，不能替代真实数据验收。",
            }
            label_path.write_text(
                json.dumps(label_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            rows.append(
                {
                    "image_id": image_id,
                    "path": f"images/{image_id}.png",
                    "mask": f"masks/glue/{image_id}.png",
                    "trajectory_id": settings["trajectory_id"],
                    "reference_image": reference_image,
                    "is_reference": label_data["is_reference"],
                    "scenario": scenario,
                    "expected_result": expected_result,
                    "expected_vm_defect_type": (
                        "" if expected_defect_type is None else expected_defect_type
                    ),
                    "width_scale": f"{width_scale:.6f}",
                    "nominal_width_px": f"{np.median(widths):.3f}",
                    "min_width_px": f"{np.min(widths):.3f}",
                    "max_width_px": f"{np.max(widths):.3f}",
                    "break_length_px": (
                        f"{defects[0]['length_px']:.3f}"
                        if scenario == "break"
                        else ""
                    ),
                    "bend_deviation_px": (
                        f"{bend_amplitude:.3f}" if scenario == "bend" else ""
                    ),
                    "component_count": component_count,
                    "glue_pixels": int((mask > 0).sum()),
                    "generator_seed": actual_seed,
                    "sample_seed": int(
                        actual_seed + global_index * 10_007 + 1_000_003
                    ),
                    "pixel_format": "Mono8",
                    "acceptance_eligible": 0,
                }
            )
            scenario_counts[scenario] += 1
            scenario_first_image.setdefault(scenario, image_path)
            global_index += 1

    if global_index != total_samples:
        raise RuntimeError(f"实际生成 {global_index} 张，与目标 {total_samples} 不一致")

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    reference_data = {
        "reference_image": reference_image,
        "reference_mask": f"masks/glue/{reference_id}.png",
        "trajectory_id": settings["trajectory_id"],
        "nominal_width_px": round(float(np.median(base_path.widths)), 6),
        "path_length_px": round(base_path_length_px, 3),
    }
    (root / "reference.json").write_text(
        json.dumps(reference_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dataset_meta = {
        "dataset_version": settings["version"],
        "task": TASK,
        "source": "synthetic",
        "acceptance_eligible": False,
        "warning": "仅用于 M06 同轨迹流程验证，不能作为真实产线验收数据。",
        "seed": actual_seed,
        "sample_count": total_samples,
        "pixel_format": "Mono8",
        "image_size": {"height": height, "width": width},
        "trajectory_id": settings["trajectory_id"],
        "reference_image": reference_image,
        "scenario_counts": scenario_counts,
        "expected_defect_type_ids": {
            "width": 1,
            "position": 2,
            "break": 4,
        },
        "generation_config": dict(settings),
    }
    (root / "dataset.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = f"""# M06 同轨迹验证集

- 版本：`{settings["version"]}`
- 总样本：`{total_samples}`
- 模板图：`{reference_image}`
- 轨迹 ID：`{settings["trajectory_id"]}`
- 场景数量：`{scenario_counts}`

## VM 预期

| scenario | expected_result | DefectType |
|---|---|---:|
| reference_good / normal | OK | 无 |
| break | BROKEN | 4 |
| narrow | TOO_NARROW | 1 |
| wide | TOO_WIDE | 1 |
| bend | BENT | 2 |

所有图像共用同一标准轨迹。`DefectType=1` 为宽度缺陷，`2` 为位置偏移，
`4` 为断裂缺陷。

## 使用方式

1. 用 `{reference_image}` 在 M06 中建立示教轨迹和标准边缘对。
2. 加载同一 `trajectory_id` 的其他图像。
3. 正常图应保持 OK；异常图应命中对应结果。

该数据集为合成数据，`acceptance_eligible=false`，不能替代真实相机数据验收。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    _write_preview(root, scenario_first_image)

    return M06ValidationSummary(
        output_dir=root,
        seed=actual_seed,
        sample_count=global_index,
        scenario_counts=scenario_counts,
        reference_image=reference_image,
    )
