"""面向 VisionTrain 图像分割训练的涂胶合成数据集生成器。

本模块生成程序化合成图像、严格对齐的掩膜、逐图标签和 manifest，用于：

- 验证 VisionTrain 的图像导入、标注映射、训练和模型导出流程；
- 在真实数据未到位前搭建 VisionMaster 流程原型；
- 在受控条件下补充断胶、气泡、拉丝和污染等困难样本。

**边界**：所有输出均为合成数据，只可用于冒烟、流程验证和预训练。
最终验收必须使用真实相机图像和人工真值，见 AGENTS.md R1 与
docs/algo/p0-spec.md §2、§6。
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

DEFECT_CLASS_IDS: dict[str, int] = {
    "background": 0,
    "break": 1,
    "bubble": 2,
    "stringing": 3,
    "contamination": 4,
}

SCENARIOS: tuple[str, ...] = (
    "ok",
    "narrow",
    "wide",
    "break",
    "bubble",
    "stringing",
    "contamination",
    "reflection",
)

SCENARIO_LABELS: dict[str, str] = {
    "ok": "OK",
    "narrow": "TOO_NARROW",
    "wide": "TOO_WIDE",
    "break": "BROKEN",
    "bubble": "BUBBLE",
    "stringing": "STRINGING",
    "contamination": "CONTAMINATION",
    "reflection": "REFLECTION",
}


@dataclass(frozen=True)
class GeneratedSample:
    """一帧合成图像及其像素级标注。"""

    image: np.ndarray
    glue_mask: np.ndarray
    defect_mask: np.ndarray
    hard_background_mask: np.ndarray
    scenario: str
    label: str
    nominal_width_px: float
    defect_records: list[dict[str, Any]]


@dataclass(frozen=True)
class DatasetSummary:
    """一次数据集生成任务的摘要。"""

    output_dir: Path
    seed: int
    sample_count: int
    split_counts: dict[str, int]
    scenario_counts: dict[str, int]


@dataclass(frozen=True)
class _GlueGeometry:
    """胶条中心线、法向量和基础掩膜。"""

    xs: np.ndarray
    ys: np.ndarray
    normals_x: np.ndarray
    normals_y: np.ndarray
    widths: np.ndarray
    mask: np.ndarray

    def center(self, index: int) -> tuple[float, float]:
        return float(self.xs[index]), float(self.ys[index])

    def normal(self, index: int) -> tuple[float, float]:
        return float(self.normals_x[index]), float(self.normals_y[index])


def _uniform(rng: np.random.Generator, bounds: list[float] | tuple[float, float]) -> float:
    """从闭区间配置中采样一个浮点数。"""
    if len(bounds) != 2:
        raise ValueError(f"区间配置必须包含两个值：{bounds}")
    low, high = float(bounds[0]), float(bounds[1])
    if high < low:
        raise ValueError(f"区间上限小于下限：{bounds}")
    return float(rng.uniform(low, high))


def _weighted_choice(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    """按配置权重选择一个场景。"""
    names = [name for name in SCENARIOS if float(weights.get(name, 0.0)) > 0]
    if not names:
        raise ValueError("scenario_weights 至少需要一个正值")
    probs = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    probs /= probs.sum()
    return str(rng.choice(names, p=probs))


def _odd_kernel(value: int) -> int:
    """把配置中的模糊核大小规范为不小于 1 的奇数。"""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _draw_background(
    rng: np.random.Generator,
    height: int,
    width: int,
    scene_cfg: Mapping[str, Any],
) -> np.ndarray:
    """生成带亮度梯度、工件平面、纹理、划痕和噪声的背景。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    projected = xx * math.cos(angle) + yy * math.sin(angle)
    projected -= projected.min()
    projected /= max(float(projected.max()), 1.0)

    base = _uniform(rng, scene_cfg["background_range"])
    gradient = _uniform(rng, scene_cfg["gradient_strength"]) * 255.0
    image = base + gradient * (projected - 0.5)

    texture_sigma = _uniform(rng, scene_cfg["texture_sigma"])
    coarse = cv2.GaussianBlur(
        rng.normal(0.0, texture_sigma, size=(height, width)).astype(np.float32),
        (0, 0),
        sigmaX=texture_sigma * 2.0,
        sigmaY=texture_sigma * 2.0,
    )
    image += coarse

    workpiece = np.zeros((height, width), dtype=np.uint8)
    rect = (
        (width * 0.5, height * 0.54),
        (width * 0.92, height * 0.64),
        float(rng.uniform(-8.0, 8.0)),
    )
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(workpiece, [points], 255)
    workpiece_value = _uniform(rng, scene_cfg["workpiece_range"])
    image[workpiece > 0] = workpiece_value + coarse[workpiece > 0] * 1.2

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=float(rng.uniform(20.0, 70.0)),
        thickness=int(rng.integers(1, 4)),
    )

    for _ in range(int(rng.integers(0, 5))):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        length = int(rng.integers(width // 20, max(width // 5, width // 20 + 1)))
        angle_deg = float(rng.uniform(-25.0, 25.0))
        x2 = int(np.clip(x1 + length * math.cos(math.radians(angle_deg)), 0, width - 1))
        y2 = int(np.clip(y1 + length * math.sin(math.radians(angle_deg)), 0, height - 1))
        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            float(rng.uniform(45.0, 105.0)),
            int(rng.integers(1, 3)),
        )

    return image.astype(np.float32)


def _build_glue_geometry(
    rng: np.random.Generator,
    height: int,
    width: int,
    glue_cfg: Mapping[str, Any],
    centerline_points: int,
    width_scale: float,
) -> _GlueGeometry:
    """根据随机中心线和宽度变化生成胶条基础几何。"""
    angle = math.radians(_uniform(rng, glue_cfg["angle_deg"]))
    length = float(height) * _uniform(rng, glue_cfg["length_ratio"])
    half_length = min(length / 2.0, min(height, width) * 0.48)

    ts = np.linspace(-1.0, 1.0, centerline_points)
    local_x = ts * half_length
    amplitude = _uniform(rng, glue_cfg["amplitude_px"])
    frequency = _uniform(rng, glue_cfg["wave_count"])
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    slope = _uniform(rng, glue_cfg["slope_ratio"])
    local_y = amplitude * np.sin(math.pi * frequency * ts + phase) + slope * local_x

    center_x = width * 0.5 + float(rng.uniform(-0.06, 0.06)) * width
    center_y = height * 0.5 + float(rng.uniform(-0.06, 0.06)) * height
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xs = center_x + cos_a * local_x - sin_a * local_y
    ys = center_y + sin_a * local_x + cos_a * local_y

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    normals_x = -dy / norm
    normals_y = dx / norm

    base_width = _uniform(rng, glue_cfg["width_px"])
    variation = _uniform(rng, glue_cfg["width_variation"])
    width_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    widths = base_width * width_scale * (
        1.0 + variation * np.sin(math.pi * frequency * ts + width_phase)
    )
    widths = np.maximum(widths, float(glue_cfg["min_width_px"]))

    left = np.stack(
        [xs + normals_x * widths / 2.0, ys + normals_y * widths / 2.0], axis=1
    )
    right = np.stack(
        [xs - normals_x * widths / 2.0, ys - normals_y * widths / 2.0], axis=1
    )
    polygon = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return _GlueGeometry(xs, ys, normals_x, normals_y, widths, mask)


def _paint_class(mask: np.ndarray, polygon: np.ndarray, class_id: int) -> None:
    """只在尚未标注的像素上写入类别，避免缺陷类别互相覆盖。"""
    candidate = np.zeros_like(mask)
    cv2.fillPoly(candidate, [np.round(polygon).astype(np.int32)], class_id)
    write = (candidate > 0) & (mask == 0)
    mask[write] = class_id


def _add_break(
    rng: np.random.Generator,
    glue: _GlueGeometry,
    defect_mask: np.ndarray,
    defect_cfg: Mapping[str, Any],
) -> np.ndarray:
    """切断胶条并返回断口区域掩膜。"""
    gap_ratio = _uniform(rng, defect_cfg["break_length_ratio"])
    gap_center = float(rng.uniform(-0.2, 0.2))
    start = gap_center - gap_ratio / 2.0
    end = gap_center + gap_ratio / 2.0
    indices = np.flatnonzero((np.linspace(-1.0, 1.0, len(glue.xs)) >= start) & (
        np.linspace(-1.0, 1.0, len(glue.xs)) <= end
    ))
    if len(indices) < 2:
        indices = np.array([len(glue.xs) // 2 - 1, len(glue.xs) // 2])

    half_width = glue.widths[indices] / 2.0 + float(defect_cfg["break_margin_px"])
    left = np.stack(
        [
            glue.xs[indices] + glue.normals_x[indices] * half_width,
            glue.ys[indices] + glue.normals_y[indices] * half_width,
        ],
        axis=1,
    )
    right = np.stack(
        [
            glue.xs[indices] - glue.normals_x[indices] * half_width,
            glue.ys[indices] - glue.normals_y[indices] * half_width,
        ],
        axis=1,
    )
    break_mask = np.zeros_like(defect_mask)
    polygon = np.concatenate([left, right[::-1]], axis=0)
    cv2.fillPoly(break_mask, [np.round(polygon).astype(np.int32)], 255)
    break_mask &= glue.mask
    defect_mask[break_mask > 0] = DEFECT_CLASS_IDS["break"]
    return break_mask


def _add_bubble(
    rng: np.random.Generator,
    image: np.ndarray,
    glue: _GlueGeometry,
    defect_mask: np.ndarray,
    defect_cfg: Mapping[str, Any],
) -> None:
    """在胶条内部生成气泡/凹坑，并写入像素类别。"""
    index = int(rng.integers(len(glue.xs) // 5, max(len(glue.xs) * 4 // 5, 2)))
    cx, cy = glue.center(index)
    radius = max(
        float(defect_cfg["bubble_min_radius_px"]),
        float(glue.widths[index]) * _uniform(rng, defect_cfg["bubble_radius_ratio"]),
    )
    axes = (int(radius), max(1, int(radius * float(rng.uniform(0.65, 1.0)))))
    angle = float(rng.uniform(0.0, 180.0))

    local = np.zeros_like(defect_mask)
    cv2.ellipse(local, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1)
    local &= glue.mask
    write = (local > 0) & (defect_mask == 0)
    image[write] = np.clip(
        image[write] * float(rng.uniform(0.35, 0.75))
        + float(rng.uniform(15.0, 55.0)),
        0,
        255,
    )
    defect_mask[write] = DEFECT_CLASS_IDS["bubble"]

    ring = cv2.ellipse(
        np.zeros_like(glue.mask),
        (int(cx), int(cy)),
        axes,
        angle,
        0,
        360,
        255,
        max(1, int(radius * float(defect_cfg["bubble_ring_ratio"]))),
    )
    ring = (ring > 0) & (glue.mask > 0) & (defect_mask == 0)
    image[ring] = np.clip(
        image[ring] * float(rng.uniform(1.05, 1.25)) + 10.0, 0, 255
    )
    defect_mask[ring] = DEFECT_CLASS_IDS["bubble"]


def _add_stringing(
    rng: np.random.Generator,
    image: np.ndarray,
    glue: _GlueGeometry,
    defect_mask: np.ndarray,
    defect_cfg: Mapping[str, Any],
) -> None:
    """从胶条边缘生成一条细拉丝。"""
    index = int(rng.integers(len(glue.xs) // 6, max(len(glue.xs) * 5 // 6, 2)))
    cx, cy = glue.center(index)
    nx, ny = glue.normal(index)
    direction = float(rng.choice([-1.0, 1.0]))
    start = np.array(
        [cx + direction * nx * glue.widths[index] / 2.0,
         cy + direction * ny * glue.widths[index] / 2.0]
    )
    length = max(
        float(defect_cfg["stringing_min_length_px"]),
        float(glue.widths[index]) * _uniform(rng, defect_cfg["stringing_length_ratio"]),
    )
    tangent = np.array([-ny, nx]) * direction
    bend = _uniform(rng, defect_cfg["stringing_bend_ratio"]) * length
    p1 = start + np.array([nx, ny]) * direction * length * 0.5 + tangent * bend * 0.5
    p2 = start + np.array([nx, ny]) * direction * length + tangent * bend
    line_width = int(defect_cfg["stringing_thickness_px"])

    local = np.zeros_like(defect_mask)
    cv2.polylines(
        local,
        [np.round(np.stack([start, p1, p2])).astype(np.int32)],
        isClosed=False,
        color=255,
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    write = (local > 0) & (defect_mask == 0)
    image[write] = np.clip(
        image[write] * float(rng.uniform(0.65, 0.95))
        + float(rng.uniform(35.0, 80.0)),
        0,
        255,
    )
    defect_mask[write] = DEFECT_CLASS_IDS["stringing"]


def _add_contamination(
    rng: np.random.Generator,
    image: np.ndarray,
    glue: _GlueGeometry,
    defect_mask: np.ndarray,
    defect_cfg: Mapping[str, Any],
) -> None:
    """在胶条附近生成不规则脏污/异物。"""
    index = int(rng.integers(len(glue.xs) // 6, max(len(glue.xs) * 5 // 6, 2)))
    cx, cy = glue.center(index)
    radius = max(
        float(defect_cfg["contamination_min_radius_px"]),
        float(glue.widths[index])
        * _uniform(rng, defect_cfg["contamination_radius_ratio"]),
    )
    count = int(rng.integers(8, 15))
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radial = radius * rng.uniform(0.55, 1.25, size=count)
    points = np.stack(
        [cx + np.cos(angles) * radial, cy + np.sin(angles) * radial], axis=1
    )
    _paint_class(defect_mask, points, DEFECT_CLASS_IDS["contamination"])

    painted = defect_mask == DEFECT_CLASS_IDS["contamination"]
    image[painted] = np.clip(
        image[painted] * float(rng.uniform(0.25, 0.65))
        + float(rng.uniform(10.0, 45.0)),
        0,
        255,
    )


def _add_reflection(
    rng: np.random.Generator,
    image: np.ndarray,
    glue: _GlueGeometry,
    hard_background_mask: np.ndarray,
    scene_cfg: Mapping[str, Any],
    strong: bool,
) -> None:
    """加入反光条纹，并把反光区域记录为困难背景。"""
    index = int(rng.integers(len(glue.xs) // 5, max(len(glue.xs) * 4 // 5, 2)))
    cx, cy = glue.center(index)
    rx = max(3, int(glue.widths[index] * float(rng.uniform(0.8, 1.8))))
    ry = max(
        8,
        int(
            glue.widths[index]
            * _uniform(
                rng,
                scene_cfg["reflection_length_ratio"] if strong else scene_cfg["highlight_length_ratio"],
            )
        ),
    )
    highlight = np.zeros_like(hard_background_mask)
    cv2.ellipse(highlight, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 255, -1)
    highlight = cv2.GaussianBlur(
        highlight,
        (0, 0),
        sigmaX=max(1.0, rx / 3.0),
        sigmaY=max(1.0, ry / 4.0),
    )
    gain = float(
        scene_cfg["strong_reflection_gain"] if strong else scene_cfg["highlight_gain"]
    )
    image += highlight.astype(np.float32) / 255.0 * gain
    hard_background_mask[(highlight > 20) & (glue.mask > 0)] = 255


def _bbox(mask: np.ndarray) -> list[int] | None:
    """返回掩膜外接框，格式与项目 Box 一致。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def generate_sample(
    settings: Mapping[str, Any],
    sample_seed: int,
) -> GeneratedSample:
    """生成一帧合成涂胶图像及像素级标注。"""
    height, width = (int(v) for v in settings["image_size"])
    rng = np.random.default_rng(sample_seed)
    scenario = _weighted_choice(rng, settings["scenario_weights"])

    image = _draw_background(rng, height, width, settings["scene"])
    scale_key = {"narrow": "narrow_scale", "wide": "wide_scale"}.get(scenario)
    width_scale = (
        _uniform(rng, settings["variants"][scale_key]) if scale_key else 1.0
    )
    glue = _build_glue_geometry(
        rng,
        height,
        width,
        settings["glue"],
        int(settings["centerline_points"]),
        width_scale,
    )
    glue_mask = glue.mask.copy()
    defect_mask = np.zeros((height, width), dtype=np.uint8)
    hard_background_mask = np.zeros((height, width), dtype=np.uint8)

    break_mask = np.zeros((height, width), dtype=np.uint8)
    if scenario == "break":
        break_mask = _add_break(rng, glue, defect_mask, settings["defects"])
        glue_mask[break_mask > 0] = 0

    strip = glue_mask > 0
    glue_value = _uniform(rng, settings["glue"]["value_range"])
    centerline_noise = rng.normal(0.0, float(settings["glue"]["texture_sigma"]), size=(height, width))
    image[strip] = glue_value + centerline_noise[strip]

    if scenario == "bubble":
        _add_bubble(rng, image, glue, defect_mask, settings["defects"])
    elif scenario == "stringing":
        _add_stringing(rng, image, glue, defect_mask, settings["defects"])
    elif scenario == "contamination":
        _add_contamination(rng, image, glue, defect_mask, settings["defects"])

    if scenario == "reflection" or bool(
        rng.random() < float(settings["scene"]["highlight_probability"])
    ):
        _add_reflection(
            rng,
            image,
            glue,
            hard_background_mask,
            settings["scene"],
            strong=scenario == "reflection",
        )

    blur_kernel = _odd_kernel(int(_uniform(rng, settings["camera"]["blur_kernel"])))
    if blur_kernel > 1:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX=0)
    image += rng.normal(
        0.0,
        float(settings["camera"]["noise_sigma"]),
        size=(height, width),
    ).astype(np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)

    defect_records: list[dict[str, Any]] = []
    for class_name, class_id in DEFECT_CLASS_IDS.items():
        if class_id == 0:
            continue
        class_mask = np.where(defect_mask == class_id, 255, 0).astype(np.uint8)
        box = _bbox(class_mask)
        area = int((class_mask > 0).sum())
        if area > 0 and box is not None:
            defect_records.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "area_px": area,
                    "box": box,
                }
            )

    return GeneratedSample(
        image=image,
        glue_mask=np.where(glue_mask > 0, 255, 0).astype(np.uint8),
        defect_mask=defect_mask,
        hard_background_mask=hard_background_mask,
        scenario=scenario,
        label=SCENARIO_LABELS[scenario],
        nominal_width_px=float(np.median(glue.widths)),
        defect_records=defect_records,
    )


def _assign_group_splits(
    rng: np.random.Generator,
    group_count: int,
    split_ratios: Mapping[str, float],
) -> list[str]:
    """按工件组分配 split，避免同一工件的多帧跨集合。"""
    if group_count <= 0:
        raise ValueError("group_count 必须大于 0")
    names = ("train", "val", "test")
    ratios = np.asarray([float(split_ratios[name]) for name in names], dtype=np.float64)
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("split 比例必须非负且总和大于 0")
    ratios /= ratios.sum()

    counts = np.floor(ratios * group_count).astype(int)
    remainder = group_count - int(counts.sum())
    for index in np.argsort(-(ratios * group_count - counts))[:remainder]:
        counts[index] += 1

    if group_count >= len(names):
        for index in range(len(names)):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] = 1

    assignments = [name for name, count in zip(names, counts) for _ in range(count)]
    rng.shuffle(assignments)
    return assignments


def generate_dataset(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    samples: int | None = None,
    seed: int | None = None,
) -> DatasetSummary:
    """生成一个可复现的涂胶分割数据集。

    参数
    ----
    output_dir:
        输出目录。目录必须不存在或为空，避免静默覆盖历史数据。
    config:
        完整项目配置，其中必须包含 ``synthetic_dataset``。
    samples:
        可选样本数覆盖，主要用于小规模验证。
    seed:
        可选随机种子覆盖。
    """
    settings = config.get("synthetic_dataset") if isinstance(config, dict) else None
    if not isinstance(settings, Mapping):
        raise ValueError("配置缺少 synthetic_dataset 段")

    sample_count = int(samples if samples is not None else settings["samples"])
    actual_seed = int(seed if seed is not None else settings["seed"])
    frames_per_object = int(settings["frames_per_object"])
    if sample_count <= 0:
        raise ValueError("samples 必须大于 0")
    if frames_per_object <= 0:
        raise ValueError("frames_per_object 必须大于 0")

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新版本目录：{root}")

    for relative in (
        "images",
        "masks/glue",
        "masks/defects",
        "masks/hard_background",
        "labels",
        "splits",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    group_count = math.ceil(sample_count / frames_per_object)
    split_rng = np.random.default_rng(actual_seed ^ 0x5F3759DF)
    group_splits = _assign_group_splits(split_rng, group_count, settings["split"])

    manifest_rows: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    scenario_counts: dict[str, int] = {name: 0 for name in SCENARIOS}

    for sample_index in range(sample_count):
        group_index = sample_index // frames_per_object
        frame_index = sample_index % frames_per_object
        object_id = f"synobj_{group_index:06d}"
        batch = f"synbatch_{group_index // 10:04d}"
        split = group_splits[group_index]
        image_id = f"syn_{actual_seed}_{sample_index:06d}"
        scene_seed = actual_seed + group_index * 10_007 + frame_index * 1_000_003

        sample = generate_sample(settings, scene_seed)
        image_path = root / "images" / f"{image_id}.png"
        glue_path = root / "masks" / "glue" / f"{image_id}.png"
        defect_path = root / "masks" / "defects" / f"{image_id}.png"
        hard_path = root / "masks" / "hard_background" / f"{image_id}.png"
        label_path = root / "labels" / f"{image_id}.json"

        if not cv2.imwrite(str(image_path), sample.image):
            raise OSError(f"无法写入图像：{image_path}")
        if not cv2.imwrite(str(glue_path), sample.glue_mask):
            raise OSError(f"无法写入胶条掩膜：{glue_path}")
        if not cv2.imwrite(str(defect_path), sample.defect_mask):
            raise OSError(f"无法写入缺陷掩膜：{defect_path}")
        if not cv2.imwrite(str(hard_path), sample.hard_background_mask):
            raise OSError(f"无法写入困难背景掩膜：{hard_path}")

        label_data = {
            "image": f"images/{image_id}.png",
            "image_size": {
                "height": int(sample.image.shape[0]),
                "width": int(sample.image.shape[1]),
            },
            "source": "synthetic",
            "scene_id": object_id,
            "split": split,
            "scenario": sample.scenario,
            "label": sample.label,
            "acceptance_eligible": False,
            "masks": {
                "glue": f"masks/glue/{image_id}.png",
                "defects": f"masks/defects/{image_id}.png",
                "hard_background": f"masks/hard_background/{image_id}.png",
            },
            "class_ids": DEFECT_CLASS_IDS,
            "defects": sample.defect_records,
            "nominal_width_px": sample.nominal_width_px,
        }
        label_path.write_text(
            json.dumps(label_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "image_id": image_id,
                "path": f"images/{image_id}.png",
                "split": split,
                "object_id": object_id,
                "batch": batch,
                "source": "synthetic",
                "scenario": sample.scenario,
                "label": sample.label,
                "defect_types": ";".join(
                    record["class"] for record in sample.defect_records
                ),
                "nominal_width_px": f"{sample.nominal_width_px:.3f}",
                "generator_seed": actual_seed,
                "scene_seed": scene_seed,
                "pixel_format": "Mono8",
                "acceptance_eligible": 0,
            }
        )
        split_ids[split].append(image_id)
        scenario_counts[sample.scenario] += 1

    for split, image_ids in split_ids.items():
        (root / "splits" / f"{split}.txt").write_text(
            "\n".join(image_ids) + ("\n" if image_ids else ""),
            encoding="utf-8",
        )

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_meta = {
        "dataset_version": settings["version"],
        "source": "synthetic",
        "acceptance_eligible": False,
        "warning": "仅用于 VisionTrain 流程验证、冒烟和预训练；不得作为真实指标验收数据。",
        "seed": actual_seed,
        "sample_count": sample_count,
        "frames_per_object": frames_per_object,
        "class_ids": DEFECT_CLASS_IDS,
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "scenario_counts": scenario_counts,
        "generation_config": settings,
    }
    (root / "dataset.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return DatasetSummary(
        output_dir=root,
        seed=actual_seed,
        sample_count=sample_count,
        split_counts={name: len(ids) for name, ids in split_ids.items()},
        scenario_counts=scenario_counts,
    )
