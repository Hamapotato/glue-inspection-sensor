"""面向 VisionTrain 的二分类涂胶合成数据集生成器。

本模块只生成以下两个语义类别：

- ``background``：背景和没有胶条的区域；
- ``glue``：背景上的可见胶条。

图像场景分为“背景上直接有胶条”和“纯背景”两类，不绘制工件板或其他承载物。
所有输出均为合成数据，只可用于冒烟、流程验证和预训练，不得作为真实指标验收数据。
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

TASK = "binary_glue_background_segmentation"
CLASS_IDS: dict[str, int] = {
    "background": 0,
    "glue": 1,
}
SCENARIOS: tuple[str, ...] = (
    "glue",
    "background",
    "break",
    "narrow",
    "wide",
    "width_unstable",
    "curve",
)
SCENARIO_LABELS: dict[str, str] = {
    "glue": "GLUE",
    "background": "BACKGROUND",
    "break": "BROKEN",
    "narrow": "TOO_NARROW",
    "wide": "TOO_WIDE",
    "width_unstable": "WIDTH_UNSTABLE",
    "curve": "BENT",
}


@dataclass(frozen=True)
class GeneratedSample:
    """一帧二分类合成图像及其像素级标注。"""

    image: np.ndarray
    glue_mask: np.ndarray
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
        """返回中心线上指定点的坐标。"""

        return float(self.xs[index]), float(self.ys[index])

    def normal(self, index: int) -> tuple[float, float]:
        """返回中心线上指定点的单位法向量。"""

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


def _draw_scene(
    rng: np.random.Generator,
    height: int,
    width: int,
    scene_cfg: Mapping[str, Any],
) -> np.ndarray:
    """生成只有背景纹理的浮点图像，不绘制工件板。"""

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

    return image.astype(np.float32)


def _build_glue_geometry(
    rng: np.random.Generator,
    height: int,
    width: int,
    glue_cfg: Mapping[str, Any],
    centerline_points: int,
    *,
    width_scale: float = 1.0,
    amplitude_scale: float = 1.0,
    amplitude_override_px: float | None = None,
    width_variation_scale: float = 1.0,
    bow_mode: bool = False,
) -> _GlueGeometry:
    """在画面背景上生成一条随机宽度、轻微弯曲的胶条。"""

    angle = math.radians(_uniform(rng, glue_cfg["angle_deg"]))
    length = width * _uniform(rng, glue_cfg["length_ratio"])
    half_length = length / 2.0

    ts = np.linspace(-1.0, 1.0, centerline_points)
    local_x = ts * half_length
    amplitude_source = (
        float(amplitude_override_px)
        if amplitude_override_px is not None
        else _uniform(rng, glue_cfg["amplitude_px"]) * amplitude_scale
    )
    amplitude = min(
        amplitude_source,
        max(1.0, height * float(glue_cfg["max_amplitude_ratio"])),
    )
    frequency = _uniform(rng, glue_cfg["wave_count"])
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    slope = _uniform(rng, glue_cfg["slope_ratio"])
    if bow_mode:
        local_y = amplitude * (1.0 - ts * ts) + slope * local_x
    else:
        local_y = amplitude * np.sin(math.pi * frequency * ts + phase) + slope * local_x

    center_x = width * _uniform(rng, glue_cfg["center_x_ratio"])
    center_y = height * _uniform(rng, glue_cfg["center_y_ratio"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xs_center = center_x + cos_a * local_x - sin_a * local_y
    ys_center = center_y + sin_a * local_x + cos_a * local_y

    dx = np.gradient(xs_center)
    dy = np.gradient(ys_center)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    normals_x = -dy / norm
    normals_y = dx / norm

    base_width = _uniform(rng, glue_cfg["width_px"]) * width_scale
    variation = _uniform(rng, glue_cfg["width_variation"]) * width_variation_scale
    width_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    widths = base_width * (
        1.0 + variation * np.sin(math.pi * frequency * ts + width_phase)
    )
    widths = np.maximum(widths, float(glue_cfg["min_width_px"]))

    left = np.stack(
        [xs_center + normals_x * widths / 2.0, ys_center + normals_y * widths / 2.0],
        axis=1,
    )
    right = np.stack(
        [xs_center - normals_x * widths / 2.0, ys_center - normals_y * widths / 2.0],
        axis=1,
    )
    polygon = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    if int((mask > 0).sum()) == 0:
        raise ValueError("胶条掩膜为空")

    return _GlueGeometry(
        xs=xs_center,
        ys=ys_center,
        normals_x=normals_x,
        normals_y=normals_y,
        widths=widths,
        mask=mask,
    )


def _bbox(mask: np.ndarray) -> list[int] | None:
    """返回掩膜外接框，格式与项目 Box 一致。"""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _defect_record(
    defect_type: str,
    mask: np.ndarray,
    **details: Any,
) -> dict[str, Any] | None:
    """构造缺陷元数据；空掩膜不产生记录。"""

    box = _bbox(mask)
    area = int((mask > 0).sum())
    if area == 0 or box is None:
        return None
    return {
        "type": defect_type,
        "area_px": area,
        "box": box,
        **details,
    }


def _cut_break(
    rng: np.random.Generator,
    glue: _GlueGeometry,
    glue_mask: np.ndarray,
    defect_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """从胶条中间切除一段，形成真实断口。"""

    gap_ratio = _uniform(rng, defect_cfg["break_length_ratio"])
    gap_center = float(rng.uniform(-0.2, 0.2))
    start = gap_center - gap_ratio / 2.0
    end = gap_center + gap_ratio / 2.0
    positions = np.linspace(-1.0, 1.0, len(glue.xs))
    indices = np.flatnonzero((positions >= start) & (positions <= end))
    if len(indices) < 2:
        indices = np.array([len(glue.xs) // 2 - 1, len(glue.xs) // 2])

    half_width = glue.widths[indices] / 2.0 + float(
        defect_cfg["break_margin_px"]
    )
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
    break_mask = np.zeros_like(glue_mask)
    polygon = np.concatenate([left, right[::-1]], axis=0)
    cv2.fillPoly(break_mask, [np.round(polygon).astype(np.int32)], 255)
    break_mask &= glue_mask
    glue_mask[break_mask > 0] = 0
    return break_mask, _defect_record(
        "break",
        break_mask,
        gap_ratio=gap_ratio,
    )


def generate_sample(
    settings: Mapping[str, Any],
    sample_seed: int,
) -> GeneratedSample:
    """生成一帧只有背景和胶条两个类别的合成图像。"""

    height, width = (int(v) for v in settings["image_size"])
    rng = np.random.default_rng(sample_seed)
    scenario = _weighted_choice(rng, settings["scenario_weights"])

    image = _draw_scene(
        rng,
        height,
        width,
        settings["scene"],
    )
    glue_mask = np.zeros((height, width), dtype=np.uint8)
    nominal_width_px = 0.0
    defect_records: list[dict[str, Any]] = []

    if scenario != "background":
        width_scale = {
            "narrow": float(settings["variants"]["narrow_width_scale"]),
            "wide": float(settings["variants"]["wide_width_scale"]),
        }.get(scenario, 1.0)
        width_variation_scale = (
            float(settings["variants"]["width_unstable_variation_scale"])
            if scenario == "width_unstable"
            else 1.0
        )
        curve_amplitude = (
            _uniform(rng, settings["variants"]["curve_amplitude_px"])
            if scenario == "curve"
            else None
        )
        glue = _build_glue_geometry(
            rng,
            height,
            width,
            settings["glue"],
            int(settings["centerline_points"]),
            width_scale=width_scale,
            amplitude_override_px=curve_amplitude,
            width_variation_scale=width_variation_scale,
            bow_mode=scenario == "curve",
        )
        glue_mask = glue.mask.copy()
        if scenario == "break":
            _, record = _cut_break(rng, glue, glue_mask, settings["defects"])
            if record is not None:
                defect_records.append(record)

        glue_value = _uniform(rng, settings["glue"]["value_range"])
        texture = rng.normal(
            0.0,
            float(settings["glue"]["texture_sigma"]),
            size=(height, width),
        )
        strip = glue_mask > 0
        image[strip] = glue_value + texture[strip]
        nominal_width_px = float(np.median(glue.widths))
        if scenario in {"narrow", "wide", "width_unstable", "curve"}:
            defect_records.append(
                {
                    "type": scenario,
                    "area_px": int((glue_mask > 0).sum()),
                    "box": _bbox(glue_mask),
                    "nominal_width_px": nominal_width_px,
                    "min_width_px": float(np.min(glue.widths)),
                    "max_width_px": float(np.max(glue.widths)),
                }
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

    return GeneratedSample(
        image=image,
        glue_mask=np.where(glue_mask > 0, 255, 0).astype(np.uint8),
        scenario=scenario,
        label=SCENARIO_LABELS[scenario],
        nominal_width_px=nominal_width_px,
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
    """生成一个可复现的胶条/背景二分类合成数据集。

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

    for relative in ("images", "masks/glue", "labels", "splits"):
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
        label_path = root / "labels" / f"{image_id}.json"

        if not cv2.imwrite(str(image_path), sample.image):
            raise OSError(f"无法写入图像：{image_path}")
        if not cv2.imwrite(str(glue_path), sample.glue_mask):
            raise OSError(f"无法写入胶条掩膜：{glue_path}")

        has_glue = bool((sample.glue_mask > 0).any())
        glue_pixels = int((sample.glue_mask > 0).sum())
        defect_types = ";".join(
            str(record["type"]) for record in sample.defect_records
        )
        label_data = {
            "image": f"images/{image_id}.png",
            "image_size": {
                "height": int(sample.image.shape[0]),
                "width": int(sample.image.shape[1]),
            },
            "source": "synthetic",
            "task": TASK,
            "scene_id": object_id,
            "split": split,
            "scenario": sample.scenario,
            "label": sample.label,
            "has_glue": has_glue,
            "has_defect": bool(sample.defect_records),
            "acceptance_eligible": False,
            "mask": f"masks/glue/{image_id}.png",
            "classes": CLASS_IDS,
            "glue_pixels": glue_pixels,
            "nominal_width_px": sample.nominal_width_px,
            "defects": sample.defect_records,
        }
        label_path.write_text(
            json.dumps(label_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "image_id": image_id,
                "path": f"images/{image_id}.png",
                "mask": f"masks/glue/{image_id}.png",
                "split": split,
                "object_id": object_id,
                "batch": batch,
                "source": "synthetic",
                "scenario": sample.scenario,
                "label": sample.label,
                "has_glue": int(has_glue),
                "has_defect": int(bool(sample.defect_records)),
                "defect_types": defect_types,
                "glue_pixels": glue_pixels,
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

    generation_config = dict(settings)
    generation_config["samples"] = sample_count
    generation_config["seed"] = actual_seed
    dataset_meta = {
        "dataset_version": settings["version"],
        "task": TASK,
        "source": "synthetic",
        "acceptance_eligible": False,
        "warning": "仅用于 VisionTrain 流程验证、冒烟和预训练；不得作为真实指标验收数据。",
        "seed": actual_seed,
        "sample_count": sample_count,
        "frames_per_object": frames_per_object,
        "class_ids": CLASS_IDS,
        "mask_files": {"glue": "masks/glue/<image_id>.png"},
        "defect_types": ["break", "narrow", "wide", "width_unstable", "curve"],
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "scenario_counts": scenario_counts,
        "generation_config": generation_config,
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
