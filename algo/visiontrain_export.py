"""将合成掩膜转换为 VisionTrain 可导入的分割标注。

格式依据 VisionTrain 2.3 导出的数据集样例：顶层 ``dataset.json`` 描述图像与标签，
每张图旁的 ``<image_id>.json`` 保存 COCO 风格的多边形标注。该转换器只服务
合成数据流程，输出仍需在 VisionTrain 中做 GUI 回读验证。
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from algo.config import Config

BACKGROUND_CATEGORY_ID = 1
GLUE_CATEGORY_ID = 2
GLUE_LABEL = "glue"
SYNTHETIC_WARNING = "合成数据仅用于流程验证、冒烟和预训练，不得用于真实验收。"
VALID_IMAGE_TYPES = frozenset({"train", "val", "test"})


@dataclass
class VisionTrainExportSummary:
    """VisionTrain 数据导出结果摘要。"""

    output_dir: Path
    image_count: int
    glue_image_count: int
    background_image_count: int
    annotation_count: int


def _prepare_output_dir(output_dir: Path) -> None:
    """创建输出目录，拒绝覆盖已有内容。"""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"输出路径不是目录：{output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    """读取并校验合成数据集 manifest。"""

    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest.csv：{manifest_path}")
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "path", "scenario", "acceptance_eligible"}
    if not rows:
        raise ValueError("manifest.csv 不包含任何样本")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"manifest.csv 缺少字段：{sorted(missing)}")
    if any(row["acceptance_eligible"] != "0" for row in rows):
        raise ValueError("仅允许导出 acceptance_eligible=0 的合成数据")
    return rows


def mask_to_polygons(
    mask: np.ndarray,
    *,
    contour_epsilon_px: float,
    min_contour_area_px: float,
) -> list[dict[str, Any]]:
    """将二值掩膜转为 VisionTrain 使用的多边形记录。

    输入 mask 中所有非零像素均视为 ``glue``。多段胶条会拆成多个对象。
    """

    if mask.ndim != 2:
        raise ValueError("缺陷掩膜必须为单通道灰度图")
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area_px:
            continue
        if len(contour) < 3:
            continue
        if contour_epsilon_px > 0:
            contour = cv2.approxPolyDP(contour, contour_epsilon_px, True)
        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2).astype(float)
        x, y, width, height = cv2.boundingRect(contour.astype(np.int32))
        segmentation = [
            float(value)
            for point in points
            for value in (point[0], point[1])
        ]
        polygons.append(
            {
                "bbox": [float(x), float(y), float(width), float(height)],
                "segmentation": [segmentation],
                "area": float(cv2.contourArea(contour)),
            }
        )
    return polygons


def _annotation_records(
    polygons: list[dict[str, Any]],
    image_name: str,
) -> list[dict[str, Any]]:
    """按 VisionTrain 导出样例组装单图标注记录。"""

    records: list[dict[str, Any]] = []
    for annotation_id, polygon in enumerate(polygons, start=1):
        records.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": GLUE_CATEGORY_ID,
                "bbox": polygon["bbox"],
                "segmentation": polygon["segmentation"],
                "area": polygon["area"],
                "iscrowd": 0,
                "type": "polygon",
                "labelname": GLUE_LABEL,
                "image_name": image_name,
            }
        )
    return records


def _label_lists() -> list[dict[str, Any]]:
    """返回胶条与背景二分类的 VisionTrain 标签配置。"""

    return [
        {
            "_LabelSliderValue": 0.5,
            "_IsMarkTotalNumber": False,
            "id": 1,
            "LabelName": "background",
            "LabelColor": "#FF73F40B",
            "IsLabelFilter": False,
            "IsPreLabelFilter": False,
            "LabelColorValue": 999,
            "LabelOpacity": 0.2,
            "LabelShortcutKey": "1",
            "IsDisplayLabel": True,
            "IsDefaultLabel": False,
            "IsSystemLabel": True,
            "IsButtonLabel": True,
            "IsVisible": True,
            "LabelSliderValue": 0.5,
            "MarkTotalNumber": 0,
            "IsMarkTotalNumber": False,
            "NumsOfPicture": 0,
        },
        {
            "_LabelSliderValue": 0.5,
            "_IsMarkTotalNumber": False,
            "id": GLUE_CATEGORY_ID,
            "LabelName": GLUE_LABEL,
            "LabelColor": "#FF18C964",
            "IsLabelFilter": False,
            "IsPreLabelFilter": False,
            "LabelColorValue": 1,
            "LabelOpacity": 0.2,
            "LabelShortcutKey": "2",
            "IsDisplayLabel": True,
            "IsDefaultLabel": False,
            "IsSystemLabel": False,
            "IsButtonLabel": False,
            "IsVisible": True,
            "LabelSliderValue": 0.5,
            "MarkTotalNumber": 0,
            "IsMarkTotalNumber": False,
            "NumsOfPicture": 0,
        },
    ]


def _dataset_metadata(
    images: list[dict[str, Any]],
    *,
    created_at: str | None,
) -> dict[str, Any]:
    """组装 VisionTrain dataset.json。"""

    timestamp = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "categories": [
            {
                "id": BACKGROUND_CATEGORY_ID,
                "name": "background",
                "color_value": 255,
                "label_color": "#FF73F40B",
            },
            {
                "id": GLUE_CATEGORY_ID,
                "name": GLUE_LABEL,
                "color_value": 1,
                "label_color": "#FF18C964",
            },
        ],
        "images": images,
        "info": {
            "calib_type": 3,
            "contributor": "vision_train_2.3.2",
            "date_created": timestamp,
            "url": "MVBTORCH_TRAIN_DATA\\V4.0",
            "version": "1.1",
            "year": 2026,
            "source": "synthetic",
            "acceptance_eligible": False,
            "warning": SYNTHETIC_WARNING,
        },
        "licenses": [{"id": 1, "name": "MVB License"}],
        "LabelLists": _label_lists(),
        "source": "synthetic",
        "acceptance_eligible": False,
        "warning": SYNTHETIC_WARNING,
    }


def _resolve_image_type(row: dict[str, str], default: str) -> str:
    """优先保留 manifest 中的分组，缺失时使用配置默认值。"""

    split = row.get("split", "").strip()
    if not split:
        return default
    if split not in VALID_IMAGE_TYPES:
        raise ValueError(f"不支持的 VisionTrain 数据分组：{split}")
    return split


def export_visiontrain_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    cfg: Config,
    *,
    image_ids: list[str] | None = None,
    created_at: str | None = None,
) -> VisionTrainExportSummary:
    """把合成数据集转换为 VisionTrain 图像管理可导入的数据目录。"""

    source_dir = Path(dataset_dir)
    target_dir = Path(output_dir)
    rows = _read_manifest(source_dir)
    if image_ids is not None:
        requested = set(image_ids)
        rows_by_id = {row["image_id"]: row for row in rows}
        missing = requested.difference(rows_by_id)
        if missing:
            raise ValueError(f"manifest 中不存在指定样本：{sorted(missing)}")
        rows = [row for row in rows if row["image_id"] in requested]
        if not rows:
            raise ValueError("至少需要指定一张图像")
    _prepare_output_dir(target_dir)

    default_image_type = str(cfg.get("visiontrain_export.image_type", "train"))
    epsilon = float(cfg.get("visiontrain_export.contour_epsilon_px", 0.5))
    min_area = float(cfg.get("visiontrain_export.min_contour_area_px", 1.0))

    images: list[dict[str, Any]] = []
    annotation_count = 0
    glue_image_count = 0

    for image_id, row in enumerate(rows, start=1):
        image_type = _resolve_image_type(row, default_image_type)
        image_name = f"{row['image_id']}.png"
        source_image = source_dir / row["path"]
        source_mask = source_dir / "masks" / "glue" / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"缺少图像：{source_image}")
        if not source_mask.is_file():
            raise FileNotFoundError(f"缺少胶条掩膜：{source_mask}")

        image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"无法读取图像或掩膜：{row['image_id']}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"图像与掩膜尺寸不一致：{row['image_id']}")

        polygons = mask_to_polygons(
            mask,
            contour_epsilon_px=epsilon,
            min_contour_area_px=min_area,
        )
        records = _annotation_records(polygons, image_name)
        if records:
            glue_image_count += 1
        annotation_count += len(records)

        shutil.copy2(source_image, target_dir / image_name)
        with open(
            target_dir / f"{row['image_id']}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "height": int(height),
                "width": int(width),
                "image_type": image_type,
                "labeled": True,
                "Marked": False,
                "license": 1,
            }
        )

    metadata = _dataset_metadata(images, created_at=created_at)
    with open(target_dir / "dataset.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return VisionTrainExportSummary(
        output_dir=target_dir,
        image_count=len(images),
        glue_image_count=glue_image_count,
        background_image_count=len(images) - glue_image_count,
        annotation_count=annotation_count,
    )
