"""数据集与数据加载（B1 RoI / B2 反光共用）。

两种来源
--------
1. **真实数据**：``data/manifest.csv`` + ``data/labels/{roi,reflection}/<image_id>.json``
   （格式见 ``data/README.md``）。划分必须按 ``object_id``/``batch``，不得按帧随机。
2. **合成数据**：``algo.synth`` 生成，用于冒烟与单元测试（不作为验收依据）。

注意：本模块依赖 torch，属于可选依赖（AGENTS.md §4.1）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .synth import generate_case


def read_manifest(path: str | Path) -> list[dict]:
    """读取 manifest.csv。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_boxes(path: str | Path) -> np.ndarray:
    """读取标注 JSON：``{"boxes": [[x1,y1,x2,y2], ...]}``。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    boxes = data.get("boxes", [])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放 + 居中填充到目标尺寸，返回 (图像, 缩放比, 左上填充)。"""
    h, w = image.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw), 114, dtype=image.dtype)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def scale_boxes(boxes: np.ndarray, scale: float, pad: tuple[int, int]) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad[0]
    out[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad[1]
    return out


def split_by_object(records: Sequence[dict], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0) -> dict[str, list[dict]]:
    """按 object_id 分组划分，避免同一条胶条跨集合泄漏。"""
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(str(row.get("object_id", row.get("image_id"))), []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    splits = {
        "train": [r for k in keys[:n_train] for r in groups[k]],
        "val": [r for k in keys[n_train:n_train + n_val] for r in groups[k]],
        "test": [r for k in keys[n_train + n_val:] for r in groups[k]],
    }
    return splits


class GlueDataset(Dataset):
    """真实数据：从 manifest + 标注目录构造检测样本。"""

    def __init__(
        self,
        records: Sequence[dict],
        image_root: str | Path,
        label_dir: str | Path,
        img_size: tuple[int, int] = (320, 320),
    ):
        self.records = list(records)
        self.image_root = Path(image_root)
        self.label_dir = Path(label_dir)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        image_id = str(row.get("image_id", ""))
        image_path = self.image_root / str(row.get("path", image_id + ".png"))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"读不到图像：{image_path}")
        label_path = self.label_dir / f"{image_id}.json"
        boxes = load_boxes(label_path) if label_path.exists() else np.zeros((0, 4), np.float32)

        canvas, scale, pad = letterbox(image, self.img_size)
        boxes = scale_boxes(boxes, scale, pad) if boxes.size else boxes
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return {"image": tensor, "boxes": torch.from_numpy(boxes), "image_id": image_id}


class SyntheticGlueDataset(Dataset):
    """合成数据：RoI 任务生成胶条框；反光任务生成反光框。"""

    def __init__(
        self,
        task: str = "roi",
        img_size: tuple[int, int] = (320, 320),
        length: int = 64,
        seed: int = 0,
    ):
        if task not in {"roi", "reflection"}:
            raise ValueError("task 只支持 roi / reflection")
        self.task = task
        self.img_size = img_size
        self.length = length
        self.seed = seed
        self.kinds = (
            ["normal", "narrow", "wide", "broken"]
            if task == "roi"
            else ["normal", "narrow", "wide", "broken", "reflection"]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        kind = self.kinds[idx % len(self.kinds)]
        case = generate_case(kind, size=(320, 480), width_px=40, seed=self.seed + idx)
        canvas, scale, pad = letterbox(case.image, self.img_size)
        if self.task == "roi":
            boxes = np.asarray([case.box], dtype=np.float32)
        else:
            boxes = np.asarray(case.reflection_boxes, dtype=np.float32).reshape(-1, 4)
        boxes = scale_boxes(boxes, scale, pad) if boxes.size else np.zeros((0, 4), np.float32)
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "image": tensor,
            "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
            "image_id": f"synthetic::{kind}::{idx}",
        }


def collate_one(batch: list[dict]) -> dict:
    """batch=1 的整理函数（B1/B2 训练均按单帧实现）。"""
    if len(batch) != 1:
        raise ValueError("当前实现仅支持 batch=1")
    return batch[0]
