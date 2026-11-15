"""检测模型接口与占位实现。

本模块只定义"谁来给出 ROI / 反光框"的契约，以及在没有训练权重时仍能跑通链路的占位实现。
真正的 RPN（论文 3.1）与 YOLOv3-5FL+SPP（论文 3.3）在 P3/P4 阶段实现，任务号 B-03、B-08~B-10。

约定见 docs/algo/interfaces.md §3。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import Config
from .measure import connected_components
from .preprocess import morph_open_close, otsu_threshold
from .types import Box, ReflectionResult


class ModelNotAvailableError(RuntimeError):
    """模型权重缺失或模型尚未实现。"""


@runtime_checkable
class RoILocator(Protocol):
    """胶条 RoI 定位（论文 3.1）。"""

    def predict(self, image: np.ndarray) -> list[Box]:
        """返回按 score 降序的候选框；无目标时返回空列表。"""
        ...


@runtime_checkable
class ReflectionDetector(Protocol):
    """反光干扰检测（论文 3.3）。"""

    def predict(self, image: np.ndarray) -> ReflectionResult:
        """返回反光区域与是否存在的结论。"""
        ...


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放 + 居中填充到 size×size，返回 (图像, 缩放比, 左上填充)。"""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, (px, py)


def _unletterbox_boxes(boxes: np.ndarray, scale: float, pad: tuple[int, int]) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / max(scale, 1e-9)
    out[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / max(scale, 1e-9)
    return out


# --------------------------------------------------------------------------- #
# RoI 定位
# --------------------------------------------------------------------------- #

class HeuristicRoILocator:
    """占位实现：阈值分割后取连通域外接框。

    只用于在没有 RPN 权重时跑通链路，**不代表论文方法**；其输出参与评测时必须在
    报告中标注为"占位实现"。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_area = int(cfg.get("measure.min_area_px", 200))
        self.morph_ksize = int(cfg.get("preprocess.morph.ksize", 5))

    def predict(self, image: np.ndarray) -> list[Box]:
        _, mask = otsu_threshold(image)
        mask = morph_open_close(mask, self.morph_ksize)
        _, comps = connected_components(mask, connectivity=8, min_area=self.min_area)
        comps.sort(key=lambda c: -c.area)
        boxes = [
            Box(float(c.bbox[0]), float(c.bbox[1]), float(c.bbox[2]), float(c.bbox[3]), score=1.0)
            for c in comps
        ]
        return boxes


class RPNLocator:
    """RPN 定位（论文 3.1），加载 ``locate.weights`` 指向的权重。

    - 结构：4 层 stride=2 卷积（stride 16）→ 3×3 → 分类/回归双头（见 ``algo.net.rpn``）；
    - 先验框：默认 9 个，可用 ``anchors_from_kmeans`` 由训练集统计；
    - 回归损失：DIoU（论文 3.1.3）；
    - 候选筛选：CIoU-NMS，阈值 0.5（论文 3.1.4、表 3-2）。

    torch 与权重缺一不可，缺失时抛 ``ModelNotAvailableError``，由工厂降级。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        weights = cfg.get("locate.weights")
        if not weights:
            raise ModelNotAvailableError("locate.weights 未配置")
        path = Path(str(weights))
        if not path.exists():
            raise ModelNotAvailableError(f"locate.weights 文件不存在：{path}")
        try:
            import torch  # noqa: F401

            from .net.rpn import RPN, ProposalConfig
        except Exception as exc:  # pragma: no cover - 环境相关
            raise ModelNotAvailableError(f"torch 或 algo.net 不可用（{exc}）") from exc

        self.model = RPN.load(path)
        self.input_size = int(cfg.get("locate.input_size", 320))
        self.score_thr = float(cfg.get("locate.score_thr", 0.5))
        self.max_rois = int(cfg.get("locate.max_rois", 10))
        self.proposal_cfg = ProposalConfig(
            nms_type=str(cfg.get("locate.nms.type", "ciou_nms")),
            nms_thr=float(cfg.get("locate.nms.thr", 0.5)),
        )

    def predict(self, image: np.ndarray) -> list[Box]:
        import torch

        from .net.rpn import propose

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas, scale, pad = _letterbox(gray, self.input_size)
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None]
        with torch.no_grad():
            cls_logits, reg, anchors, _ = self.model(tensor)
            boxes, scores = propose(cls_logits[0], reg[0], anchors,
                                    (self.input_size, self.input_size), self.proposal_cfg)
        if boxes.numel() == 0:
            return []
        np_boxes = _unletterbox_boxes(boxes.cpu().numpy().astype(np.float64), scale, pad)
        np_scores = scores.cpu().numpy().astype(np.float64)
        h, w = gray.shape[:2]
        out: list[Box] = []
        for (x1, y1, x2, y2), score in zip(np_boxes, np_scores):
            if score < self.score_thr:
                continue
            x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
            y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            out.append(Box(x1, y1, x2, y2, score=float(score)))
            if len(out) >= self.max_rois:
                break
        out.sort(key=lambda b: -b.score)
        return out


# --------------------------------------------------------------------------- #
# 反光检测
# --------------------------------------------------------------------------- #

class NullReflector:
    """空实现：永远判为无反光。用于反光模型缺失时的降级。"""

    def predict(self, image: np.ndarray) -> ReflectionResult:
        return ReflectionResult(has_reflection=False, boxes=[], max_score=0.0)


class HeuristicReflector:
    """占位实现：高亮饱和区域即视为反光。

    反光在灰度图上通常表现为接近饱和的高亮块（论文 3.3 的动机），因此在没有
    YOLO 权重时可用阈值法近似。**不代表论文方法**。
    """

    def __init__(self, cfg: Config):
        self.high_value = int(cfg.get("reflect.high_value", 245))
        self.min_area = int(cfg.get("reflect.min_area_px", 30))
        self.score_thr = float(cfg.get("reflect.score_thr", 0.5))

    def predict(self, image: np.ndarray) -> ReflectionResult:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray >= self.high_value).astype(np.uint8) * 255
        _, comps = connected_components(mask, connectivity=8, min_area=self.min_area)
        boxes = [
            Box(float(c.bbox[0]), float(c.bbox[1]), float(c.bbox[2]), float(c.bbox[3]), score=1.0)
            for c in comps
        ]
        boxes.sort(key=lambda b: -b.area)
        return ReflectionResult(has_reflection=bool(boxes), boxes=boxes, max_score=1.0 if boxes else 0.0)


class YOLO5FLDetector:
    """YOLOv3-5FL + SPP 反光检测（论文 3.3），加载 ``reflect.weights``。

    - 结构：5 个输出尺度 + SPP（见 ``algo.net.yolo``），与论文 3.3.2/3.3.3 对应；
    - 只判"是否存在反光 + 定位"，不做修复（论文 3.3.4、6.2）；
    - 推理含 NMS，阈值来自 ``reflect.score_thr`` 与配置。

    torch 与权重缺一不可，缺失时抛 ``ModelNotAvailableError``，由工厂降级。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        weights = cfg.get("reflect.weights")
        if not weights:
            raise ModelNotAvailableError("reflect.weights 未配置")
        path = Path(str(weights))
        if not path.exists():
            raise ModelNotAvailableError(f"reflect.weights 文件不存在：{path}")
        try:
            import torch  # noqa: F401

            from .net.yolo import YOLO5FL
        except Exception as exc:  # pragma: no cover - 环境相关
            raise ModelNotAvailableError(f"torch 或 algo.net 不可用（{exc}）") from exc

        self.model = YOLO5FL.load(path)
        self.input_size = int(cfg.get("reflect.input_size", 416))
        self.score_thr = float(cfg.get("reflect.score_thr", 0.5))
        self.max_det = int(cfg.get("reflect.max_det", 20))

    def predict(self, image: np.ndarray) -> ReflectionResult:
        import torch

        from .net.yolo import detect

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas, scale, pad = _letterbox(gray, self.input_size)
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None]
        result = detect(self.model, tensor, conf_thr=self.score_thr,
                        nms_thr=float(self.cfg.get("reflect.nms_thr", 0.5)),
                        max_det=self.max_det)[0]
        boxes_np = result["boxes"].cpu().numpy().astype(np.float64)
        scores_np = result["scores"].cpu().numpy().astype(np.float64)
        h, w = gray.shape[:2]
        boxes: list[Box] = []
        for (x1, y1, x2, y2), score in zip(boxes_np, scores_np):
            x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
            y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append(Box(x1, y1, x2, y2, score=float(score)))
        boxes.sort(key=lambda b: -b.score)
        max_score = max((b.score for b in boxes), default=0.0)
        return ReflectionResult(has_reflection=bool(boxes), boxes=boxes, max_score=float(max_score))


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #

def build_roi_locator(cfg: Config) -> tuple[RoILocator, str | None]:
    """按配置构建定位器；返回 (locator, 降级说明)。"""
    kind = str(cfg.get("locate.type", "heuristic"))
    if kind == "rpn":
        try:
            return RPNLocator(cfg), None
        except ModelNotAvailableError as exc:
            return HeuristicRoILocator(cfg), f"RPN 不可用（{exc}），降级为 heuristic"
    return HeuristicRoILocator(cfg), None


def build_reflector(cfg: Config) -> tuple[ReflectionDetector, str | None]:
    """按配置构建反光检测器；返回 (detector, 降级说明)。"""
    kind = str(cfg.get("reflect.type", "null_reflector"))
    if kind == "yolo5fl":
        try:
            return YOLO5FLDetector(cfg), None
        except ModelNotAvailableError as exc:
            return NullReflector(), f"YOLO 反光模型不可用（{exc}），降级为 null"
    if kind == "heuristic":
        return HeuristicReflector(cfg), None
    return NullReflector(), None
