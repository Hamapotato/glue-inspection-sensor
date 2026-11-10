"""检测模型接口与占位实现。

本模块只定义"谁来给出 ROI / 反光框"的契约，以及在没有训练权重时仍能跑通链路的占位实现。
真正的 RPN（论文 3.1）与 YOLOv3-5FL+SPP（论文 3.3）在 P3/P4 阶段实现，任务号 B-03、B-08~B-10。

约定见 docs/algo/interfaces.md §3。
"""

from __future__ import annotations

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
    """RPN 定位（论文 3.1）：加载 ``algo.net.rpn`` 的权重做推理。

    预处理与训练一致：灰度 → letterbox 到 ``locate.input_size`` 的正方形 →
    前向 → decode → NMS/CIoU-NMS → 映射回原图坐标。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.weights = cfg.get("locate.weights")
        if not self.weights:
            raise ModelNotAvailableError("locate.weights 未配置")
        self.input_size = int(cfg.get("locate.input_size", 320))
        self.nms_type = str(cfg.get("locate.nms.type", "ciou_nms"))
        self.nms_thr = float(cfg.get("locate.nms.thr", 0.5))
        self.score_thr = float(cfg.get("locate.score_thr", 0.5))
        self.max_boxes = int(cfg.get("locate.post_nms_topk", 300))
        self.device = str(cfg.get("runtime.device", "cpu"))
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch  # 可选依赖，延迟导入

        from .net.rpn import RPN

        ckpt = torch.load(self.weights, map_location="cpu")
        model = RPN()
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model
        return model

    def predict(self, image: np.ndarray) -> list[Box]:
        import torch

        from .data import letterbox
        from .net.rpn import ProposalConfig, propose

        model = self._ensure_model()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas, scale, pad = letterbox(gray, (self.input_size, self.input_size))
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cls_logits, reg, anchors, _ = model(tensor)
            boxes, scores = propose(
                cls_logits[0], reg[0], anchors, (self.input_size, self.input_size),
                ProposalConfig(post_nms_topk=self.max_boxes, nms_thr=self.nms_thr,
                               nms_type=self.nms_type, score_thr=self.score_thr),
            )
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        if len(boxes):
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
        return [
            Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), score=float(s))
            for b, s in zip(boxes, scores)
        ]


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
    """YOLOv3-5FL + SPP 反光检测（论文 3.3）——待实现。

    实现要点（P4）：
    - Darknet-53 + FPN，输出 5 个尺度（3.3.2，图 3-20）；
    - SPP 采用 4x4 / 2x2 / 1x1 池化（3.3.3，图 3-21）；
    - 只判"是否存在反光 + 定位"，不做修复（3.3.4、6.2）。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.weights = cfg.get("reflect.weights")
        if not self.weights:
            raise ModelNotAvailableError("reflect.weights 未配置")
        self.input_size = int(cfg.get("reflect.input_size", 320))
        self.score_thr = float(cfg.get("reflect.score_thr", 0.5))
        self.device = str(cfg.get("runtime.device", "cpu"))
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch  # 可选依赖，延迟导入

        from .net.yolo import YOLO5FL

        ckpt = torch.load(self.weights, map_location="cpu")
        model = YOLO5FL(num_levels=int(ckpt.get("num_levels", 5)), use_spp=bool(ckpt.get("use_spp", True)))
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model
        return model

    def predict(self, image: np.ndarray) -> ReflectionResult:
        import torch

        from .data import letterbox
        from .net.common import nms_torch
        from .net.yolo import decode_yolo

        model = self._ensure_model()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas, scale, pad = letterbox(gray, (self.input_size, self.input_size))
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            decoded = decode_yolo(outputs, (self.input_size, self.input_size),
                                  anchors=model.anchors, conf_thr=self.score_thr)
        boxes, scores = decoded[0]
        if boxes.numel():
            keep = nms_torch(boxes, scores, 0.5)
            boxes, scores = boxes[keep], scores[keep]
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        if len(boxes_np):
            boxes_np[:, [0, 2]] = (boxes_np[:, [0, 2]] - pad[0]) / scale
            boxes_np[:, [1, 3]] = (boxes_np[:, [1, 3]] - pad[1]) / scale
        out = [
            Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), score=float(s))
            for b, s in zip(boxes_np, scores_np)
        ]
        return ReflectionResult(has_reflection=bool(out), boxes=out,
                                max_score=float(scores_np.max()) if scores_np.size else 0.0)


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
