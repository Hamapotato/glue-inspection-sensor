"""胶条分割模型接口与输出后处理。

当前主线要求由外部胶条/背景分割模型提供概率图或二值掩膜。本模块不训练模型，
只定义稳定接口、二值化和形态学后处理。没有模型时，除显式启用开发 fallback
外不会静默改用其他检测算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import cv2
import numpy as np

from .config import Config


class ModelNotAvailableError(RuntimeError):
    """分割模型不可用。"""


@dataclass
class SegmentationOutput:
    """分割模型的一次输出。"""

    mask: np.ndarray
    source: str
    threshold: float = 0.5
    fallback: bool = False
    score_map: np.ndarray | None = None


@dataclass
class SegmentationInfo:
    """写入检测结果的去敏分割摘要。"""

    source: str
    threshold: float
    fallback: bool
    foreground_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "threshold": self.threshold,
            "fallback": self.fallback,
            "foreground_ratio": self.foreground_ratio,
        }


@runtime_checkable
class GlueSegmenter(Protocol):
    """胶条/背景分割模型接口。"""

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        """返回与输入图像同尺寸的分割掩膜或概率图。"""


class MaskSegmenter:
    """返回预计算掩膜的分割器，用于离线结果、测试和坐标对齐校验。"""

    def __init__(
        self,
        mask: np.ndarray,
        *,
        source: str = "provided_mask",
    ):
        self.mask = np.asarray(mask)
        self.source = source

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        if self.mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"掩膜尺寸 {self.mask.shape[:2]} 与图像尺寸 {image.shape[:2]} 不一致"
            )
        return SegmentationOutput(
            mask=self.mask.copy(),
            source=self.source,
            threshold=0.5,
        )


class CallableSegmenter:
    """把外部模型推理函数适配为 ``GlueSegmenter``。"""

    def __init__(
        self,
        predictor: Callable[[np.ndarray], SegmentationOutput | np.ndarray],
        *,
        source: str = "external_model",
    ):
        self.predictor = predictor
        self.source = source

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        output = self.predictor(image)
        if isinstance(output, SegmentationOutput):
            return output
        return SegmentationOutput(
            mask=np.asarray(output),
            source=self.source,
            threshold=0.5,
        )


class DevelopmentThresholdSegmenter:
    """仅用于无模型环境冒烟的开发 fallback。

    该实现不属于当前产品主线，使用时必须在结果 ``degradations`` 中记录。
    """

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return SegmentationOutput(
            mask=mask,
            source="development_threshold_fallback",
            threshold=0.5,
            fallback=True,
        )


def build_segmenter(
    cfg: Config,
    segmenter: GlueSegmenter | None,
) -> GlueSegmenter:
    """返回调用方注入的模型；没有模型时只允许显式启用开发 fallback。"""

    if segmenter is not None:
        return segmenter
    mode = str(cfg.get("segmentation.mode", "external"))
    if mode == "development_threshold":
        return DevelopmentThresholdSegmenter()
    model_path = cfg.get("segmentation.model_path")
    raise ModelNotAvailableError(
        "主线需要胶条分割模型。请注入 GlueSegmenter，"
        "或在明确允许的开发环境中设置 segmentation.mode=development_threshold。"
        f" 当前 model_path={model_path!r}"
    )


def postprocess_segmentation(
    output: SegmentationOutput,
    cfg: Config,
) -> SegmentationOutput:
    """把模型概率图或掩膜统一为 0/255 二值图并做形态学后处理。"""

    array = np.asarray(output.mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"分割输出必须是二维掩膜，实际形状为 {array.shape}")

    threshold = float(cfg.get("segmentation.threshold", output.threshold))
    values = array.astype(np.float32)
    max_value = float(values.max()) if values.size else 0.0
    normalized = values / 255.0 if max_value > 1.0 else values
    binary = np.where(normalized >= threshold, 255, 0).astype(np.uint8)

    ksize = int(cfg.get("segmentation.morph.ksize", 5))
    order = str(cfg.get("segmentation.morph.order", "open-close"))
    if ksize > 1:
        binary = _morph_open_close(binary, ksize, order)

    return SegmentationOutput(
        mask=binary,
        source=output.source,
        threshold=threshold,
        fallback=output.fallback,
        score_map=output.score_map,
    )


def _morph_open_close(mask: np.ndarray, ksize: int, order: str) -> np.ndarray:
    operations = {
        "open": (cv2.MORPH_OPEN,),
        "close": (cv2.MORPH_CLOSE,),
        "open-close": (cv2.MORPH_OPEN, cv2.MORPH_CLOSE),
        "close-open": (cv2.MORPH_CLOSE, cv2.MORPH_OPEN),
    }
    if order not in operations:
        raise ValueError(f"未知形态学顺序 {order}，可选 {list(operations)}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    result = mask
    for operation in operations[order]:
        result = cv2.morphologyEx(result, operation, kernel)
    return result
