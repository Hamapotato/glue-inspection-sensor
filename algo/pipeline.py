"""旋转扫描矩形胶条检测主线。

流程：

    胶条分割模型
      -> 概率图/掩膜二值化与形态学后处理
      -> 沿胶条方向生成旋转扫描矩形
      -> 每个矩形内寻找左右边缘点
      -> 左右边缘点中点形成中心曲线
      -> 断胶、宽窄不一致、弯曲判定

分割模型通过 ``GlueSegmenter`` 注入；无模型时不会静默切换到其他算法。
"""

from __future__ import annotations

import time

import numpy as np

from .calib import Calibration
from .config import Config
from .decide import decide
from .maskops import connected_components
from .scanline import apply_scan_result, scan_glue_strip
from .segmentation import (
    DevelopmentThresholdSegmenter,
    GlueSegmenter,
    SegmentationInfo,
    build_segmenter,
    postprocess_segmentation,
)
from .types import Box, FrameResult, MeasureResult


class GlueInspector:
    """胶条分割与旋转扫描量测器。"""

    def __init__(
        self,
        cfg: Config,
        segmenter: GlueSegmenter | None,
        calib: Calibration | None = None,
    ):
        self.cfg = cfg
        self.segmenter = build_segmenter(cfg, segmenter)
        self.calib = calib
        self.mm_per_px = self._resolve_mm_per_px(calib)
        self.min_area_px = int(cfg.get("measure.min_area_px", 200))
        self.mask_connectivity = int(cfg.get("measure.mask_connectivity", 8))
        self.degradations: list[str] = []
        if isinstance(self.segmenter, DevelopmentThresholdSegmenter):
            self.degradations.append(
                "分割模型不可用：当前使用开发阈值 fallback，结果不可作为模型验收"
            )
        if self.mm_per_px is None:
            self.degradations.append("未标定：宽度只输出像素，判定跳过毫米宽度项")

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        segmenter: GlueSegmenter | None = None,
        calib: Calibration | None = None,
    ) -> "GlueInspector":
        """按配置构建；仅在显式配置时允许开发阈值 fallback。"""

        return cls(cfg, segmenter=segmenter, calib=calib)

    def run(self, image: np.ndarray) -> FrameResult:
        """对单帧图像执行分割、扫描量测和判定。"""

        timings: dict[str, float] = {}

        started = time.perf_counter()
        model_output = self.segmenter.predict(image)
        segmentation_output = postprocess_segmentation(model_output, self.cfg)
        timings["segment"] = (time.perf_counter() - started) * 1000.0
        mask = segmentation_output.mask

        started = time.perf_counter()
        _, components = connected_components(
            mask,
            connectivity=self.mask_connectivity,
            min_area=self.min_area_px,
        )
        measure = self._build_measure(mask, components)
        scan = scan_glue_strip(mask, self.cfg, mm_per_px=self.mm_per_px)
        apply_scan_result(measure, scan)
        timings["measure"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        verdict = decide(measure, self.cfg)
        timings["decide"] = (time.perf_counter() - started) * 1000.0
        timings["total"] = sum(
            value for key, value in timings.items() if key != "total"
        )

        pixels = int(mask.size)
        foreground_ratio = (
            float(np.count_nonzero(mask)) / float(pixels) if pixels else 0.0
        )
        info = SegmentationInfo(
            source=segmentation_output.source,
            threshold=segmentation_output.threshold,
            fallback=segmentation_output.fallback,
            foreground_ratio=foreground_ratio,
        )

        return FrameResult(
            verdict=verdict,
            measure=measure,
            segmentation=info.as_dict(),
            roi=measure.bbox,
            timings_ms={
                key: round(value, 3) for key, value in timings.items()
            },
            degradations=list(self.degradations),
        )

    @staticmethod
    def _build_measure(
        mask: np.ndarray,
        components: list,
    ) -> MeasureResult:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return MeasureResult()
        return MeasureResult(
            area_px=float(len(xs)),
            n_components=len(components),
            bbox=Box(
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ),
        )

    def _resolve_mm_per_px(self, calib: Calibration | None) -> float | None:
        if calib is not None and calib.mm_per_px:
            return float(calib.mm_per_px)
        value = self.cfg.get("calib.mm_per_px")
        return float(value) if value else None
