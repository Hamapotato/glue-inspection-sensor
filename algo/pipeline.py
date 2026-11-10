"""单帧检测流水线。

流程（对应论文第 5 章 5.2.3 的图像处理，以及第 4 章的量测链路）：

    RoI 定位 → 裁剪 → 去噪 → 分割 → 形态学 → 面积/骨架/宽度 → 反光检测 → 判定

设计约束（见 docs/algo/interfaces.md）：
- 缺标定 / 缺模型时降级，并把降级原因写进 ``FrameResult.degradations``（不静默）；
- 所有阈值来自配置，代码不硬编码。
"""

from __future__ import annotations

import time

import numpy as np

from .calib import Calibration
from .config import Config
from .decide import decide
from .measure import measure_mask
from .models import ReflectionDetector, RoILocator, build_reflector, build_roi_locator
from .preprocess import median_denoise, morph_open_close, segment, wavelet_denoise
from .types import Box, FrameResult


class GlueInspector:
    """胶条检测器：把各模块串成单帧流水线。"""

    def __init__(
        self,
        cfg: Config,
        locator: RoILocator | None = None,
        reflector: ReflectionDetector | None = None,
        calib: Calibration | None = None,
    ):
        self.cfg = cfg
        self.degradations: list[str] = []

        if locator is None:
            locator, note = build_roi_locator(cfg)
            if note:
                self.degradations.append(note)
        self.locator = locator

        if reflector is None:
            reflector, note = build_reflector(cfg)
            if note:
                self.degradations.append(note)
        self.reflector = reflector

        self.calib = calib
        self.mm_per_px = self._resolve_mm_per_px(calib)
        self.min_area_px = int(cfg.get("measure.min_area_px", 200))
        self.morph_ksize = int(cfg.get("preprocess.morph.ksize", 5))
        self.morph_order = str(cfg.get("preprocess.morph.order", "open-close"))
        self.denoise_method = str(cfg.get("preprocess.denoise.method", "wavelet"))
        self.wavelet_level = int(cfg.get("preprocess.denoise.level", 2))
        self.wavelet_mode = str(cfg.get("preprocess.denoise.mode", "hard"))
        self.segment_method = str(cfg.get("preprocess.segment.method", "otsu"))
        self.bfs_connectivity = int(cfg.get("measure.bfs_connectivity", 8))

    def _resolve_mm_per_px(self, calib: Calibration | None) -> float | None:
        if calib is not None and calib.mm_per_px:
            return float(calib.mm_per_px)
        cfg_value = self.cfg.get("calib.mm_per_px")
        if cfg_value:
            return float(cfg_value)
        return None

    def run(self, image: np.ndarray) -> FrameResult:
        """对单帧图像执行完整检测，返回结构化的 FrameResult。"""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        boxes = self.locator.predict(image)
        timings["locate"] = (time.perf_counter() - t0) * 1000.0

        n_rois = len(boxes)
        roi: Box | None = boxes[0] if boxes else None
        crop = self._crop(image, roi)

        t0 = time.perf_counter()
        denoised = self._denoise(crop)
        timings["denoise"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        _, mask = segment(denoised, self.segment_method)
        mask = morph_open_close(mask, self.morph_ksize, self.morph_order)
        timings["segment"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        measure = measure_mask(
            mask,
            seeds=None,
            connectivity=self.bfs_connectivity,
            min_area_px=self.min_area_px,
            mm_per_px=self.mm_per_px,
        )
        timings["measure"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        reflection = self.reflector.predict(image)
        timings["reflect"] = (time.perf_counter() - t0) * 1000.0

        verdict = decide(measure, n_rois, reflection, self.cfg)
        timings["total"] = sum(v for k, v in timings.items() if k != "total")

        degradations = list(self.degradations)
        if self.mm_per_px is None:
            degradations.append("未标定：宽度只输出像素，判定跳过宽度项")

        return FrameResult(
            verdict=verdict,
            measure=measure,
            reflection=reflection,
            roi=roi,
            timings_ms={k: round(v, 3) for k, v in timings.items()},
            degradations=degradations,
        )

    # ------------------------------------------------------------------ #

    def _crop(self, image: np.ndarray, roi: Box | None) -> np.ndarray:
        if roi is None:
            return image
        h, w = image.shape[:2]
        x1 = max(0, int(np.floor(roi.x1)))
        y1 = max(0, int(np.floor(roi.y1)))
        x2 = min(w, int(np.ceil(roi.x2)))
        y2 = min(h, int(np.ceil(roi.y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return image
        return image[y1:y2, x1:x2]

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        if self.denoise_method == "wavelet":
            return wavelet_denoise(image, level=self.wavelet_level, mode=self.wavelet_mode)
        if self.denoise_method == "median":
            return median_denoise(image, ksize=5)
        return image
