"""当前旋转扫描量测主线的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Box:
    """轴对齐框，像素坐标，x1<=x2, y1<=y2。"""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: int = 0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def clip(self, w: int, h: int) -> "Box":
        return Box(
            min(max(self.x1, 0), w - 1),
            min(max(self.y1, 0), h - 1),
            min(max(self.x2, 0), w - 1),
            min(max(self.y2, 0), h - 1),
            self.score,
            self.label,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanWindow:
    """沿胶条方向生成的一个旋转扫描矩形。"""

    index: int
    cx: float
    cy: float
    angle_deg: float
    length_px: float
    window_width_px: float = 0.0
    width_px: float = 0.0
    left: tuple[float, float] | None = None
    right: tuple[float, float] | None = None
    midpoint: tuple[float, float] | None = None
    valid: bool = False
    break_detected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MeasureResult:
    """一条胶条的扫描量测结果。"""

    area_px: float = 0.0
    width_px: float = 0.0
    width_mm: float = 0.0
    n_components: int = 0
    bbox: Box | None = None
    min_width_px: float = 0.0
    max_width_px: float = 0.0
    width_std_px: float = 0.0
    min_width_mm: float = 0.0
    max_width_mm: float = 0.0
    width_std_mm: float = 0.0
    curve_deviation_px: float = 0.0
    curve_deviation_mm: float = 0.0
    break_windows: int = 0
    missing_edge_windows: int = 0
    scan_windows: list[ScanWindow] = field(default_factory=list)
    centerline_segments: list[list[tuple[float, float]]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.bbox is not None:
            d["bbox"] = self.bbox.as_dict()
        d["scan_windows"] = [window.as_dict() for window in self.scan_windows]
        d["centerline_segments"] = [
            [[float(x), float(y)] for x, y in segment]
            for segment in self.centerline_segments
        ]
        return d


@dataclass
class Verdict:
    """缺陷判定结论。"""

    code: str = "OK"
    severity: str = "none"  # none / warn / alarm
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    """单帧完整输出。"""

    verdict: Verdict
    measure: MeasureResult
    segmentation: dict[str, Any]
    roi: Box | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.as_dict(),
            "measure": self.measure.as_dict(),
            "segmentation": dict(self.segmentation),
            "roi": None if self.roi is None else self.roi.as_dict(),
            "timings_ms": dict(self.timings_ms),
            "degradations": list(self.degradations),
        }
