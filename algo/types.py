"""核心数据结构。"""

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
class MeasureResult:
    """一条胶条的几何量测结果（论文 4.4）。"""

    area_px: float = 0.0
    length_px: float = 0.0
    width_px: float = 0.0
    width_mm: float = 0.0
    n_components: int = 0
    bbox: Box | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.bbox is not None:
            d["bbox"] = self.bbox.as_dict()
        return d


@dataclass
class ReflectionResult:
    """反光检测结果（论文 3.3）。"""

    has_reflection: bool = False
    boxes: list[Box] = field(default_factory=list)
    max_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_reflection": self.has_reflection,
            "boxes": [b.as_dict() for b in self.boxes],
            "max_score": self.max_score,
        }


@dataclass
class Verdict:
    """缺陷判定结论（论文 5.3.3）。"""

    # OK / BROKEN / TOO_WIDE / TOO_NARROW / WARN_WIDE / WARN_NARROW / REFLECTION_SUSPECT
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
    reflection: ReflectionResult
    roi: Box | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.as_dict(),
            "measure": self.measure.as_dict(),
            "reflection": self.reflection.as_dict(),
            "roi": None if self.roi is None else self.roi.as_dict(),
            "timings_ms": dict(self.timings_ms),
        }
