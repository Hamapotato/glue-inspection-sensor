"""视觉传感器快速标定（论文 4.3）。

- 4.3.1 相机成像模型：针孔模型 + 径向/切向畸变
- 4.3.2 标定验证：棋盘格标定 + 重投影误差（对齐表 4-2 的统计方式）
- 4.4.3 用标定后的映射关系把像素宽度换算为毫米

映射模型有两种，按需选用：
- ``mm_per_px``：工作距离处的平均尺度（论文"比值关系"的做法，默认）；
- ``H``：平面单应，适合视野内尺度变化不可忽略的场合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass
class Calibration:
    """标定结果。"""

    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    rms: float = 0.0
    mm_per_px: float | None = None
    H: np.ndarray | None = None
    image_size: tuple[int, int] | None = None  # (w, h)
    per_view_errors: list[float] = field(default_factory=list)

    def px_to_mm(self, px: float) -> float:
        """像素长度换算为毫米。未标定时返回 0（调用方需检查 ``is_calibrated``）。"""
        if not self.mm_per_px:
            return 0.0
        return float(px) * float(self.mm_per_px)

    @property
    def is_calibrated(self) -> bool:
        """是否具备毫米换算能力。"""
        return bool(self.mm_per_px)

    def to_dict(self) -> dict:
        return {
            "K": None if self.K is None else self.K.tolist(),
            "dist": None if self.dist is None else self.dist.ravel().tolist(),
            "rms": self.rms,
            "mm_per_px": self.mm_per_px,
            "H": None if self.H is None else self.H.tolist(),
            "image_size": self.image_size,
            "per_view_errors": self.per_view_errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(
            K=None if data.get("K") is None else np.asarray(data["K"], dtype=np.float64),
            dist=None if data.get("dist") is None else np.asarray(data["dist"], dtype=np.float64),
            rms=float(data.get("rms", 0.0)),
            mm_per_px=data.get("mm_per_px"),
            H=None if data.get("H") is None else np.asarray(data["H"], dtype=np.float64),
            image_size=None if data.get("image_size") is None else tuple(data["image_size"]),
            per_view_errors=list(data.get("per_view_errors", [])),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def scale_from_homography(H: np.ndarray, image_size: tuple[int, int], grid: int = 9) -> float:
    """由平面单应估计视野内的平均毫米/像素（在图像上采样若干点取均值）。"""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("H 必须是 3x3 单应矩阵")
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise ValueError("单应矩阵不可逆") from exc
    w, h = image_size
    xs = np.linspace(w * 0.1, w * 0.9, grid)
    ys = np.linspace(h * 0.1, h * 0.9, grid)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    dx = (pts + np.array([[[1.0, 0.0]]], dtype=np.float32))
    dy = (pts + np.array([[[0.0, 1.0]]], dtype=np.float32))
    world_dx = cv2.perspectiveTransform(dx, H_inv).reshape(-1, 2)
    world_dy = cv2.perspectiveTransform(dy, H_inv).reshape(-1, 2)
    step_x = np.linalg.norm(world_dx - world, axis=1)
    step_y = np.linalg.norm(world_dy - world, axis=1)
    return float(np.mean(np.concatenate([step_x, step_y])))


def reprojection_errors(
    obj_points: Sequence[np.ndarray],
    img_points: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> list[float]:
    """逐视图重投影误差（RMS，像素），用于对齐论文表 4-2 的统计方式。"""
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = proj.reshape(-1, 2) - np.asarray(img).reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))))
    return errors


def calibrate_from_points(
    obj_points: Sequence[np.ndarray],
    img_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> Calibration:
    """由已知点对完成标定（离线或测试用）。

    obj_points: 每视图 (N,3) 的物方点（使用平面靶标时 Z=0）
    img_points: 每视图 (N,2) 的图像点
    """
    obj = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in obj_points]
    img = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in img_points]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None)
    calib = Calibration(K=K, dist=dist, rms=float(rms), image_size=image_size)
    calib.per_view_errors = reprojection_errors(obj, img, rvecs, tvecs, K, dist)
    H, _ = cv2.findHomography(
        np.asarray(obj_points[0], dtype=np.float32)[:, :2], np.asarray(img_points[0], dtype=np.float32), 0
    )
    if H is not None:
        calib.H = H
        calib.mm_per_px = scale_from_homography(H, image_size)
    return calib


def calibrate_from_images(
    images: Sequence[str | Path],
    pattern: tuple[int, int] = (9, 6),
    square_mm: float = 5.0,
) -> Calibration:
    """棋盘格标定（论文 4.3.2）。

    pattern: 内角点列数、行数
    square_mm: 方格边长（毫米）
    """
    cols, rows = int(pattern[0]), int(pattern[1])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_mm)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        found, corners = cv2.findChessboardCorners(img, (cols, rows), None)
        if not found:
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners.reshape(-1, 2))

    if not obj_points or image_size is None:
        raise ValueError("没有找到可用的棋盘格图像，请检查 pattern 与图片路径")
    return calibrate_from_points(obj_points, img_points, image_size)
