"""图像预处理与分割（论文第 4 章 4.1–4.2）。

- 4.1.1 基于小波分解的滤波：论文采用硬阈值处理，以尽量保留边界细节
- 4.1.2 与中值滤波做实验对比
- 4.2.1 阈值分割实验对比：直方图双峰法 / 全局阈值法 / 最大类间方差法（Otsu）
- 4.2.2 开闭操作修正图像缺陷

论文使用的小波基与分解层数未公开，本实现默认 Haar（可用 pywt 替换），
层数与阈值都在配置文件里给出，属于"复现假设"，需在报告中注明。
"""

from __future__ import annotations

import cv2
import numpy as np

_SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# 小波分解（Haar，正交，可逆）
# --------------------------------------------------------------------------- #

def _pad_to_even(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    ph, pw = h % 2, w % 2
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), mode="reflect")
    return arr


def dwt2_haar(arr: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """单层二维 Haar 小波分解，返回 (LL, (LH, HL, HH))。"""
    a = np.asarray(arr, dtype=np.float64)
    a = _pad_to_even(a)
    lo = (a[:, 0::2] + a[:, 1::2]) / _SQRT2
    hi = (a[:, 0::2] - a[:, 1::2]) / _SQRT2
    ll = (lo[0::2, :] + lo[1::2, :]) / _SQRT2
    lh = (lo[0::2, :] - lo[1::2, :]) / _SQRT2
    hl = (hi[0::2, :] + hi[1::2, :]) / _SQRT2
    hh = (hi[0::2, :] - hi[1::2, :]) / _SQRT2
    return ll, (lh, hl, hh)


def idwt2_haar(ll: np.ndarray, details: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """单层二维 Haar 小波重构。"""
    lh, hl, hh = details
    lo = np.zeros((ll.shape[0] * 2, ll.shape[1]), dtype=np.float64)
    lo[0::2, :] = (ll + lh) / _SQRT2
    lo[1::2, :] = (ll - lh) / _SQRT2
    hi = np.zeros_like(lo)
    hi[0::2, :] = (hl + hh) / _SQRT2
    hi[1::2, :] = (hl - hh) / _SQRT2
    out = np.zeros((lo.shape[0], lo.shape[1] * 2), dtype=np.float64)
    out[:, 0::2] = (lo + hi) / _SQRT2
    out[:, 1::2] = (lo - hi) / _SQRT2
    return out


def _universal_threshold(detail: np.ndarray) -> float:
    """通用阈值 T = σ·sqrt(2·ln N)，σ = median(|HH|)/0.6745。"""
    sigma = np.median(np.abs(detail)) / 0.6745
    n = detail.size
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def wavelet_denoise(
    img: np.ndarray,
    level: int = 2,
    mode: str = "hard",
    threshold: float | None = None,
) -> np.ndarray:
    """小波阈值去噪（论文 4.1.1，默认硬阈值）。

    img: 灰度 uint8；返回同尺寸 uint8。
    分解 level 层 → 对各层细节系数做阈值处理 → 由深到浅逐层重构。
    """
    img = _as_gray(img)
    h, w = img.shape[:2]
    ll = img.astype(np.float64) / 255.0
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(max(1, level)):
        ll, d = dwt2_haar(ll)
        details.append(d)

    processed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for lh, hl, hh in details:
        t = threshold if threshold is not None else _universal_threshold(hh)
        processed.append(
            (_apply_threshold(lh, t, mode), _apply_threshold(hl, t, mode), _apply_threshold(hh, t, mode))
        )

    for d in reversed(processed):
        ll = idwt2_haar(ll, d)

    out = np.clip(ll * 255.0, 0, 255)
    return out[:h, :w].astype(np.uint8)


def _apply_threshold(coef: np.ndarray, thr: float, mode: str) -> np.ndarray:
    if mode == "soft":
        return np.sign(coef) * np.maximum(np.abs(coef) - thr, 0.0)
    return np.where(np.abs(coef) >= thr, coef, 0.0)


def median_denoise(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """中值滤波（论文 4.1.2 的对比方法）。"""
    return cv2.medianBlur(_as_gray(img), ksize)


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #

def otsu_threshold(img: np.ndarray) -> tuple[float, np.ndarray]:
    """最大类间方差法（论文 4.2.1 选用的方法）。"""
    img = _as_gray(img)
    thr, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr), mask


def global_threshold(img: np.ndarray, thr: float | None = None) -> tuple[float, np.ndarray]:
    """全局阈值法（论文对比项）。"""
    img = _as_gray(img)
    t = float(np.mean(img)) if thr is None else float(thr)
    return t, np.where(img > t, 255, 0).astype(np.uint8)


def double_peak_threshold(img: np.ndarray) -> tuple[float, np.ndarray]:
    """直方图双峰法（论文对比项）。

    对直方图做平滑后找两个主峰，取两峰之间的谷底作为阈值；
    若找不到明显双峰，退化为 Otsu。
    """
    img = _as_gray(img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(hist, kernel, mode="same")
    peaks = [i for i in range(1, 255) if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0]
    if len(peaks) < 2:
        return otsu_threshold(img)
    peaks.sort(key=lambda i: -smooth[i])
    p1 = peaks[0]
    p2 = next((p for p in peaks[1:] if abs(p - p1) >= 10), None)
    if p2 is None:
        return otsu_threshold(img)
    lo, hi = sorted((p1, p2))
    valley = lo + int(np.argmin(smooth[lo:hi + 1]))
    mask = np.where(img > valley, 255, 0).astype(np.uint8)
    return float(valley), mask


SEGMENT_METHODS = {
    "otsu": otsu_threshold,
    "global": global_threshold,
    "double_peak": double_peak_threshold,
}


def segment(img: np.ndarray, method: str = "otsu", **kwargs) -> tuple[float, np.ndarray]:
    if method not in SEGMENT_METHODS:
        raise ValueError(f"未知分割方法 {method}，可选 {list(SEGMENT_METHODS)}")
    return SEGMENT_METHODS[method](img, **kwargs)


def morph_open_close(mask: np.ndarray, ksize: int = 5, order: str = "open-close") -> np.ndarray:
    """开闭运算修正图像缺陷（论文 4.2.2）。

    开运算：先腐蚀后膨胀，去除小噪点与毛刺；
    闭运算：先膨胀后腐蚀，填补孔洞与断裂。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = mask
    if order in ("open-close", "open", "close-open"):
        if order.startswith("open"):
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        if order != "open":
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if order == "close-open":
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


# --------------------------------------------------------------------------- #

def _as_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
