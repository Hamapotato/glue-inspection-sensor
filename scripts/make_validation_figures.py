"""生成验证结果配图（docs/algo/figures/validation-*.png）。

三张图全部由实际代码运行产生，可重复生成：
    1. validation-five-cases.png    五类合成用例的端到端检测结果
    2. validation-preprocessing.png 预处理与量测各阶段中间结果
    3. validation-calibration.png   标定重投影误差与内参恢复情况

用法：
    py -3 scripts/make_validation_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algo.calib import Calibration, calibrate_from_points
from algo.config import load_config
from algo.measure import skeleton_length, zhang_suen_thinning
from algo.pipeline import GlueInspector
from algo.preprocess import morph_open_close, otsu_threshold, wavelet_denoise
from algo.synth import generate_case
from algo.visualize import draw_result

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("docs/algo/figures")
MM_PER_PX = 0.25  # [假设] 合成数据尺度，仅用于验证链路

KINDS = [
    ("normal", "正常胶条"),
    ("narrow", "过窄"),
    ("wide", "过宽"),
    ("broken", "断胶"),
    ("reflection", "反光 + 过窄"),
]


def fig_five_cases() -> Path:
    cfg = load_config()
    cfg["reflect"]["type"] = "heuristic"
    inspector = GlueInspector(cfg, calib=Calibration(mm_per_px=MM_PER_PX))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), dpi=150)
    axes = axes.ravel()
    for ax, (kind, title) in zip(axes, KINDS):
        case = generate_case(kind, size=(320, 480), width_px=40, seed=0)
        result = inspector.run(case.image)
        overlay = draw_result(case.image, result)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title} → {result.verdict.code}", fontsize=12)
        ax.axis("off")

    last = axes[len(KINDS)]
    last.axis("off")
    last.text(0.02, 0.95,
              "判定说明\n"
              "OK：合格\n"
              "TOO_NARROW：过窄\n"
              "TOO_WIDE：过宽\n"
              "BROKEN：断胶（RoI 数量 > 1）\n"
              "REFLECTION_SUSPECT：反光可疑，提示调参\n\n"
              f"合成数据尺度 {MM_PER_PX} mm/px\n"
              "结果仅验证链路，不作为指标结论",
              va="top", ha="left", fontsize=10.5, linespacing=1.6)

    fig.suptitle("五类合成用例端到端检测结果", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "validation-five-cases.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_preprocessing() -> Path:
    case = generate_case("narrow", size=(320, 480), width_px=40, seed=0)
    denoised = wavelet_denoise(case.image, level=2, mode="hard")
    _, mask = otsu_threshold(denoised)
    morph = morph_open_close(mask, ksize=5, order="open-close")
    skel = zhang_suen_thinning(morph > 0)
    length_px = skeleton_length(skel)
    area_px = float((morph > 0).sum())
    width_px = area_px / length_px if length_px else 0.0

    skel_rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
    skel_rgb[skel] = (255, 60, 60)

    panels = [
        (case.image, "① 原始图像", "gray"),
        (denoised, "② 小波去噪", "gray"),
        (mask, "③ Otsu 阈值分割", "gray"),
        (morph, "④ 开闭运算", "gray"),
        (skel_rgb, f"⑤ 骨架与面积\n宽≈{width_px:.1f}px / 长≈{length_px:.0f}px", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), dpi=150)
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11.5)
        ax.axis("off")
    fig.suptitle("预处理与量测各阶段中间结果（过窄用例）", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT_DIR / "validation-preprocessing.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_calibration() -> Path:
    """合成标定点 → 标定 → 画逐视图重投影误差。"""
    K_true = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]], dtype=np.float64)
    cols, rows, square = 9, 6, 5.0
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    objs, imgs = [], []
    rng = np.random.default_rng(0)
    for i in range(8):
        rvec = np.array([0.10 * ((i % 3) - 1), 0.10 * ((i // 3) - 1), 0.02 * i], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 400.0 + 10.0 * i], dtype=np.float64)
        pts, _ = cv2.projectPoints(objp, rvec, tvec, K_true, None)
        # 加入 0.1 px 的角点检测噪声，模拟真实的棋盘格角点定位误差
        pts = pts + rng.normal(0.0, 0.1, size=pts.shape)
        objs.append(objp)
        imgs.append(pts.reshape(-1, 2))

    calib = calibrate_from_points(objs, imgs, (640, 480))

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    xs = np.arange(1, len(calib.per_view_errors) + 1)
    ax.bar(xs, calib.per_view_errors, color="#42a5f5", edgecolor="#1565c0", width=0.6)
    ax.set_xlabel("视图编号")
    ax.set_ylabel("重投影误差 RMS（像素）")
    ax.set_ylim(0, max(0.30, max(calib.per_view_errors) * 1.6))
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_title(
        f"标定重投影误差（整体 RMS = {calib.rms:.4f} px）\n"
        f"恢复内参 fx = {calib.K[0, 0]:.1f}（真值 800.0），mm/px = {calib.mm_per_px:.4f}",
        fontsize=11.5,
    )
    fig.tight_layout()
    path = OUT_DIR / "validation-calibration.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_five_cases, fig_preprocessing, fig_calibration):
        print(f"saved {fn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
