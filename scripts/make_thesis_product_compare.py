"""生成"论文算法 vs 产品算法"对比框图（v2：结构优化版）。

依据：docs/project/thesis-vs-product-algorithm.md（论文 3.1–5.3；产品册 P9–P17）。
产品册原文只写"系统功能——智能检测（选配）"（P11–P12）；P10/P13 描述的
"示教 + 图像处理"路线本文档简称"基础检测"，不是厂商的官方叫法。

输出：docs/project/figures/thesis-vs-product-algorithm.png
用法：py -3 scripts/make_thesis_product_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/project/figures/thesis-vs-product-algorithm.png")

DL = ("#e3f2fd", "#1565c0")        # 深度学习
TRAD = ("#e8f5e9", "#2e7d32")      # 传统图像处理/几何
TEACH = ("#fff8e1", "#ef6c00")     # 示教/规则
IO = ("#eceff1", "#455a64")        # 输入输出
NOTE = "#546e7a"


def box(ax, x, y, w, h, text, kind=TRAD, fontsize=9.6, ls="-"):
    face, edge = kind
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            linewidth=1.4, edgecolor=edge, facecolor=face, linestyle=ls, zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#212121", zorder=3, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#37474f", lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", linestyle=ls, mutation_scale=11,
        linewidth=lw, color=color, zorder=1, shrinkA=1, shrinkB=1,
    ))


def flow(ax, cx, w, items, colors, h=7.0):
    """垂直流：items = [(y, text, key)]。"""
    for (y, text, key) in items:
        box(ax, cx, y, w, h, text, colors[key] if isinstance(key, str) else key)
    ys = [it[0] for it in items]
    for a, b in zip(ys, ys[1:]):
        arrow(ax, cx, a - h / 2, cx, b + h / 2)


def main() -> int:
    fig, ax = plt.subplots(figsize=(17, 11.5), dpi=150)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 100)
    ax.axis("off")

    title_y, sub_y = 98.0, 94.8
    tips = [
        (21, "论文（天津大学，2022）", "深度学习辅助 + 几何量测"),
        (66, "产品册 V3.3 · 基础检测", "示教 + 图像处理（不含深度学习，P10/P13）"),
        (106, "产品册 V3.3 · 智能检测（选配）", "CNN（结构/数据/指标未公开，P11–P12）"),
    ]
    for x, title, sub in tips:
        ax.text(x, title_y, title, ha="center", va="center", fontsize=13, weight="bold", color="#212121")
        ax.text(x, sub_y, sub, ha="center", va="center", fontsize=9.2, color=NOTE)

    colors = {"io": IO, "trad": TRAD, "dl": DL, "teach": TEACH, "note": IO}

    # ---------- 论文 ----------
    paper = [
        (90, "输入：三相机 120° 图像\n（红色 LED / 24V 触发）", "io"),
        (79, "预处理：小波硬阈值去噪", "trad"),
        (68, "RoI 定位：RPN\n（K-means 9 先验框 + DIoU/CIoU + NMS）", "dl"),
        (57, "RoI 裁剪", "trad"),
        (46, "分割：Otsu/阈值 + 开闭运算", "trad"),
        (35, "量测：BFS 面积 + Zhang-Suen 骨架\n宽度 = 面积 ÷ 骨架长度\n（标定：13×9 棋盘格 → 毫米，离线一次）", "trad"),
        (21, "判定：断胶（RoI>1）/ 过宽 / 过窄\n（YOLOv3-5FL+SPP 反光复判，可选提示）", "trad"),
        (10, "输出：结果 / 报警", "io"),
    ]
    flow(ax, 21, 35, paper, colors)

    # ---------- 产品·基础检测 ----------
    basic = [
        (90, "输入：1K/2K 三相机 360°\n（红色 LED 均匀照射）", "io"),
        (79, "示教：车型 / 轨迹 / ROI / 阈值", "teach"),
        (68, "图像处理：灰度、位置信息", "trad"),
        (57, "基于示教判断\n连续性 / 胶宽 / 位置", "teach"),
        (46, "缺陷判定：断胶 / 过宽 / 过窄\n/ 位置偏差 / 多胶", "trad"),
        (22, "输出：结果 / 报警 / 报表 / 追溯", "io"),
    ]
    flow(ax, 66, 35, basic, colors)

    # ---------- 产品·智能检测（离线训练 + 在线检测） ----------
    ax.add_patch(Rectangle((88.5, 66.5), 35.5, 22.5, linewidth=1.2, edgecolor="#90a4ae",
                           facecolor="none", linestyle="--", zorder=0))
    ax.text(106.2, 87.4, "离线训练", ha="center", va="center", fontsize=9, color=NOTE)
    box(ax, 106, 82, 33, 7, "大量涂胶图像 + 标签", DL, 9.6, ls="--")
    arrow(ax, 106, 78.5, 106, 74.5, ls="--")
    box(ax, 106, 71, 33, 7, "CNN 训练（Backbone + Neck + 头）", DL, 9.2, ls="--")

    detect = [
        (60, "在线检测：输入图像", "io"),
        (50, "CNN 推理（Backbone + Neck\n+ 检测/坐标头）", "dl"),
        (40, "输出：缺陷类别 / 坐标", "dl"),
        (30, "判定：依赖网络输出", "dl"),
        (20, "输出：结果 / 报警", "io"),
    ]
    flow(ax, 106, 34, detect, colors)
    arrow(ax, 106, 67.5, 106, 63.5, ls="--", color="#1565c0", lw=1.2)

    # 图例
    legend = [
        (8, "深度学习", DL, "-"),
        (26, "传统图像处理/几何", TRAD, "-"),
        (52, "示教/规则", TEACH, "-"),
        (70, "输入/输出", IO, "-"),
        (86, "虚线：离线或可选", IO, "--"),
    ]
    for x, label, (face, edge), ls in legend:
        ax.add_patch(Rectangle((x, 2.2), 3.2, 2.4, facecolor=face, edgecolor=edge, linewidth=1.2, linestyle=ls))
        ax.text(x + 4.2, 3.4, label, fontsize=9, va="center", color="#37474f")

    # 共同点/差异
    ax.text(
        65, -4.2,
        "共同点：三相机 + 环形红色 LED + IO 触发；缺陷定义一致（断胶、过宽/过窄、多胶、位置偏差）。\n"
        "差异：论文把深度学习用在“定位/抗反光”，宽度由几何量测保证；产品基础检测不含深度学习，智能检测把检测交给 CNN（选配）。",
        ha="center", va="center", fontsize=9.2, color=NOTE, linespacing=1.5,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
