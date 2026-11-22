"""生成"论文算法 vs 产品算法"对比框图。

依据：docs/project/thesis-vs-product-algorithm.md（论文 3.1–5.3；产品册 P9–P17）。
输出：docs/project/figures/thesis-vs-product-algorithm.png

用法：
    py -3 scripts/make_thesis_product_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/project/figures/thesis-vs-product-algorithm.png")

# 配色
DL = ("#e3f2fd", "#1565c0")        # 深度学习相关
TRAD = ("#e8f5e9", "#2e7d32")      # 传统图像处理/几何
TEACH = ("#fff8e1", "#ef6c00")     # 示教/规则
IO = ("#eceff1", "#455a64")        # 输入输出/未公开
NOTE = "#546e7a"


def box(ax, x, y, w, h, text, kind=TRAD, fontsize=9.4, ls="-"):
    face, edge = kind
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
            linestyle=ls,
            zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#212121", zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#37474f"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", linestyle=ls, mutation_scale=11,
            linewidth=1.3, color=color, zorder=1, shrinkA=1, shrinkB=1,
        )
    )


def column(ax, cx, w, boxes, color_map):
    for (y, text, key), next_y in zip(boxes, [b[0] for b in boxes[1:]] + [None]):
        box(ax, cx, y, w, 7.0, text, color_map[key] if not isinstance(key, tuple) else key)
        if next_y is not None:
            arrow(ax, cx, y - 3.5, cx, next_y + 3.5)


def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(-8, 100)
    ax.axis("off")

    # 列标题
    for cx, title, sub in [
        (18, "论文（天津大学，2022）", "深度学习辅助 + 几何量测"),
        (54, "产品册 V3.3 · 标准检测", "示教 + 图像处理（不含深度学习）"),
        (86, "产品册 V3.3 · 智能检测", "深度学习 CNN（选配）"),
    ]:
        ax.text(cx, 97.5, title, ha="center", va="center", fontsize=13, weight="bold", color="#212121")
        ax.text(cx, 93.6, sub, ha="center", va="center", fontsize=9.5, color=NOTE)

    left = [
        (87, "输入：三相机 120° 图像\n（红色 LED / 24V 触发）", "io"),
        (77, "预处理：小波硬阈值去噪（式 4-4）", "trad"),
        (67, "RoI 定位：RPN（K-means 9 先验框\n+ DIoU/CIoU + CIoU-NMS）", "dl"),
        (57, "RoI 裁剪", "trad"),
        (47, "分割：Otsu/阈值 + 开闭运算", "trad"),
        (37, "量测：BFS 面积 + Zhang-Suen 骨架\n→ 宽度 = 面积 ÷ 骨架长度", "trad"),
        (27, "标定：13×9 棋盘格 → 像素→毫米", "trad"),
        (17, "反光复判：YOLOv3-5FL + SPP\n（复判提示，不直接判废）", "dl"),
        (7, "判定：断胶（RoI>1）/ 过宽 / 过窄\n→ 输出结果 / 报警", "io"),
    ]
    mid = [
        (87, "输入：1K/2K 相机\n360° 环形传感器", "io"),
        (77, "红色 LED 均匀照射", "trad"),
        (67, "示教：车型 / 轨迹 / ROI / 阈值", "teach"),
        (57, "图像处理：灰度、位置信息", "trad"),
        (47, "基于示教信息判断\n连续性 / 胶宽 / 位置", "teach"),
        (37, "缺陷判定：断胶 / 过宽 / 过窄\n/ 位置偏差 / 多胶", "trad"),
        (27, "输出：结果 / 报警 / 报表 / 追溯", "io"),
    ]
    right = [
        (87, "输入：三相机图像", "io"),
        (77, "训练：标签 + 大量胶水图像", "dl"),
        (67, "CNN：Backbone + Neck\n+ 检测/坐标头（结构未公开）", "dl"),
        (57, "输出：缺陷类别 / 坐标", "dl"),
        (47, "判定：依赖网络输出", "dl"),
        (27, "输出：结果 / 报警", "io"),
    ]

    colors = {"io": IO, "trad": TRAD, "dl": DL, "teach": TEACH}
    column(ax, 18, 32, left, colors)
    column(ax, 54, 30, mid, colors)

    # 智能检测列：虚线框表示"选配"
    ax.add_patch(
        FancyBboxPatch(
            (72.5, 20), 25.5, 70.5, boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.2, edgecolor="#90a4ae", facecolor="none", linestyle="--", zorder=0,
        )
    )
    column(ax, 86, 24, right, colors)

    ax.text(86, 18.0, "注：训练数据、网络结构、指标未公开", ha="center", va="center", fontsize=8.8, color=NOTE)

    # 差异/共同点注释
    ax.text(
        50, -4.2,
        "共同点：三相机 + 环形红色 LED + IO 触发；缺陷定义一致（断胶、过宽/过窄、多胶、位置偏差）。\n"
        "差异：论文把深度学习用在“定位/抗反光”，宽度由几何量测保证；产品标准模式不含深度学习，智能模式把检测交给 CNN（结构未公开）。",
        ha="center", va="center", fontsize=9.2, color=NOTE, linespacing=1.5,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
