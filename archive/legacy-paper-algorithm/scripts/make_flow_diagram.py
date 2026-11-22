"""生成算法流程框图（docs/algo/figures/algorithm-pipeline.png）。

只做绘图，不含算法逻辑。颜色编码实现状态：
    绿色 = 已实现并有单元测试
    橙色 = 代码就绪、尚未验证
    蓝色 = 输入 / 输出
用法：
    py -3 scripts/make_flow_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/algo/figures/algorithm-pipeline.png")

DONE = ("#e8f5e9", "#2e7d32")   # 已实现
WIP = ("#fff8e1", "#ef6c00")    # 就绪未验证
IO = ("#e3f2fd", "#1565c0")     # 输入输出
RAIL = "#90a4ae"


def box(ax, x, y, w, h, text, kind=DONE, fontsize=12):
    face, edge = kind
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.30,rounding_size=1.3",
        linewidth=1.7, edgecolor=edge, facecolor=face, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#212121", zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#37474f", lw=1.7):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", linestyle=ls,
        mutation_scale=14, linewidth=lw, color=color, zorder=1,
        shrinkA=1, shrinkB=1,
    ))


def stage_rail(ax, stages):
    """左侧阶段标注轨：竖线 + 圆点 + 文字。"""
    ys = [y for _, y in stages]
    ax.plot([8.2, 8.2], [min(ys) - 2, max(ys) + 2], color=RAIL, lw=1.2, zorder=0)
    for label, y in stages:
        ax.plot([8.2], [y], marker="o", markersize=4.5, color=RAIL, zorder=1)
        ax.text(6.6, y, label, ha="right", va="center", fontsize=9, color="#546e7a")


def main() -> int:
    fig, ax = plt.subplots(figsize=(13.5, 10), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 101)
    ax.axis("off")

    # 阶段标注
    stage_rail(ax, [
        ("输入", 95),
        ("定位", 78.5),
        ("预处理", 61),
        ("量测", 42),
        ("判定", 24),
        ("输出", 11),
    ])

    # ① 输入
    box(ax, 52, 95, 60, 6.4, "输入：工业相机图像 1280×960（3 相机 / 24V 硬触发）", IO, 12.5)

    # ② 定位
    box(ax, 52, 84, 58, 7.0, "B1  RoI 自动定位\nRPN + K-means 先验框 + DIoU 损失 + CIoU-NMS", WIP, 11.3)
    box(ax, 52, 73, 58, 6.0, "RoI 裁剪", DONE, 12)

    # ③ 预处理
    box(ax, 22, 61, 26, 6.4, "小波阈值去噪\n（Haar，硬阈值）", DONE, 10.6)
    box(ax, 52, 61, 20, 6.4, "Otsu 阈值分割", DONE, 11.2)
    box(ax, 82, 61, 26, 6.4, "开 / 闭运算\n补孔洞、去噪点", DONE, 10.6)

    # ④ 量测
    box(ax, 32, 48, 28, 6.4, "BFS 面积\n（外轮廓法对照）", DONE, 10.8)
    box(ax, 72, 48, 28, 6.4, "Zhang-Suen 骨架\n→ 长度", DONE, 10.8)
    box(ax, 52, 36, 40, 6.4, "平均宽度 = 面积 / 骨架长度", DONE, 11.2)

    # 标定（离线旁路）
    box(ax, 90, 36, 18, 6.4, "相机标定\n像素 → 毫米", DONE, 9.6)
    arrow(ax, 81.5, 36, 72.5, 36, ls="--", color="#1565c0", lw=1.4)

    # ⑤ 判定
    box(ax, 34, 24, 44, 7.0, "缺陷判定\n断胶（RoI 数 > 1）｜ 过宽 / 过窄（双报警线）", DONE, 10.6)
    box(ax, 80, 24, 32, 7.0, "B2 反光复判\nYOLOv3-5FL + SPP（未验证）", WIP, 10.4)

    # ⑥ 输出
    box(ax, 52, 11, 60, 6.4, "输出：结果 JSON ｜ 报警信号 ｜ 数据库留档 / 报表", IO, 12)

    # ---------------- 箭头 ----------------
    arrow(ax, 52, 91.8, 52, 87.5)        # 输入 → B1
    arrow(ax, 52, 80.5, 52, 76.0)        # B1 → 裁剪
    arrow(ax, 46, 70.0, 22, 64.2)        # 裁剪 → 去噪
    arrow(ax, 52, 70.0, 52, 64.2)        # 裁剪 → Otsu
    arrow(ax, 58, 70.0, 82, 64.2)        # 裁剪 → 形态学
    arrow(ax, 22, 57.8, 32, 51.2)        # 去噪 → 面积
    arrow(ax, 52, 57.8, 32, 51.2)        # Otsu → 面积
    arrow(ax, 52, 57.8, 72, 51.2)        # Otsu → 骨架
    arrow(ax, 82, 57.8, 72, 51.2)        # 形态学 → 骨架
    arrow(ax, 32, 44.8, 44, 39.2)        # 面积 → 宽度
    arrow(ax, 72, 44.8, 60, 39.2)        # 骨架 → 宽度
    arrow(ax, 52, 32.8, 40, 27.5)        # 宽度 → 判定
    arrow(ax, 56.2, 24, 63.8, 24, ls="--", color="#ef6c00", lw=1.4)  # 判定 → 反光复判
    arrow(ax, 34, 20.5, 42, 14.2)        # 判定 → 输出
    arrow(ax, 80, 20.5, 62, 14.2)        # 反光复判 → 输出

    # 图例
    items = [
        ("已实现并有测试", DONE, "-"),
        ("代码就绪、未验证", WIP, "-"),
        ("输入 / 输出", IO, "-"),
        ("虚线：反光复判", WIP, "--"),
    ]
    for i, (label, (face, edge), ls) in enumerate(items):
        x = 6 + i * 23
        ax.add_patch(plt.Rectangle((x, -5.4), 2.8, 2.4, facecolor=face,
                                   edgecolor=edge, linewidth=1.4, linestyle=ls))
        ax.text(x + 3.9, -4.2, label, fontsize=9, va="center", color="#37474f")

    ax.set_title("涂胶检测算法流程与实现状态", fontsize=16, weight="bold", pad=14, color="#212121")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
