"""生成算法流程框图（docs/algo/figures/algorithm-pipeline.png）。

只做绘图，不含算法逻辑；颜色表示实现状态：
    绿色 = 已实现并有测试
    橙色 = 代码就绪、尚未验证
    灰色 = 未实现
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

COLORS = {
    "done": ("#e8f5e9", "#2e7d32"),      # 绿：已实现
    "wip": ("#fff8e1", "#ef6c00"),       # 橙：就绪未验证
    "todo": ("#f5f5f5", "#616161"),      # 灰：未实现
    "io": ("#e3f2fd", "#1565c0"),        # 蓝：输入输出
}


def box(ax, x, y, w, h, text, kind="done", fontsize=12):
    face, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.32,rounding_size=1.4",
        linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#212121", zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#37474f", lw=1.8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", linestyle=ls,
        mutation_scale=15, linewidth=lw, color=color, zorder=1,
        shrinkA=1, shrinkB=1,
    ))


def main() -> int:
    fig, ax = plt.subplots(figsize=(13, 10), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 100)
    ax.axis("off")

    # ① 输入
    box(ax, 50, 94, 64, 6.5, "输入：工业相机图像 1280×960（3 相机 / 24V 硬触发）", "io", 13)

    # ② 定位
    box(ax, 50, 82, 58, 7, "B1  RoI 自动定位\nRPN + K-means 先验框 + DIoU 损失 + CIoU-NMS", "wip", 11.5)

    # ③ 裁剪
    box(ax, 50, 70, 58, 6, "RoI 裁剪", "done", 12)

    # ④ 预处理
    box(ax, 21, 57, 27, 6.5, "小波阈值去噪\n（Haar，硬阈值）", "done", 10.8)
    box(ax, 50, 57, 22, 6.5, "Otsu 阈值分割", "done", 11.5)
    box(ax, 79, 57, 27, 6.5, "开 / 闭运算\n补孔洞、去噪点", "done", 10.8)

    # ⑤ 量测
    box(ax, 30, 44, 28, 6.5, "BFS 面积\n（外轮廓法对照）", "done", 11)
    box(ax, 70, 44, 28, 6.5, "Zhang-Suen 骨架\n→ 长度", "done", 11)
    box(ax, 50, 32, 40, 6.5, "平均宽度 = 面积 / 骨架长度", "done", 11.5)

    # ⑥ 标定（旁路输入，不是流程节点）
    box(ax, 85, 32, 22, 6.5, "相机标定（离线）\n像素 → 毫米", "done", 10.2)
    arrow(ax, 74, 32, 70.5, 32, ls="--", color="#1565c0", lw=1.5)
    ax.text(72.2, 33.6, "标定值", ha="center", fontsize=8, color="#1565c0")

    # ⑦ 判定
    box(ax, 33, 18, 48, 6.8, "缺陷判定\n断胶（RoI 数 > 1）｜ 过宽 / 过窄（双报警线）", "done", 10.8)
    box(ax, 79, 18, 34, 6.8, "B2 反光复判\nYOLOv3-5FL + SPP（未验证）", "wip", 10.5)

    # ⑧ 输出
    box(ax, 50, 5, 58, 6.5, "输出：结果 JSON ｜ 报警信号 ｜ 数据库留档 / 报表", "io", 12)

    # ---------------- 箭头 ----------------
    arrow(ax, 50, 90.6, 50, 85.6)       # 输入 → B1
    arrow(ax, 50, 78.4, 50, 73.1)       # B1 → 裁剪
    arrow(ax, 44, 66.9, 21, 60.4)       # 裁剪 → 去噪
    arrow(ax, 50, 66.9, 50, 60.4)       # 裁剪 → Otsu
    arrow(ax, 56, 66.9, 79, 60.4)       # 裁剪 → 形态学
    arrow(ax, 21, 53.6, 30, 47.4)       # 去噪 → 面积
    arrow(ax, 50, 53.6, 30, 47.4)       # Otsu → 面积
    arrow(ax, 50, 53.6, 70, 47.4)       # Otsu → 骨架
    arrow(ax, 79, 53.6, 70, 47.4)       # 形态学 → 骨架
    arrow(ax, 30, 40.6, 40, 35.4)       # 面积 → 宽度
    arrow(ax, 70, 40.6, 60, 35.4)       # 骨架 → 宽度
    arrow(ax, 45, 28.6, 36, 21.6)       # 宽度 → 判定
    arrow(ax, 57, 17.5, 61.8, 17.5, ls="--", color="#ef6c00", lw=1.5)  # 判定 → 反光复判
    arrow(ax, 79, 14.4, 62, 8.6)        # 反光复判 → 输出
    arrow(ax, 33, 14.4, 44, 8.6)        # 判定 → 输出

    # 图例
    handles = [
        ("已实现并有测试", COLORS["done"]),
        ("代码就绪、未验证", COLORS["wip"]),
        ("输入 / 输出", COLORS["io"]),
    ]
    for i, (label, (face, edge)) in enumerate(handles):
        x = 8 + i * 26
        ax.add_patch(plt.Rectangle((x, -4.2), 3, 2.4, facecolor=face, edgecolor=edge, linewidth=1.5))
        ax.text(x + 4.2, -3.0, label, fontsize=9.5, va="center", color="#37474f")

    ax.text(99, -3.0, "状态：19/37 任务完成", ha="right", fontsize=9.5, color="#78909c")

    ax.set_title("涂胶检测算法流程与实现状态", fontsize=16, weight="bold", pad=12, color="#212121")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
