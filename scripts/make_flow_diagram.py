"""生成算法流程框图（默认 docs/algo/figures/algorithm-pipeline.png；--timing 生成带耗时版）。

只做绘图，不含算法逻辑。颜色编码实现状态：
    绿色 = 已实现并有单元测试
    橙色 = 代码就绪、尚未验证
    蓝色 = 输入 / 输出
红色文字 = 板端实测耗时（p50 / p95，20 张样例，commit e09418b）。

用法：
    py -3 scripts/make_flow_diagram.py
    py -3 scripts/make_flow_diagram.py --timing
    py -3 scripts/make_flow_diagram.py --timing --out docs/algo/figures/algorithm-pipeline-timing.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUT = Path("docs/algo/figures/algorithm-pipeline.png")

DONE = ("#e8f5e9", "#2e7d32")   # 已实现
WIP = ("#fff8e1", "#ef6c00")    # 就绪未验证
IO = ("#e3f2fd", "#1565c0")     # 输入输出
RAIL = "#90a4ae"
TIMING = "#c62828"              # 实测耗时标注


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


def timing_text(ax, x, y, text, fontsize=9.2):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=TIMING,
            zorder=4, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3",
                                edgecolor="#ef9a9a", linewidth=0.8))


def draw(ax, timing: bool = False):
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 104)
    ax.axis("off")

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
    b1_text = "B1  RoI 自动定位\nRPN + K-means 先验框 + DIoU 损失 + CIoU-NMS"
    if timing:
        b1_text += "\nRPN 实测 p50 23.81 / p95 54.28 ms（启发式 6.41 / 12.10）"
    box(ax, 52, 84, 62, 8.2 if timing else 7.0, b1_text, WIP, 10.0 if timing else 11.3)
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
    arrow(ax, 52, 91.8, 52, 88.1)        # 输入 → B1
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

    if timing:
        timing_text(
            ax, 52, 101.4,
            "质量门控（前置，工程新增）：p50 3.06 / p95 9.05 ms ｜ 不通过则整帧短路（约 3 ms）",
            fontsize=8.6,
        )
        timing_text(
            ax, 52, 29.6,
            "量测组（裁剪 + 预处理 + BFS/骨架）：p50 4.56 / p95 13.72 ms（与定位方式无关）",
            fontsize=8.4,
        )
        timing_text(
            ax, 52, 3.2,
            "整帧：RPN p50 31.87 / p95 62.65 ms ｜ 启发式 p50 10.61 / p95 25.85 ms ｜ 相机 10 fps = 100 ms/帧",
            fontsize=8.4,
        )

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

    title = "涂胶检测算法流程与实现状态" if not timing else "涂胶检测算法流程与板端实测耗时"
    ax.set_title(title, fontsize=16, weight="bold", pad=14, color="#212121")



def draw_simple_timing(ax):
    """简化流程 + 耗时：隐藏算法细节，只保留阶段和实测耗时（便于汇报）。"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 101)
    ax.axis("off")

    stages = [("输入", 92), ("门控", 80), ("定位", 64), ("量测", 44), ("判定", 26), ("输出", 12)]
    for label, y in stages:
        ax.text(8, y, label, ha="right", va="center", fontsize=11, color="#546e7a")
    ax.plot([10, 10], [10, 94], color=RAIL, lw=1.2, zorder=0)

    boxes = [
        (92, "输入：相机灰度图像（1440×1080 Mono8）", IO),
        (80, "质量门控（工程前置）\n暗 / 过曝 / 模糊 / 对比度不足 → NO_QUALITY", DONE),
        (64, "ROI 定位\nRPN（K-means 先验框 + NMS）", DONE),
        (44, "ROI 裁剪 → 预处理 → 量测\n分割/形态学 → 面积 & 骨架 → 平均宽度", DONE),
        (26, "缺陷判定\n断胶（RoI 数 > 1）｜ 过宽 / 过窄 ｜ 无法判定", DONE),
        (12, "输出：结果 JSON ｜ 报警信号 ｜ 记录", IO),
    ]
    for y, text, kind in boxes:
        box(ax, 46, y, 54, 7.2, text, kind, 10.6)
        ax.plot([10], [y], marker="o", markersize=4.5, color=RAIL, zorder=1)

    for y in (92, 80, 64, 44, 26):
        arrow(ax, 46, y - 3.7, 46, y - 8.4)

    timing = [
        (80, "3.06 / 9.05 ms\n（不通过则整帧短路 ≈3 ms）"),
        (64, "RPN 23.81 / 54.28 ms"),
        (44, "4.56 / 13.72 ms"),
        (26, "—（含在整帧内）"),
    ]
    for y, text in timing:
        timing_text(ax, 99, y, text, fontsize=9.0)

    ax.text(46, 2.5,
            "整帧：p50 31.87 ms / p95 62.65 ms",
            ha="center", va="center", fontsize=10.2, color=TIMING,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3f3", edgecolor="#ef9a9a"))

    ax.text(46, -4.2, "红色：板端实测 p50 / p95（20 张样例，commit e09418b）",
            ha="center", va="center", fontsize=9.2, color="#546e7a")
    ax.set_title("涂胶检测算法各阶段耗时（简化流程）", fontsize=16, weight="bold", pad=14, color="#212121")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true", help="在框图上标注板端实测耗时")
    ap.add_argument("--simple", action="store_true", help="简化流程（隐藏算法细节，突出耗时）")
    ap.add_argument("--out", default=None, help="输出路径（默认按模式决定）")
    args = ap.parse_args()

    if args.simple:
        fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=150)
        draw_simple_timing(ax)
        default_out = Path("docs/project/figures/algorithm-pipeline-timing-simple.png")
    else:
        fig, ax = plt.subplots(figsize=(13.5, 10), dpi=150)
        draw(ax, timing=args.timing)
        default_out = (
            Path("docs/project/figures/algorithm-pipeline-timing.png") if args.timing else DEFAULT_OUT
        )
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
