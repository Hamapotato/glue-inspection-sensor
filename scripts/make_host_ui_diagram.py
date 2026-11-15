"""生成上位机 UI 交互框图（docs/host-ui-flow.png），配色与 host/AGENTS.md 第 6 节一致。
用法：py -3 scripts/make_host_ui_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/host-ui-flow.png")
BG = "#f4f6f9"
NAV = "#1e293b"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
GRAY = "#6b7280"
WHITE = "#ffffff"


def box(ax, x, y, w, h, text, face=WHITE, edge=GRAY, fs=10.2, bold=False, tc="#111827"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.22,rounding_size=1.1",
                                linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, color="#475569", ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 100)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    ax.text(50, 96, "涂胶检测上位机 · UI 交互框图", ha="center", va="center",
            fontsize=19, fontweight="bold", color="#111827")
    ax.text(50, 92, "文件回放模式 ｜ 检测 → 判定 → 入库 → 历史/统计 ｜ 配方驱动阈值",
            ha="center", va="center", fontsize=10.3, color=GRAY)

    # 左侧导航
    ax.add_patch(FancyBboxPatch((3, 46), 15, 42, boxstyle="round,pad=0.3,rounding_size=1.5",
                                facecolor=NAV, edgecolor=NAV, zorder=1))
    ax.text(10.5, 84.5, "导航", ha="center", va="center", fontsize=13, fontweight="bold", color="#e2e8f0")
    for label, y, hl in [("检测", 82, BLUE), ("配方", 75, None), ("历史", 68, None),
                         ("统计", 61, None), ("设置", 54, None)]:
        ax.add_patch(FancyBboxPatch((4.6, y - 2.2), 11.8, 4.4, boxstyle="round,pad=0.18,rounding_size=0.9",
                                    facecolor=hl or "#334155", edgecolor=hl or "#334155", zorder=2))
        ax.text(10.5, y, label, ha="center", va="center", fontsize=10.2, color="#f8fafc", zorder=3)

    # 快捷键卡片（导航下方独立）
    ax.add_patch(FancyBboxPatch((3, 27), 15, 17, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor="#e2e8f0", edgecolor="#cbd5e1", zorder=1))
    ax.text(10.5, 41.5, "快捷键", ha="center", fontsize=9.5, color="#334155", fontweight="bold")
    ax.text(10.5, 34, "← →  上/下一张\n空格  检测当前\nCtrl+B  批量检测\nF5  刷新统计",
            ha="center", va="center", fontsize=8.4, color="#334155", linespacing=1.75)

    # 中间：检测流程
    cx = 40
    box(ax, cx, 82, 30, 6.6, "① 扫描目录\n选择图片目录 / 自动枚举 png·jpg·bmp", edge=BLUE, fs=9.4)
    box(ax, cx, 72, 30, 6.6, "② 加载图像\n上一张 / 下一张 / 缩略预览", edge=BLUE, fs=9.4)
    box(ax, cx, 62, 30, 6.6, "③ 检测\n单帧 ｜ 批量（带进度）", edge=BLUE, fs=9.4)
    box(ax, cx, 52, 30, 6.6, "④ 结果\n判定徽章 ｜ 宽度 mm ｜ 分段耗时", edge=GREEN, fs=9.4)
    box(ax, cx, 42, 30, 6.6, "⑤ 入库\nSQLite inspection 表", edge=AMBER, fs=9.4)
    for y1, y2 in [(78.7, 75.3), (68.7, 65.3), (58.7, 55.3), (48.7, 45.3)]:
        arrow(ax, (cx, y1), (cx, y2))

    # 右列：配方 / 历史 / 统计
    box(ax, 79, 84, 30, 9.5, "配方页\n标准宽度 ｜ 预警/报警线 ｜ 标定 mm/px", edge=BLUE, fs=9.4)
    box(ax, 79, 72, 30, 8, "保存 / 加载配方 JSON\n影响下一次检测的判定阈值", edge=BLUE, fs=9.0, face="#eff6ff")
    arrow(ax, (79, 79.2), (79, 76.2), color=BLUE)
    arrow(ax, (64, 84), (56.5, 63.5), color=BLUE, ls="--")
    ax.text(61.5, 74.5, "阈值/标定", fontsize=8.2, color=BLUE, ha="center")

    box(ax, 79, 57, 30, 8.5, "历史页\n记录列表 ｜ 筛选 ｜ 清空 ｜ 导出", edge=GRAY, fs=9.4)
    box(ax, 79, 44, 30, 9, "统计页\n总检测 ｜ 合格率 ｜ p50/p95\n判定分布 ｜ 宽度直方图", edge=GRAY, fs=9.0)
    arrow(ax, (55.5, 42), (63.5, 54), color=AMBER, ls="--")
    arrow(ax, (79, 52.6), (79, 48.6), color=GRAY)

    # 底部：状态栏与降级
    box(ax, 40, 26, 48, 8.5, "状态栏：就绪 / 检测中 / 报警 / 相机断线（后续）\n错误原因显示在状态栏，不弹模态框", edge=GRAY, fs=9.2)
    box(ax, 79, 26, 30, 8.5, "降级\n未标定 → 只出像素宽度\n模型缺失 → 定位回退", edge=RED, fs=9.0)

    ax.text(50, 12, "交互原则：空状态给下一步指引 ｜ 批量操作不阻塞界面 ｜ 危险操作二次确认 ｜ 所有数值来自实测",
            ha="center", va="center", fontsize=9.3, color=GRAY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
