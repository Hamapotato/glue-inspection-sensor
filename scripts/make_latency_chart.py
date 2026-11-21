"""生成阶段耗时对比图（docs/project/figures/stage-latency.png）。

数据来源：板端实测，20 张样例，commit e09418b（p50 柱高，误差线到 p95）。

用法：
    py -3 scripts/make_latency_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/project/figures/stage-latency.png")

STAGES = ["质量门控", "定位", "量测", "整帧"]
HEUR_P50 = [3.07, 6.41, 4.28, 10.61]
HEUR_P95 = [6.30, 12.10, 13.75, 25.85]
RPN_P50 = [3.06, 23.81, 4.56, 31.87]
RPN_P95 = [9.05, 54.28, 13.72, 62.65]


def main() -> int:
    x = np.arange(len(STAGES))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    zero = [0.0] * len(STAGES)
    b1 = ax.bar(x - w / 2, HEUR_P50, w, label="启发式定位", color="#90caf9",
                yerr=[zero, [p - q for p, q in zip(HEUR_P95, HEUR_P50)]], capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#1565c0"})
    b2 = ax.bar(x + w / 2, RPN_P50, w, label="RPN 定位", color="#a5d6a7",
                yerr=[zero, [p - q for p, q in zip(RPN_P95, RPN_P50)]], capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#2e7d32"})
    for bars, vals in ((b1, HEUR_P50), (b2, RPN_P50)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=9.5, color="#212121")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=11)
    ax.set_ylabel("耗时 (ms)", fontsize=11)
    ax.set_ylim(0, 72)
    ax.set_title("各阶段耗时：启发式 vs RPN（柱高 p50，误差线到 p95）", fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "数据：板端实测，20 张样例（commit e09418b）", fontsize=8.5, color="#607d8b")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
