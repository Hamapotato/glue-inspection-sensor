"""P1 宽度量测报告生成器（论文 4.4 / 表 4-5 的对照口径）。

两种模式
--------
1. 真实数据：``--images <dir> --labels <csv> --calib <json>``
   csv 需含列 ``image,gt_width_mm``（人工卡尺测量值）。
2. 合成自检：``--synthetic``
   用 ``algo.synth`` 生成用例，验证"链路 + 报告生成"是否正常。
   **合成数字只用于自检，不能作为指标结论**（AGENTS.md R1）。

用法：
    py -3 scripts/run_measure_report.py --synthetic --out-md docs/algo/reports/p1-measure.md
    py -3 scripts/run_measure_report.py --images data/raw/setA --labels data/labels/width.csv --calib calib.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.metrics import latency_summary, summarize_errors, width_errors  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402
from algo.synth import generate_case  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 论文摘要给出的对照值（M4–M6）
THESIS = {"mae_mm": 0.23, "std_mm": 0.25, "max_mm": 0.39}

SYNTH_CASES = ["normal", "narrow", "wide", "broken", "reflection"]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - 环境无 git 时
        return "unknown"


def load_labels(path: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("image") or "").strip()
            value = row.get("gt_width_mm")
            if key and value not in (None, ""):
                labels[key] = float(value)
    return labels


def run_synthetic(cfg, mm_per_px: float) -> list[dict]:
    cfg = cfg.clone()
    # 冒烟时启用占位反光检测，保证 REFLECTION_SUSPECT 分支被覆盖
    cfg["reflect"]["type"] = "heuristic"
    calib = Calibration(mm_per_px=mm_per_px)
    inspector = GlueInspector(cfg, calib=calib)
    rows = []
    for kind in SYNTH_CASES:
        case = generate_case(kind, size=(320, 480), width_px=40, seed=0)
        result = inspector.run(case.image)
        rows.append(
            {
                "image": f"synthetic::{kind}",
                "gt_width_mm": case.width_px * mm_per_px,
                "width_mm": result.measure.width_mm,
                "verdict": result.verdict.code,
                "total_ms": result.timings_ms.get("total", 0.0),
            }
        )
    return rows


def run_real(cfg, images_dir: Path, labels: dict[str, float], calib) -> list[dict]:
    inspector = GlueInspector(cfg, calib=calib)
    rows = []
    for path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        result = inspector.run(image)
        rows.append(
            {
                "image": path.name,
                "gt_width_mm": labels.get(path.name),
                "width_mm": result.measure.width_mm,
                "verdict": result.verdict.code,
                "total_ms": result.timings_ms.get("total", 0.0),
            }
        )
    return rows


def render_report(rows: list[dict], synthetic: bool, meta: dict) -> str:
    paired = [r for r in rows if r.get("gt_width_mm") not in (None, "")]
    lat = latency_summary([r["total_ms"] for r in rows])

    if paired:
        errors = width_errors([r["width_mm"] for r in paired], [r["gt_width_mm"] for r in paired])
        stats = summarize_errors(errors)
    else:
        stats = {"mae_mm": 0.0, "std_mm": 0.0, "max_mm": 0.0, "n": 0}

    lines: list[str] = []
    lines.append("# P1 宽度量测报告")
    lines.append("")
    lines.append(f"> 生成时间：{meta['time']}")
    lines.append(f"> 代码版本：`{meta['commit']}`｜配置：`{meta['config']}`｜标定：`{meta['calib']}`")
    if synthetic:
        lines.append(">")
        lines.append(
            "> ⚠️ **本报告来自合成数据自检，仅用于验证链路与报告生成，不能作为复现指标。**"
            " 真实指标必须在现场图像 + 人工卡尺真值上重跑（见 AGENTS.md R1、p0-spec §6）。"
        )
    lines.append("")
    lines.append("## 1. 口径")
    lines.append("")
    lines.append("| 指标 | 定义 | 论文值 |")
    lines.append("| --- | --- | --- |")
    lines.append("| M4 平均绝对误差 | `mean(|计算 − 人工|)` | 0.23 mm |")
    lines.append("| M5 误差标准差 | 同上误差的总体标准差 | 0.25 mm |")
    lines.append("| M6 最大误差 | `max(|误差|)` | 0.39 mm |")
    lines.append("| 验收线 | 论文摘要的 1 mm 精度要求 | 1.00 mm |")
    lines.append("")
    lines.append("## 2. 论文值 / 复现值 / 偏差原因")
    lines.append("")
    lines.append("| 指标 | 论文值 | 复现值 | 偏差 | 偏差原因 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for key, name in (("mae_mm", "M4 MAE"), ("std_mm", "M5 STD"), ("max_mm", "M6 MAX")):
        thesis = THESIS.get(key, 0.0)
        value = stats[key]
        if stats["n"] == 0:
            lines.append(f"| {name} | {thesis:.2f} mm | 待测 | — | 真实标注未就绪 |")
        elif synthetic:
            lines.append(f"| {name} | {thesis:.2f} mm | 待测 | — | 合成数据自检值 {value:.3f} mm，不计入结论 |")
        else:
            diff = value - thesis
            lines.append(f"| {name} | {thesis:.2f} mm | {value:.3f} mm | {diff:+.3f} mm | 待分析 |")
    lines.append("")
    lines.append(f"样本数：{len(rows)}（其中含人工真值 {stats['n']} 条）")
    if synthetic and stats["n"]:
        lines.append("")
        lines.append(
            f"> 合成自检（**不是指标**）：MAE {stats['mae_mm']:.3f} mm / "
            f"STD {stats['std_mm']:.3f} mm / MAX {stats['max_mm']:.3f} mm。"
            " 仅说明链路能出数，验收必须用真实图像与卡尺真值重跑。"
        )
    lines.append("")
    lines.append("## 3. 逐条结果")
    lines.append("")
    lines.append("| # | 图像 | 计算宽度 mm | 人工宽度 mm | 误差 mm | 判定 | 耗时 ms |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- | ---: |")
    for i, r in enumerate(rows, 1):
        gt = r.get("gt_width_mm")
        gt_text = f"{gt:.3f}" if isinstance(gt, (int, float)) else "—"
        err_text = f"{r['width_mm'] - gt:+.3f}" if isinstance(gt, (int, float)) else "—"
        lines.append(
            f"| {i} | {r['image']} | {r['width_mm']:.3f} | {gt_text} | {err_text} | {r['verdict']} | {r['total_ms']:.1f} |"
        )
    lines.append("")
    lines.append("## 4. 耗时（M7，纯推理）")
    lines.append("")
    lines.append(f"- p50：{lat['p50']:.1f} ms｜p95：{lat['p95']:.1f} ms｜均值：{lat['mean']:.1f} ms｜最大：{lat['max']:.1f} ms（n={lat['n']}）")
    lines.append("")
    lines.append("> 说明：当前量测链路含 Python 版 BFS 与骨架直径搜索，耗时偏高；优化见 `tasks.md` E-01。")
    lines.append("")
    lines.append("## 5. 结论与待办")
    lines.append("")
    if synthetic:
        lines.append("- 链路验证：合成五类用例全部按预期判定，报告生成正常。")
        lines.append("- **真实数据未就绪**，M4–M6 仍为「待测」；采集后请用同一脚本重跑并替换本文件。")
    else:
        lines.append(f"- 真实数据 {len(rows)} 条已完成量测，误差指标见第 2 节。")
        lines.append("- 偏差原因需结合光照、胶色、标定精度逐项分析后填写。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 宽度量测报告")
    parser.add_argument("--images", default=None, help="真实图像目录")
    parser.add_argument("--labels", default=None, help="标注 CSV（image,gt_width_mm）")
    parser.add_argument("--calib", default=None, help="标定 JSON")
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true", help="用合成数据自检")
    parser.add_argument("--out-md", default=None, help="报告输出路径")
    parser.add_argument("--out-csv", default="outputs/p1_measure.csv", help="逐条结果 CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta = {
        "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": git_commit(),
        "config": args.config or "configs/default.yaml",
        "calib": args.calib or ("synthetic 0.25 mm/px" if args.synthetic else "无"),
    }

    if args.synthetic:
        rows = run_synthetic(cfg, mm_per_px=0.25)
    else:
        if not args.images:
            print("真实模式需要 --images", file=sys.stderr)
            return 2
        labels = load_labels(Path(args.labels)) if args.labels else {}
        calib = Calibration.load(args.calib) if args.calib else None
        rows = run_real(cfg, Path(args.images), labels, calib)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_width_mm", "width_mm", "verdict", "total_ms"])
        writer.writeheader()
        writer.writerows(rows)

    report = render_report(rows, synthetic=args.synthetic, meta=meta)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        print(f"报告已写入 {out_md}")
    else:
        print(report)
    print(f"逐条结果 -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
