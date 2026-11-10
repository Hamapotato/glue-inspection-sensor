"""评测指标（定义见 docs/algo/p0-spec.md §1）。

注意口径差异：
- ``mean_iou``：RoI 定位平均 IoU（M1），不是 mAP；
- ``sample_recall``：样本级召回（M2），一张图至少一个框达标即算检出；
- 宽度误差 ``mae/std/max``（M4–M6）以毫米为单位，真值来自人工测量。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .geometry import bbox_iou


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """单对框的 IoU。"""
    return float(bbox_iou(np.asarray(box_a, dtype=np.float64), np.asarray([box_b], dtype=np.float64))[0])


def best_iou(preds: Sequence[Sequence[float]], gts: Sequence[Sequence[float]]) -> float:
    """一张图内预测框与真值框的最佳匹配 IoU（单类）。

    多预测框取与任一真值框的最大值；无预测或无真值时返回 0。
    """
    if len(preds) == 0 or len(gts) == 0:
        return 0.0
    pred_arr = np.asarray(preds, dtype=np.float64).reshape(-1, 4)
    best = 0.0
    for gt in gts:
        ious = bbox_iou(np.asarray(gt, dtype=np.float64), pred_arr)
        best = max(best, float(np.max(ious)) if ious.size else 0.0)
    return best


def mean_iou(
    preds_per_image: Iterable[Sequence[Sequence[float]]],
    gts_per_image: Iterable[Sequence[Sequence[float]]],
) -> float:
    """M1：逐图最佳 IoU 的算术平均。"""
    values = [best_iou(p, g) for p, g in zip(preds_per_image, gts_per_image)]
    return float(np.mean(values)) if values else 0.0


def sample_recall(
    preds_per_image: Iterable[Sequence[Sequence[float]]],
    gts_per_image: Iterable[Sequence[Sequence[float]]],
    thr: float = 0.5,
) -> float:
    """M2：样本级召回（best IoU ≥ thr 记为检出）。"""
    values = [best_iou(p, g) >= thr for p, g in zip(preds_per_image, gts_per_image)]
    return float(np.mean(values)) if values else 0.0


def width_errors(pred_mm: Sequence[float], gt_mm: Sequence[float]) -> np.ndarray:
    """宽度误差数组（预测 − 真值）。"""
    pred = np.asarray(pred_mm, dtype=np.float64)
    gt = np.asarray(gt_mm, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("预测与真值长度不一致")
    return pred - gt


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    """M4–M6：MAE、标准差、最大绝对误差。"""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"mae_mm": 0.0, "std_mm": 0.0, "max_mm": 0.0, "n": 0}
    abs_err = np.abs(errors)
    return {
        "mae_mm": float(np.mean(abs_err)),
        "std_mm": float(np.std(errors)),
        "max_mm": float(np.max(abs_err)),
        "n": int(errors.size),
    }


def latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    """M7：耗时统计（p50 / p95 / 均值 / 最大值）。"""
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }


def compare_with_thesis(
    reproduced: dict[str, float],
    thesis: dict[str, float],
) -> list[dict[str, float | str]]:
    """生成"论文值 / 复现值 / 偏差"三列对照（AGENTS.md §6.3 要求）。

    偏差原因列留空，由报告撰写者填写；本函数只负责数值。
    """
    rows: list[dict[str, float | str]] = []
    for key, thesis_value in thesis.items():
        if key not in reproduced:
            continue
        value = reproduced[key]
        diff = value - thesis_value
        ratio = diff / thesis_value if thesis_value else float("nan")
        rows.append(
            {
                "metric": key,
                "thesis": thesis_value,
                "reproduced": value,
                "diff": diff,
                "ratio": ratio,
                "reason": "",
            }
        )
    return rows
