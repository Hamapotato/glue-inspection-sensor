"""评测指标测试（口径见 docs/algo/p0-spec.md §1）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo.metrics import (  # noqa: E402
    best_iou,
    compare_with_thesis,
    latency_summary,
    mean_iou,
    sample_recall,
    summarize_errors,
    width_errors,
)


class TestBoxMetrics(unittest.TestCase):
    def test_best_iou_perfect(self):
        self.assertAlmostEqual(best_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]]), 1.0)

    def test_best_iou_no_prediction(self):
        self.assertEqual(best_iou([], [[0, 0, 10, 10]]), 0.0)

    def test_mean_iou_across_images(self):
        preds = [[[0, 0, 10, 10]], [[0, 0, 10, 10]]]
        gts = [[[0, 0, 10, 10]], [[5, 0, 15, 10]]]
        # 第二张交 50 并 150 -> 1/3
        self.assertAlmostEqual(mean_iou(preds, gts), (1.0 + 1 / 3) / 2, places=6)

    def test_sample_recall(self):
        preds = [[[0, 0, 10, 10]], [[0, 0, 10, 10]]]
        gts = [[[0, 0, 10, 10]], [[5, 0, 15, 10]]]
        self.assertAlmostEqual(sample_recall(preds, gts, thr=0.5), 0.5)


class TestWidthMetrics(unittest.TestCase):
    def test_width_errors_and_summary(self):
        errors = width_errors([10.1, 9.8, 10.5], [10.0, 10.0, 10.0])
        np.testing.assert_allclose(errors, [0.1, -0.2, 0.5], atol=1e-9)
        stats = summarize_errors(errors)
        self.assertAlmostEqual(stats["mae_mm"], 0.2666666, places=5)
        self.assertAlmostEqual(stats["max_mm"], 0.5, places=6)
        self.assertEqual(stats["n"], 3)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            width_errors([1.0, 2.0], [1.0])

    def test_empty_errors(self):
        stats = summarize_errors(np.array([]))
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["mae_mm"], 0.0)


class TestLatency(unittest.TestCase):
    def test_percentiles(self):
        stats = latency_summary(list(range(1, 101)))
        self.assertAlmostEqual(stats["p50"], 50.5, places=6)
        self.assertAlmostEqual(stats["max"], 100.0)
        self.assertEqual(stats["n"], 100)

    def test_empty(self):
        self.assertEqual(latency_summary([])["n"], 0)


class TestCompareWithThesis(unittest.TestCase):
    def test_three_columns(self):
        rows = compare_with_thesis({"mae_mm": 0.30}, {"mae_mm": 0.23})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["diff"], 0.07, places=6)
        self.assertEqual(rows[0]["reason"], "")


if __name__ == "__main__":
    unittest.main()
