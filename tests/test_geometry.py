"""几何与 anchor 工具测试（论文 3.1.2–3.1.4）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo import geometry as g  # noqa: E402


class TestIoUFamily(unittest.TestCase):
    def test_iou_identical(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        self.assertAlmostEqual(float(g.bbox_iou(a, np.array([[0, 0, 10, 10]]))[0]), 1.0, places=6)

    def test_iou_disjoint(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        self.assertAlmostEqual(float(g.bbox_iou(a, np.array([[20, 20, 30, 30]]))[0]), 0.0, places=6)

    def test_iou_known_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        # 交 50，并 150 -> 1/3
        self.assertAlmostEqual(float(g.bbox_iou(a, np.array([[5, 0, 15, 10]]))[0]), 1 / 3, places=6)

    def test_giou_disjoint_negative(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        # 外接框 30x30=900，并集 200 -> giou = -(900-200)/900
        self.assertAlmostEqual(
            float(g.bbox_giou(a, np.array([[20, 20, 30, 30]]))[0]), -700 / 900, places=6
        )

    def test_diou_equals_ciou_for_same_aspect(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([[5, 5, 15, 15]])
        self.assertAlmostEqual(float(g.bbox_diou(a, b)[0]), float(g.bbox_ciou(a, b)[0]), places=6)

    def test_ciou_penalizes_aspect_ratio(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([[0, 0, 20, 10]])  # 同面积不同长宽比
        self.assertLess(float(g.bbox_ciou(a, b)[0]), float(g.bbox_diou(a, b)[0]))


class TestNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float
        )
        self.scores = np.array([0.9, 0.8, 0.7])

    def test_nms_suppresses_overlap(self):
        keep = g.nms(self.boxes, self.scores, 0.3)
        self.assertEqual(sorted(keep.tolist()), [0, 2])

    def test_ciou_nms_suppresses_overlap(self):
        keep = g.ciou_nms(self.boxes, self.scores, 0.3)
        self.assertEqual(sorted(keep.tolist()), [0, 2])

    def test_nms_empty(self):
        self.assertEqual(len(g.nms(np.zeros((0, 4)), np.zeros((0,)))), 0)


class TestAnchors(unittest.TestCase):
    def test_kmeans_returns_k_sorted(self):
        rng = np.random.default_rng(0)
        wh = np.concatenate(
            [rng.normal([20, 60], 2, size=(50, 2)), rng.normal([100, 40], 3, size=(50, 2))]
        )
        anchors = g.kmeans_anchors(wh, k=4, seed=0)
        self.assertEqual(anchors.shape, (4, 2))
        areas = anchors[:, 0] * anchors[:, 1]
        self.assertTrue(np.all(np.diff(areas) >= 0), "先验框应按面积升序")

    def test_encode_decode_roundtrip(self):
        anchors = np.array([[10, 10, 50, 50], [20, 20, 80, 60]], dtype=float)
        targets = np.array([[12, 14, 55, 58], [18, 25, 90, 70]], dtype=float)
        deltas = g.encode_boxes(anchors, targets)
        decoded = g.decode_boxes(anchors, deltas)
        np.testing.assert_allclose(decoded, targets, atol=1e-6)

    def test_generate_anchors_shape(self):
        anchors = g.generate_anchors(4, 5, 16, np.array([[32, 32], [64, 64]], dtype=float))
        self.assertEqual(anchors.shape, (4 * 5 * 2, 4))


if __name__ == "__main__":
    unittest.main()
