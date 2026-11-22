"""量测链路测试（论文 4.4）：BFS 面积、骨架、宽度。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo.measure import (  # noqa: E402
    connected_components,
    contour_area,
    measure_mask,
    skeleton_length,
    width_from_area_length,
    zhang_suen_thinning,
)
from algo.synth import generate_case  # noqa: E402


class TestComponents(unittest.TestCase):
    def test_two_components(self):
        mask = np.zeros((50, 100), dtype=np.uint8)
        mask[10:20, 10:30] = 255
        mask[30:40, 60:90] = 255
        _, comps = connected_components(mask, connectivity=8, min_area=1)
        self.assertEqual(len(comps), 2)
        self.assertEqual(sorted(c.area for c in comps), [200, 300])

    def test_min_area_filters_noise(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[0, 0] = 255
        mask[10:20, 10:20] = 255
        _, comps = connected_components(mask, connectivity=8, min_area=10)
        self.assertEqual(len(comps), 1)


class TestSkeleton(unittest.TestCase):
    def test_thinning_reduces_pixels(self):
        mask = np.zeros((120, 200), dtype=np.uint8)
        mask[50:70, 20:180] = 255  # 20px 宽的横条
        skel = zhang_suen_thinning(mask)
        self.assertLess(skel.sum(), mask.sum() / 5)
        self.assertGreater(skel.sum(), 100)

    def test_skeleton_length_horizontal(self):
        skel = np.zeros((20, 100), dtype=bool)
        skel[10, 10:90] = True
        self.assertAlmostEqual(skeleton_length(skel), 79.0, places=6)

    def test_width_from_area_length(self):
        self.assertAlmostEqual(width_from_area_length(1000, 100), 10.0)
        self.assertEqual(width_from_area_length(1000, 0), 0.0)


class TestMeasureSynthetic(unittest.TestCase):
    def setUp(self):
        self.case = generate_case("normal", size=(320, 480), width_px=40, seed=0)
        self.mm_per_px = 0.25

    def test_area_close_to_mask(self):
        result = measure_mask(self.case.mask, mm_per_px=self.mm_per_px)
        self.assertEqual(result.n_components, 1)
        self.assertAlmostEqual(result.area_px, float((self.case.mask > 0).sum()), delta=10)

    def test_width_approx_expected(self):
        result = measure_mask(self.case.mask, mm_per_px=self.mm_per_px)
        self.assertAlmostEqual(result.width_px, self.case.width_px, delta=self.case.width_px * 0.2)
        self.assertAlmostEqual(result.width_mm, result.width_px * self.mm_per_px, places=4)

    def test_length_approx_expected(self):
        result = measure_mask(self.case.mask, mm_per_px=None)
        self.assertAlmostEqual(result.length_px, self.case.length_px, delta=self.case.length_px * 0.2)

    def test_contour_area_close(self):
        filled = float((self.case.mask > 0).sum())
        approx = contour_area(self.case.mask)
        self.assertAlmostEqual(approx, filled, delta=filled * 0.05)

    def test_broken_two_components(self):
        case = generate_case("broken", size=(320, 480), width_px=40, seed=1)
        result = measure_mask(case.mask, mm_per_px=self.mm_per_px)
        self.assertGreaterEqual(result.n_components, 2)

    def test_empty_mask(self):
        result = measure_mask(np.zeros((30, 30), dtype=np.uint8))
        self.assertEqual(result.area_px, 0.0)
        self.assertEqual(result.n_components, 0)


if __name__ == "__main__":
    unittest.main()
