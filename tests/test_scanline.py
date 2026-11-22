"""旋转扫描矩形、边缘点和中心线测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo.config import load_config  # noqa: E402
from algo.scanline import scan_glue_strip  # noqa: E402
from algo.synthetic_dataset import SCENARIOS, generate_sample  # noqa: E402


def single_scenario_config(scenario: str):
    cfg = load_config()
    cfg["synthetic_dataset"]["image_size"] = [320, 480]
    cfg["synthetic_dataset"]["scenario_weights"] = {
        name: 1.0 if name == scenario else 0.0 for name in SCENARIOS
    }
    return cfg


class TestScanline(unittest.TestCase):
    def test_horizontal_strip_has_rectangles_edges_and_centerline(self):
        cfg = load_config()
        cfg["measure.min_area_px"] = 10
        mask = np.zeros((120, 220), dtype=np.uint8)
        mask[55:66, 25:195] = 255

        result = scan_glue_strip(mask, cfg, mm_per_px=0.1)

        self.assertGreater(len(result.windows), 5)
        self.assertTrue(all(window.valid for window in result.windows))
        self.assertAlmostEqual(result.width_px, 11.0, delta=1.5)
        self.assertEqual(len(result.centerline_segments), 1)
        self.assertGreater(len(result.centerline_segments[0]), 5)
        self.assertLess(result.curve_deviation_px, 2.0)
        for window in result.windows:
            self.assertIsNotNone(window.left)
            self.assertIsNotNone(window.right)
            self.assertIsNotNone(window.midpoint)
            self.assertGreaterEqual(
                window.window_width_px,
                3.0 * window.width_px,
            )

    def test_missing_edge_windows_detect_break(self):
        cfg = single_scenario_config("break")
        sample = generate_sample(cfg["synthetic_dataset"], 501)

        result = scan_glue_strip(sample.glue_mask, cfg, mm_per_px=0.25)

        self.assertGreater(result.break_windows, 0)
        self.assertGreater(result.missing_edge_windows, 0)
        self.assertGreaterEqual(len(result.centerline_segments), 2)

    def test_curve_centerline_deviation_is_reported(self):
        cfg = single_scenario_config("curve")
        sample = generate_sample(cfg["synthetic_dataset"], 505)

        result = scan_glue_strip(sample.glue_mask, cfg, mm_per_px=0.25)

        self.assertGreater(result.curve_deviation_px, 10.0)
        self.assertGreater(result.curve_deviation_mm, 2.5)
        self.assertEqual(result.break_windows, 0)


if __name__ == "__main__":
    unittest.main()
