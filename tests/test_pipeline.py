"""分割模型主线与判定测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.decide import decide  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402
from algo.segmentation import (  # noqa: E402
    MaskSegmenter,
    ModelNotAvailableError,
)
from algo.synthetic_dataset import SCENARIOS, generate_sample  # noqa: E402
from algo.types import MeasureResult, ScanWindow  # noqa: E402


def measured(width_mm: float) -> MeasureResult:
    return MeasureResult(
        width_mm=width_mm,
        min_width_mm=width_mm,
        max_width_mm=width_mm,
        scan_windows=[ScanWindow(0, 10, 10, 0, 20, 60, 10)],
    )


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_ok(self):
        verdict = decide(measured(10.0), self.cfg)
        self.assertEqual(verdict.code, "OK")

    def test_warn_wide(self):
        verdict = decide(measured(10.6), self.cfg)
        self.assertEqual(verdict.code, "WARN_WIDE")

    def test_alarm_narrow(self):
        verdict = decide(measured(8.8), self.cfg)
        self.assertEqual(verdict.code, "TOO_NARROW")
        self.assertEqual(verdict.severity, "alarm")

    def test_no_scan_windows_is_broken(self):
        verdict = decide(MeasureResult(width_mm=10.0), self.cfg)
        self.assertEqual(verdict.code, "BROKEN")

    def test_uncalibrated_skips_width(self):
        measure = measured(0.0)
        verdict = decide(measure, self.cfg)
        self.assertEqual(verdict.code, "OK")
        self.assertTrue(verdict.details.get("uncalibrated"))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.calib = Calibration(mm_per_px=0.25)

    def run_dataset_case(self, scenario: str, seed: int = 500):
        settings = self.cfg["synthetic_dataset"]
        settings["image_size"] = [320, 480]
        settings["scenario_weights"] = {
            name: 1.0 if name == scenario else 0.0 for name in SCENARIOS
        }
        sample = generate_sample(settings, seed)
        inspector = GlueInspector(
            self.cfg,
            segmenter=MaskSegmenter(sample.glue_mask),
            calib=self.calib,
        )
        return inspector.run(sample.image)

    def test_normal_ok(self):
        result = self.run_dataset_case("glue", seed=500)
        self.assertEqual(result.verdict.code, "OK")
        self.assertEqual(result.measure.n_components, 1)
        self.assertGreater(result.measure.width_mm, 0)

    def test_break_detected_by_missing_scan_edges(self):
        result = self.run_dataset_case("break", seed=501)
        self.assertEqual(result.verdict.code, "BROKEN")
        self.assertGreater(result.measure.break_windows, 0)

    def test_narrow(self):
        result = self.run_dataset_case("narrow", seed=502)
        self.assertEqual(result.verdict.code, "TOO_NARROW")

    def test_wide(self):
        result = self.run_dataset_case("wide", seed=503)
        self.assertEqual(result.verdict.code, "TOO_WIDE")

    def test_width_unstable(self):
        result = self.run_dataset_case("width_unstable", seed=500)
        self.assertEqual(result.verdict.code, "WIDTH_UNSTABLE")

    def test_curve(self):
        result = self.run_dataset_case("curve", seed=505)
        self.assertEqual(result.verdict.code, "BENT")

    def test_uncalibrated_degrades_visibly(self):
        settings = self.cfg["synthetic_dataset"]
        settings["image_size"] = [320, 480]
        settings["scenario_weights"] = {
            name: 1.0 if name == "glue" else 0.0 for name in SCENARIOS
        }
        sample = generate_sample(settings, 500)
        inspector = GlueInspector(
            self.cfg,
            segmenter=MaskSegmenter(sample.glue_mask),
            calib=Calibration(),
        )
        result = inspector.run(sample.image)
        self.assertEqual(result.measure.width_mm, 0.0)
        self.assertTrue(any("未标定" in note for note in result.degradations))
        self.assertTrue(result.verdict.details.get("uncalibrated"))

    def test_result_serializable(self):
        result = self.run_dataset_case("glue", seed=500)
        data = result.as_dict()
        self.assertIn("verdict", data)
        self.assertIn("measure", data)
        self.assertIn("segmentation", data)
        self.assertIn("degradations", data)
        self.assertIn("scan_windows", data["measure"])

    def test_missing_model_is_not_silently_replaced(self):
        with self.assertRaises(ModelNotAvailableError):
            GlueInspector.from_config(self.cfg)

    def test_explicit_development_fallback_is_recorded(self):
        cfg = load_config()
        cfg["segmentation.mode"] = "development_threshold"
        inspector = GlueInspector.from_config(cfg)
        self.assertTrue(any("fallback" in note for note in inspector.degradations))

    def test_empty_model_mask_is_broken(self):
        settings = self.cfg["synthetic_dataset"]
        settings["image_size"] = [320, 480]
        settings["scenario_weights"] = {
            name: 1.0 if name == "background" else 0.0 for name in SCENARIOS
        }
        sample = generate_sample(settings, 500)
        inspector = GlueInspector(
            self.cfg,
            segmenter=MaskSegmenter(sample.glue_mask),
            calib=self.calib,
        )
        result = inspector.run(sample.image)
        self.assertEqual(result.verdict.code, "BROKEN")
        self.assertEqual(result.segmentation["foreground_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
