"""流水线与判定测试：合成四类用例应命中对应判定分支。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.calib import Calibration  # noqa: E402
from algo.config import load_config  # noqa: E402
from algo.decide import decide  # noqa: E402
from algo.pipeline import GlueInspector  # noqa: E402
from algo.synth import generate_case  # noqa: E402
from algo.types import MeasureResult, ReflectionResult  # noqa: E402


def make_cfg():
    cfg = load_config()
    cfg["reflect"]["type"] = "heuristic"
    return cfg


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def test_ok(self):
        verdict = decide(MeasureResult(width_mm=10.0), 1, ReflectionResult(), self.cfg)
        self.assertEqual(verdict.code, "OK")

    def test_warn_wide(self):
        verdict = decide(MeasureResult(width_mm=10.6), 1, ReflectionResult(), self.cfg)
        self.assertEqual(verdict.code, "WARN_WIDE")

    def test_alarm_narrow(self):
        verdict = decide(MeasureResult(width_mm=8.8), 1, ReflectionResult(), self.cfg)
        self.assertEqual(verdict.code, "TOO_NARROW")
        self.assertEqual(verdict.severity, "alarm")

    def test_broken_by_roi_count(self):
        verdict = decide(MeasureResult(width_mm=10.0), 2, ReflectionResult(), self.cfg)
        self.assertEqual(verdict.code, "BROKEN")

    def test_reflection_override(self):
        verdict = decide(
            MeasureResult(width_mm=8.8), 1, ReflectionResult(has_reflection=True), self.cfg
        )
        self.assertEqual(verdict.code, "REFLECTION_SUSPECT")
        self.assertEqual(verdict.details.get("pre_override_code"), "TOO_NARROW")

    def test_reflection_without_defect_stays_ok(self):
        verdict = decide(
            MeasureResult(width_mm=10.0), 1, ReflectionResult(has_reflection=True), self.cfg
        )
        self.assertEqual(verdict.code, "OK")

    def test_uncalibrated_skips_width(self):
        verdict = decide(MeasureResult(width_mm=0.0), 1, ReflectionResult(), self.cfg)
        self.assertEqual(verdict.code, "OK")
        self.assertTrue(verdict.details.get("uncalibrated"))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg()
        cls.calib = Calibration(mm_per_px=0.25)

    def run_case(self, kind: str, calib=None):
        inspector = GlueInspector(self.cfg, calib=calib if calib is not None else self.calib)
        case = generate_case(kind, size=(320, 480), width_px=40, seed=0)
        return inspector, inspector.run(case.image)

    def test_normal_ok(self):
        _, result = self.run_case("normal")
        self.assertEqual(result.verdict.code, "OK")
        self.assertEqual(result.measure.n_components, 1)
        self.assertGreater(result.measure.width_mm, 0)

    def test_broken(self):
        _, result = self.run_case("broken")
        self.assertEqual(result.verdict.code, "BROKEN")

    def test_narrow(self):
        _, result = self.run_case("narrow")
        self.assertEqual(result.verdict.code, "TOO_NARROW")
        self.assertLess(result.measure.width_mm, 10.0)

    def test_wide(self):
        _, result = self.run_case("wide")
        self.assertEqual(result.verdict.code, "TOO_WIDE")

    def test_reflection_suspect(self):
        _, result = self.run_case("reflection")
        self.assertTrue(result.reflection.has_reflection)
        self.assertEqual(result.verdict.code, "REFLECTION_SUSPECT")

    def test_uncalibrated_degrades_visibly(self):
        inspector, result = self.run_case("normal", calib=Calibration())
        self.assertEqual(result.measure.width_mm, 0.0)
        self.assertTrue(any("未标定" in note for note in result.degradations))
        self.assertTrue(result.verdict.details.get("uncalibrated"))

    def test_result_serializable(self):
        _, result = self.run_case("normal")
        data = result.as_dict()
        self.assertIn("verdict", data)
        self.assertIn("measure", data)
        self.assertIn("degradations", data)


if __name__ == "__main__":
    unittest.main()
