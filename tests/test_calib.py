"""标定测试（论文 4.3）：内参恢复、重投影误差、毫米/像素换算。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algo.calib import (  # noqa: E402
    Calibration,
    MappingTable,
    calibrate_from_points,
    scale_from_homography,
    two_point_scale,
)


def make_board(cols: int = 9, rows: int = 6, square_mm: float = 5.0) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


class TestCalibration(unittest.TestCase):
    def test_recover_intrinsics(self):
        K_true = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]], dtype=np.float64)
        objp = make_board()
        objs, imgs = [], []
        for i in range(8):
            rvec = np.array([0.10 * ((i % 3) - 1), 0.10 * ((i // 3) - 1), 0.02 * i], dtype=np.float64)
            tvec = np.array([0.0, 0.0, 400.0 + 10.0 * i], dtype=np.float64)
            pts, _ = cv2.projectPoints(objp, rvec, tvec, K_true, None)
            objs.append(objp)
            imgs.append(pts.reshape(-1, 2))

        calib = calibrate_from_points(objs, imgs, (640, 480))

        self.assertAlmostEqual(float(calib.K[0, 0]), 800.0, delta=800.0 * 0.03)
        self.assertAlmostEqual(float(calib.K[1, 1]), 800.0, delta=800.0 * 0.03)
        self.assertLess(calib.rms, 0.5)
        self.assertEqual(len(calib.per_view_errors), 8)
        self.assertTrue(calib.is_calibrated)
        self.assertGreater(calib.mm_per_px, 0)

    def test_px_to_mm(self):
        calib = Calibration(mm_per_px=0.25)
        self.assertAlmostEqual(calib.px_to_mm(40), 10.0)
        self.assertTrue(calib.is_calibrated)

    def test_uncalibrated(self):
        calib = Calibration()
        self.assertEqual(calib.px_to_mm(40), 0.0)
        self.assertFalse(calib.is_calibrated)

    def test_scale_from_homography(self):
        # 4 px = 1 mm  =>  0.25 mm/px
        H = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 1.0]])
        scale = scale_from_homography(H, (640, 480))
        self.assertAlmostEqual(scale, 0.25, places=6)

    def test_two_point_scale(self):
        # 论文式 (4-13)~(4-15)：10 mm 对应 40 px -> 0.25 mm/px
        scale = two_point_scale([0, 0, 0], [10, 0, 0], [0, 0], [40, 0])
        self.assertAlmostEqual(scale, 0.25, places=9)

    def test_two_point_scale_zero_distance_raises(self):
        with self.assertRaises(ValueError):
            two_point_scale([0, 0, 0], [10, 0, 0], [5, 5], [5, 5])

    def test_mapping_table_roundtrip(self):
        table = MappingTable()
        table.add("pos-01", 0.25)
        table.add("pos-02", 0.251)
        self.assertAlmostEqual(table.get("pos-01"), 0.25)
        self.assertIsNone(table.get("missing"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            table.save(path)
            loaded = MappingTable.load(path)
        self.assertEqual(loaded.entries, table.entries)

    def test_save_load_roundtrip(self):
        calib = Calibration(
            K=np.eye(3), dist=np.zeros(5), rms=0.1, mm_per_px=0.25, image_size=(640, 480),
            per_view_errors=[0.1, 0.2],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.json"
            calib.save(path)
            loaded = Calibration.load(path)
        self.assertAlmostEqual(loaded.mm_per_px, 0.25)
        self.assertEqual(loaded.image_size, (640, 480))
        np.testing.assert_allclose(loaded.K, np.eye(3))


if __name__ == "__main__":
    unittest.main()
