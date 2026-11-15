"""预处理与分割测试（论文 4.1–4.2）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo.preprocess import (  # noqa: E402
    _threshold_lambda,
    double_peak_threshold,
    global_threshold,
    median_denoise,
    morph_open_close,
    otsu_threshold,
    segment,
    wavelet_denoise,
)


class TestDenoise(unittest.TestCase):
    def test_wavelet_reduces_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        out = wavelet_denoise(noise, level=2, mode="hard")
        self.assertEqual(out.shape, noise.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(float(out.std()), float(noise.std()))

    def test_threshold_formula(self):
        # n <= 32 时论文式 (4-4) 取 0
        small = np.linspace(-0.2, 0.2, 16).reshape(4, 4)
        self.assertEqual(_threshold_lambda(small), 0.0)
        # n > 32 时为正，且细节系数越强阈值越大
        weak = np.full((8, 8), 0.01)
        strong = np.full((8, 8), 0.10)
        self.assertGreater(_threshold_lambda(strong), _threshold_lambda(weak))
        self.assertGreater(_threshold_lambda(weak), 0.0)

    def test_wavelet_lossless_when_threshold_zero(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        out = wavelet_denoise(img, level=2, threshold=0.0)
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)

    def test_median_keeps_shape(self):
        img = np.random.default_rng(1).integers(0, 256, (64, 64), dtype=np.uint8)
        out = median_denoise(img, ksize=5)
        self.assertEqual(out.shape, img.shape)


class TestSegment(unittest.TestCase):
    def setUp(self):
        img = np.full((80, 120), 30, dtype=np.uint8)
        img[20:60, 30:90] = 200
        self.img = img

    def test_otsu_separates(self):
        thr, mask = otsu_threshold(self.img)
        # 阈值语义为"像素值 > thr 判为前景"，故 thr 落在背景值上即正确
        self.assertGreaterEqual(thr, 30)
        self.assertLess(thr, 200)
        self.assertGreater(int((mask > 0).sum()), 0)
        self.assertFalse(bool(mask[0, 0]), "背景不应被判为前景")

    def test_global_threshold(self):
        thr, mask = global_threshold(self.img)
        self.assertGreater(thr, 0)
        self.assertEqual(mask.dtype, np.uint8)

    def test_double_peak(self):
        thr, mask = double_peak_threshold(self.img)
        self.assertTrue(0 <= thr <= 255)
        self.assertGreater(int((mask > 0).sum()), 0)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            segment(self.img, "nope")


class TestMorphology(unittest.TestCase):
    def test_close_fills_hole(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        mask[28:32, 28:32] = 0  # 4x4 孔洞
        out = morph_open_close(mask, ksize=7, order="close")
        self.assertTrue((out[28:32, 28:32] > 0).all())

    def test_open_removes_single_pixel(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[20, 20] = 255
        mask[5:20, 5:20] = 255
        out = morph_open_close(mask, ksize=5, order="open")
        self.assertEqual(int(out[20, 20]), 0)


if __name__ == "__main__":
    unittest.main()
