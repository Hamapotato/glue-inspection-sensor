"""胶条分割接口与后处理测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from algo.config import load_config  # noqa: E402
from algo.segmentation import (  # noqa: E402
    CallableSegmenter,
    MaskSegmenter,
    SegmentationOutput,
    postprocess_segmentation,
)


class TestSegmentation(unittest.TestCase):
    def test_mask_segmenter_requires_matching_shape(self):
        segmenter = MaskSegmenter(np.zeros((10, 20), dtype=np.uint8))
        with self.assertRaises(ValueError):
            segmenter.predict(np.zeros((20, 20), dtype=np.uint8))

    def test_probability_is_binarized_and_postprocessed(self):
        cfg = load_config()
        cfg["segmentation.threshold"] = 0.6
        cfg["segmentation.morph.ksize"] = 1
        output = SegmentationOutput(
            mask=np.array([[0.2, 0.7], [0.9, 0.1]], dtype=np.float32),
            source="test_model",
            threshold=0.5,
        )
        processed = postprocess_segmentation(output, cfg)
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(processed.mask, expected)
        self.assertEqual(processed.source, "test_model")

    def test_callable_segmenter_accepts_array(self):
        segmenter = CallableSegmenter(
            lambda image: np.full(image.shape[:2], 255, dtype=np.uint8),
            source="function_model",
        )
        output = segmenter.predict(np.zeros((8, 12), dtype=np.uint8))
        self.assertEqual(output.source, "function_model")
        self.assertEqual(output.mask.shape, (8, 12))


if __name__ == "__main__":
    unittest.main()
