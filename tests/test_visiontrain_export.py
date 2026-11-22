"""VisionTrain 胶条/背景标注转换器测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algo.config import load_config  # noqa: E402
from algo.visiontrain_export import (  # noqa: E402
    GLUE_CATEGORY_ID,
    export_visiontrain_dataset,
    mask_to_polygons,
)


def make_source_dataset(root: Path) -> None:
    """构造一张纯背景图和一张胶条图的最小导出输入。"""

    (root / "images").mkdir(parents=True)
    (root / "masks" / "glue").mkdir(parents=True)
    image = np.full((80, 120), 64, dtype=np.uint8)
    glue_mask = np.zeros_like(image)
    cv2.fillPoly(
        glue_mask,
        [np.array([[30, 30], [70, 32], [68, 55], [28, 53]], dtype=np.int32)],
        255,
    )
    cv2.imwrite(str(root / "images" / "background.png"), image)
    cv2.imwrite(str(root / "images" / "glue.png"), image)
    cv2.imwrite(
        str(root / "masks" / "glue" / "background.png"),
        np.zeros_like(image),
    )
    cv2.imwrite(str(root / "masks" / "glue" / "glue.png"), glue_mask)
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "path",
                "scenario",
                "split",
                "acceptance_eligible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "background",
                "path": "images/background.png",
                "scenario": "background",
                "split": "test",
                "acceptance_eligible": "0",
            }
        )
        writer.writerow(
            {
                "image_id": "glue",
                "path": "images/glue.png",
                "scenario": "break",
                "split": "train",
                "acceptance_eligible": "0",
            }
        )


class TestVisionTrainExport(unittest.TestCase):
    def test_mask_to_polygons_preserves_bbox(self):
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[10:21, 15:31] = 255
        polygons = mask_to_polygons(
            mask,
            contour_epsilon_px=0.5,
            min_contour_area_px=1.0,
        )
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0]["bbox"], [15.0, 10.0, 16.0, 11.0])
        self.assertGreaterEqual(len(polygons[0]["segmentation"][0]), 6)

    def test_export_matches_visiontrain_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            output = Path(temp_dir) / "visiontrain"
            make_source_dataset(source)
            summary = export_visiontrain_dataset(
                source,
                output,
                load_config(),
                created_at="2026-11-21 01:00:00",
            )

            self.assertEqual(summary.image_count, 2)
            self.assertEqual(summary.glue_image_count, 1)
            self.assertEqual(summary.background_image_count, 1)
            self.assertEqual(summary.annotation_count, 1)
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            self.assertTrue((output / "dataset.json").is_file())
            self.assertTrue((output / "background.json").is_file())
            self.assertTrue((output / "glue.json").is_file())

            metadata = json.loads((output / "dataset.json").read_text("utf-8"))
            self.assertEqual(metadata["info"]["source"], "synthetic")
            self.assertFalse(metadata["acceptance_eligible"])
            self.assertEqual(len(metadata["images"]), 2)
            self.assertEqual(
                [category["name"] for category in metadata["categories"]],
                ["background", "glue"],
            )
            self.assertTrue(all(image["labeled"] for image in metadata["images"]))
            self.assertIn("合成数据仅用于", metadata["warning"])
            self.assertEqual(
                {
                    image["file_name"]: image["image_type"]
                    for image in metadata["images"]
                },
                {"background.png": "test", "glue.png": "train"},
            )

            background = json.loads(
                (output / "background.json").read_text("utf-8")
            )
            glue = json.loads((output / "glue.json").read_text("utf-8"))
            self.assertEqual(background, [])
            self.assertEqual(glue[0]["category_id"], GLUE_CATEGORY_ID)
            self.assertEqual(glue[0]["labelname"], "glue")
            self.assertEqual(glue[0]["type"], "polygon")

    def test_polygon_round_trip_has_high_iou(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            output = Path(temp_dir) / "visiontrain"
            make_source_dataset(source)
            export_visiontrain_dataset(source, output, load_config())

            source_mask = cv2.imread(
                str(source / "masks" / "glue" / "glue.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            records = json.loads((output / "glue.json").read_text("utf-8"))
            restored = np.zeros_like(source_mask)
            for record in records:
                for polygon in record["segmentation"]:
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(restored, [np.rint(points).astype(np.int32)], 1)
            source_binary = source_mask > 0
            restored_binary = restored > 0
            iou = np.logical_and(source_binary, restored_binary).sum() / max(
                1,
                np.logical_or(source_binary, restored_binary).sum(),
            )
            self.assertGreater(iou, 0.90)

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            output = Path(temp_dir) / "visiontrain"
            make_source_dataset(source)
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_visiontrain_dataset(source, output, load_config())

    def test_subset_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            output = Path(temp_dir) / "visiontrain"
            make_source_dataset(source)
            summary = export_visiontrain_dataset(
                source,
                output,
                load_config(),
                image_ids=["glue"],
            )
            self.assertEqual(summary.image_count, 1)
            self.assertTrue((output / "glue.png").is_file())
            self.assertFalse((output / "background.png").exists())
            metadata = json.loads((output / "dataset.json").read_text("utf-8"))
            self.assertEqual(
                [image["file_name"] for image in metadata["images"]],
                ["glue.png"],
            )


if __name__ == "__main__":
    unittest.main()
