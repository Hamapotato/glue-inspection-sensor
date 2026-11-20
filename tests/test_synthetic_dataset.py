"""VisionTrain 合成数据集生成器测试。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algo.config import load_config  # noqa: E402
from algo.synthetic_dataset import (  # noqa: E402
    DEFECT_CLASS_IDS,
    SCENARIOS,
    generate_dataset,
)


def make_config(
    scenario: str = "ok",
    *,
    samples: int = 6,
    frames_per_object: int = 1,
):
    cfg = load_config()
    settings = cfg["synthetic_dataset"]
    settings["image_size"] = [96, 128]
    settings["samples"] = samples
    settings["seed"] = 1234
    settings["frames_per_object"] = frames_per_object
    settings["centerline_points"] = 80
    settings["scenario_weights"] = {
        name: 1.0 if name == scenario else 0.0 for name in SCENARIOS
    }
    return cfg


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSyntheticDataset(unittest.TestCase):
    def test_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            summary = generate_dataset(root, make_config(samples=6))

            self.assertEqual(summary.sample_count, 6)
            self.assertTrue((root / "manifest.csv").is_file())
            self.assertTrue((root / "dataset.json").is_file())
            self.assertEqual(len(list((root / "images").glob("*.png"))), 6)
            self.assertEqual(len(list((root / "masks/glue").glob("*.png"))), 6)
            self.assertEqual(len(list((root / "masks/defects").glob("*.png"))), 6)
            self.assertEqual(len(list((root / "labels").glob("*.json"))), 6)

            meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["source"], "synthetic")
            self.assertFalse(meta["acceptance_eligible"])
            self.assertIn("不得作为真实指标", meta["warning"])

    def test_image_and_masks_are_aligned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            generate_dataset(root, make_config(samples=4))
            for image_path in sorted((root / "images").glob("*.png")):
                image_id = image_path.stem
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                glue = cv2.imread(
                    str(root / "masks" / "glue" / f"{image_id}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                defects = cv2.imread(
                    str(root / "masks" / "defects" / f"{image_id}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                hard = cv2.imread(
                    str(root / "masks" / "hard_background" / f"{image_id}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertEqual(image.shape, glue.shape)
                self.assertEqual(image.shape, defects.shape)
                self.assertEqual(image.shape, hard.shape)
                self.assertGreater(int((glue > 0).sum()), 0)
                self.assertTrue(set(np.unique(defects)).issubset(set(DEFECT_CLASS_IDS.values())))

    def test_deterministic_for_same_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            cfg = make_config(scenario="bubble", samples=3)
            generate_dataset(first, cfg, seed=777)
            generate_dataset(second, cfg, seed=777)

            for relative in ("images", "masks/glue", "masks/defects", "labels"):
                first_files = sorted((first / relative).glob("*"))
                second_files = sorted((second / relative).glob("*"))
                self.assertEqual(len(first_files), len(second_files))
                for left, right in zip(first_files, second_files):
                    self.assertEqual(file_digest(left), file_digest(right))

    def test_different_seed_changes_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            cfg = make_config(samples=2)
            generate_dataset(first, cfg, seed=100)
            generate_dataset(second, cfg, seed=101)
            first_image = sorted((first / "images").glob("*.png"))[0]
            second_image = sorted((second / "images").glob("*.png"))[0]
            self.assertNotEqual(file_digest(first_image), file_digest(second_image))

    def test_break_scenario_creates_break_pixels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            generate_dataset(root, make_config(scenario="break", samples=3))
            for mask_path in sorted((root / "masks/defects").glob("*.png")):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                self.assertGreater(
                    int((mask == DEFECT_CLASS_IDS["break"]).sum()),
                    0,
                )

    def test_group_split_has_no_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            generate_dataset(
                root,
                make_config(samples=9, frames_per_object=3),
                seed=55,
            )
            split_ids: list[str] = []
            for split in ("train", "val", "test"):
                ids = (root / "splits" / f"{split}.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                split_ids.extend(ids)
            self.assertEqual(len(split_ids), 9)
            self.assertEqual(len(set(split_ids)), 9)

            with open(root / "manifest.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            object_splits: dict[str, set[str]] = {}
            for row in rows:
                object_splits.setdefault(row["object_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in object_splits.values()))

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            root.mkdir()
            (root / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_dataset(root, make_config(samples=1))


if __name__ == "__main__":
    unittest.main()
