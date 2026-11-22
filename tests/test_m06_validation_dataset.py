"""M06 同轨迹验证集生成器测试。"""

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
from algo.m06_validation_dataset import (  # noqa: E402
    SCENARIOS,
    generate_m06_validation_dataset,
)


def make_config() -> dict:
    cfg = load_config()
    settings = cfg["m06_validation"]
    settings["image_size"] = [192, 256]
    settings["total_samples"] = 6
    settings["counts"] = {name: 1 for name in SCENARIOS}
    settings["fixed_trajectory"]["centerline_points"] = 121
    settings["fixed_trajectory"]["amplitude_px"] = 10.0
    settings["fixed_trajectory"]["base_width_px"] = 16.0
    settings["variants"]["break_length_px"] = [30.0, 30.0]
    settings["variants"]["bend_amplitude_px"] = [8.0, 8.0]
    settings["variants"]["bend_span_ratio"] = [0.3, 0.3]
    settings["min_component_area_px"] = 5
    return cfg


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM06ValidationDataset(unittest.TestCase):
    def test_artifacts_and_expected_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            summary = generate_m06_validation_dataset(root, make_config(), seed=42)

            self.assertEqual(summary.sample_count, 6)
            self.assertEqual(summary.scenario_counts, {name: 1 for name in SCENARIOS})
            self.assertTrue((root / "dataset.json").is_file())
            self.assertTrue((root / "reference.json").is_file())
            self.assertTrue((root / "preview_overview.png").is_file())
            self.assertEqual(len(list((root / "images").glob("*.png"))), 6)
            self.assertEqual(len(list((root / "labels").glob("*.json"))), 6)

            with open(root / "manifest.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["expected_result"] for row in rows],
                ["OK", "OK", "BROKEN", "TOO_NARROW", "TOO_WIDE", "BENT"],
            )
            self.assertEqual(sum(int(row["is_reference"]) for row in rows), 1)
            self.assertTrue(all(row["acceptance_eligible"] == "0" for row in rows))
            reference = json.loads(
                (root / "reference.json").read_text(encoding="utf-8")
            )
            self.assertTrue((root / reference["reference_image"]).is_file())
            self.assertTrue(
                all((root / row["reference_image"]).is_file() for row in rows)
            )

    def test_masks_are_binary_and_break_has_two_components(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            generate_m06_validation_dataset(root, make_config(), seed=42)
            for image_path in sorted((root / "images").glob("*.png")):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(
                    str(root / "masks" / "glue" / image_path.name),
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertEqual(image.shape, mask.shape)
                self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

            with open(
                root / "manifest.csv",
                newline="",
                encoding="utf-8",
            ) as handle:
                break_id = next(
                    row["image_id"]
                    for row in csv.DictReader(handle)
                    if row["scenario"] == "break"
                )
            break_label = json.loads(
                (root / "labels" / f"{break_id}.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(break_label["component_count"], 2)
            self.assertEqual(break_label["expected_vm_defect_type"], 4)

    def test_deterministic_for_same_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            cfg = make_config()
            generate_m06_validation_dataset(first, cfg, seed=77)
            generate_m06_validation_dataset(second, cfg, seed=77)
            for relative in ("images", "masks/glue", "labels"):
                left_files = sorted((first / relative).glob("*"))
                right_files = sorted((second / relative).glob("*"))
                self.assertEqual(len(left_files), len(right_files))
                for left, right in zip(left_files, right_files):
                    self.assertEqual(digest(left), digest(right))

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            root.mkdir()
            (root / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_m06_validation_dataset(root, make_config())


if __name__ == "__main__":
    unittest.main()
