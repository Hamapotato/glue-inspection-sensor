"""评估 YOLOv3-5FL：反光样本召回（M2）+ 框 IoU（M3），对应表 3-4 口径。

用法：
    py -3 scripts/eval_yolo.py --weights outputs/train_yolo_5l_spp/best.pt --synthetic
    py -3 scripts/eval_yolo.py --weights ... --manifest data/manifest.csv --json outputs/eval_yolo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.data import GlueDataset, SyntheticGlueDataset, read_manifest  # noqa: E402
from algo.net.yolo import YOLO5FL  # noqa: E402
from algo.train import evaluate_yolo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 YOLO5FL 反光检测（论文 3.3）")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels/reflection")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--conf-thr", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    model = YOLO5FL.load(args.weights)
    img_size = (args.img_size, args.img_size)
    if args.synthetic:
        ds = SyntheticGlueDataset("reflection", img_size=img_size, length=args.length, seed=777)
        source = "synthetic"
    else:
        if not args.manifest:
            print("非合成模式需要 --manifest", file=sys.stderr)
            return 2
        ds = GlueDataset(read_manifest(args.manifest), args.image_root, args.labels, img_size)
        source = "real"

    metrics = evaluate_yolo(model, ds, conf_thr=args.conf_thr, device_name=args.device)
    result = {"weights": args.weights, "source": source, "n": metrics["n"],
              "recall": metrics["recall"], "mean_best_iou": metrics["mean_best_iou"],
              "false_alarms": metrics["false_alarms"], "conf_thr": args.conf_thr,
              "num_levels": model.num_levels, "use_spp": model.use_spp,
              "note": "synthetic 结果仅用于冒烟，不能作为论文指标（p0-spec §6）" if source == "synthetic" else ""}
    print(f"recall = {result['recall']:.3f}  mean best IoU = {result['mean_best_iou']:.4f}  "
          f"误报 = {result['false_alarms']} (n={result['n']}, levels={model.num_levels}, "
          f"spp={model.use_spp}, source={source})")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
