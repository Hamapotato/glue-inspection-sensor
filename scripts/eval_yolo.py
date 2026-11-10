"""评估 YOLOv3-5FL：反光样本召回（M2）+ 框 IoU（M3），对应表 3-4 口径。

用法：
    py -3 scripts/eval_yolo.py --weights outputs/train_yolo/yolo5fl.pt --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from algo.data import SyntheticGlueDataset  # noqa: E402
from algo.net.yolo import YOLO5FL  # noqa: E402
from algo.train import evaluate_yolo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 YOLO5FL 反光检测")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--conf-thr", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    model = YOLO5FL(num_levels=int(ckpt.get("num_levels", 5)), use_spp=bool(ckpt.get("use_spp", True)))
    model.load_state_dict(ckpt["model"])

    img_size = (args.img_size, args.img_size)
    ds = SyntheticGlueDataset("reflection", img_size=img_size, length=args.length, seed=777)
    metrics = evaluate_yolo(model, ds, conf_thr=args.conf_thr, device_name=args.device)

    result = {
        "recall": metrics["recall"],
        "mean_iou": metrics["mean_iou"],
        "n": metrics["n"],
        "conf_thr": args.conf_thr,
        "num_levels": int(ckpt.get("num_levels", 5)),
        "use_spp": bool(ckpt.get("use_spp", True)),
        "weights": args.weights,
    }
    print(f"recall = {result['recall']:.3f}  mean IoU = {result['mean_iou']:.4f} "
          f"(n={result['n']}, levels={result['num_levels']}, spp={result['use_spp']})")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
