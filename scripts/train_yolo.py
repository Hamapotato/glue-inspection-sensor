"""训练 YOLOv3-5FL + SPP（B2，论文 3.3）。

用法（消融：--levels 3/5 + --spp/--no-spp 对应表 3-4 四组）：
    py -3 scripts/train_yolo.py --synthetic --epochs 30 --img-size 160 \\
        --levels 5 --spp --out outputs/train_yolo_5l_spp

真实数据：
    py -3 scripts/train_yolo.py --manifest data/manifest.csv --image-root data/raw \\
        --labels data/labels/reflection --epochs 300 --out outputs/train_yolo_5l_spp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.data import GlueDataset, SyntheticGlueDataset, read_manifest  # noqa: E402
from algo.net.yolo import YOLO5FL, YOLOConfig, anchors_from_kmeans_yolo  # noqa: E402
from algo.train import TrainConfig, evaluate_yolo, train_yolo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 YOLOv3-5FL + SPP（论文 3.3）")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels/reflection")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-length", type=int, default=48)
    parser.add_argument("--val-length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--levels", type=int, default=5, choices=[3, 5])
    parser.add_argument("--spp", dest="spp", action="store_true", default=True)
    parser.add_argument("--no-spp", dest="spp", action="store_false")
    parser.add_argument("--conf-thr", type=float, default=0.5)
    parser.add_argument("--kmeans-anchors", action="store_true",
                        help="用训练集反光框做 K-means 生成 15 个先验框（论文 5.3.2.1）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_yolo")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    if args.synthetic:
        train_ds = SyntheticGlueDataset("reflection", img_size=img_size, length=args.train_length, seed=args.seed)
        val_ds = SyntheticGlueDataset("reflection", img_size=img_size, length=args.val_length, seed=args.seed + 1000)
    else:
        if not args.manifest:
            print("非合成模式需要 --manifest", file=sys.stderr)
            return 2
        records = read_manifest(args.manifest)
        train_ds = GlueDataset(records, args.image_root, args.labels, img_size)
        val_ds = train_ds

    cfg_yolo = YOLOConfig(num_levels=args.levels, use_spp=args.spp, conf_thr=args.conf_thr)
    if args.kmeans_anchors:
        gt = [tuple(map(float, item["boxes"][0].tolist())) for item in train_ds if item["boxes"].numel()]
        try:
            cfg_yolo.anchors = [[list(p) for p in level]
                                for level in anchors_from_kmeans_yolo(gt, k=15, seed=args.seed,
                                                                      num_levels=args.levels)]
            print(f"K-means 先验框（{args.levels} 层）："
                  f"{[[tuple(round(v, 1) for v in p) for p in lv] for lv in cfg_yolo.anchors]}")
        except ValueError as exc:
            print(f"K-means 先验框跳过（{exc}），使用默认 anchor")
    model = YOLO5FL(cfg_yolo)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device,
                      out_dir=args.out, seed=args.seed)
    result = train_yolo(model, train_ds, cfg, val_dataset=val_ds)

    best = YOLO5FL.load(result["best"]) if Path(result["best"]).exists() else model
    metrics = evaluate_yolo(best, val_ds, conf_thr=args.conf_thr, device_name=args.device)
    summary = {"weights": result["best"], "recall": metrics["recall"],
               "mean_best_iou": metrics["mean_best_iou"], "n": metrics["n"],
               "false_alarms": metrics["false_alarms"], "levels": args.levels,
               "spp": args.spp, "conf_thr": args.conf_thr, "best_epoch": result["best_epoch"]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best: {result['best']} (epoch {result['best_epoch']})")
    print(f"验证集 recall = {metrics['recall']:.3f}，最佳框 IoU = {metrics['mean_best_iou']:.4f}，"
          f"误报 = {metrics['false_alarms']} (n={metrics['n']}, levels={args.levels}, spp={args.spp})")
    print("注意：合成数据结果仅用于冒烟，不能作为论文指标（见 p0-spec §6）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
