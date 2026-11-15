"""训练 RPN（B1，论文 3.1）。

用法（合成数据冒烟）：
    py -3 scripts/train_rpn.py --synthetic --epochs 30 --img-size 160 --out outputs/train_rpn

真实数据（manifest + 标注，格式见 data/README.md）：
    py -3 scripts/train_rpn.py --manifest data/manifest.csv --image-root data/raw \\
        --labels data/labels/roi --epochs 300 --out outputs/train_rpn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.data import GlueDataset, SyntheticGlueDataset, read_manifest  # noqa: E402
from algo.net.rpn import RPN, RPNConfig, anchors_from_kmeans  # noqa: E402
from algo.train import TrainConfig, evaluate_rpn, train_rpn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 RPN（论文 3.1）")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels/roi")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32, help="骨干通道宽度")
    parser.add_argument("--iou-mode", default="diou", choices=["iou", "giou", "diou", "ciou"])
    parser.add_argument("--kmeans-anchors", action="store_true",
                        help="用训练集真实框做 K-means 生成先验框（论文 3.1.2）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_rpn")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    if args.synthetic:
        train_ds = SyntheticGlueDataset("roi", img_size=img_size, length=48, seed=args.seed)
        val_ds = SyntheticGlueDataset("roi", img_size=img_size, length=16, seed=args.seed + 1000)
    else:
        if not args.manifest:
            print("非合成模式需要 --manifest", file=sys.stderr)
            return 2
        records = read_manifest(args.manifest)
        train_ds = GlueDataset(records, args.image_root, args.labels, img_size)
        val_ds = train_ds

    cfg_rpn = RPNConfig(in_ch=1, width=args.width, iou_mode=args.iou_mode, stride=16)
    if args.kmeans_anchors:
        gt = [tuple(map(float, item["boxes"][0].tolist())) for item in train_ds if item["boxes"].numel()]
        cfg_rpn.anchors = anchors_from_kmeans(gt, k=cfg_rpn.num_anchors, seed=args.seed)
        print(f"K-means 先验框：{[(round(w, 1), round(h, 1)) for w, h in cfg_rpn.anchors]}")

    model = RPN(cfg_rpn)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device,
                      out_dir=args.out, seed=args.seed)
    result = train_rpn(model, train_ds, cfg, iou_mode=args.iou_mode, val_dataset=val_ds)

    best = RPN.load(result["best"]) if Path(result["best"]).exists() else model
    metrics = evaluate_rpn(best, val_ds, device_name=args.device)
    summary = {"weights": result["best"], "mean_iou": metrics["mean_iou"],
               "hit_rate": metrics["hit_rate"], "n": metrics["n"],
               "iou_mode": args.iou_mode, "best_epoch": result["best_epoch"]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best: {result['best']} (epoch {result['best_epoch']})")
    print(f"验证集 mean IoU = {metrics['mean_iou']:.4f}，IoU>=0.5 命中率 = {metrics['hit_rate']:.3f} "
          f"(n={metrics['n']}, iou_mode={args.iou_mode})")
    print("注意：合成数据结果仅用于冒烟，不能作为论文指标（见 p0-spec §6）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
