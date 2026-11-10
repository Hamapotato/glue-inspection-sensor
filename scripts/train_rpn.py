"""训练 RPN（B1，论文 3.1）。

用法：
    py -3 scripts/train_rpn.py --synthetic --epochs 20 --iou-mode diou --out outputs/train_rpn

真实数据：
    py -3 scripts/train_rpn.py --manifest data/manifest.csv --image-root data/raw \
        --labels data/labels/roi --epochs 300 --out outputs/train_rpn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.data import GlueDataset, SyntheticGlueDataset, read_manifest  # noqa: E402
from algo.net.rpn import RPN  # noqa: E402
from algo.train import TrainConfig, evaluate_rpn, train_rpn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 RPN")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels/roi")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iou-mode", default="diou", choices=["iou", "giou", "diou", "ciou"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_rpn")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    if args.synthetic:
        train_ds = SyntheticGlueDataset("roi", img_size=img_size, length=32, seed=args.seed)
        val_ds = SyntheticGlueDataset("roi", img_size=img_size, length=12, seed=args.seed + 100)
    else:
        if not args.manifest:
            print("非合成模式需要 --manifest", file=sys.stderr)
            return 2
        records = read_manifest(args.manifest)
        train_ds = GlueDataset(records, args.image_root, args.labels, img_size)
        val_ds = train_ds

    model = RPN()
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device,
                      out_dir=args.out, seed=args.seed)
    result = train_rpn(model, train_ds, cfg, iou_mode=args.iou_mode)

    metrics = evaluate_rpn(model, val_ds, device_name=args.device)
    print(f"checkpoint: {result['checkpoint']}")
    print(f"验证集 mean IoU: {metrics['mean_iou']:.4f}（n={metrics['n']}，iou_mode={args.iou_mode}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
