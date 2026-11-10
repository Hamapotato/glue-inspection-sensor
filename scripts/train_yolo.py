"""训练 YOLOv3-5FL + SPP（B2，论文 3.3）。

用法（消融：--levels 3/5 + --spp/--no-spp 对应表 3-4 四组）：
    py -3 scripts/train_yolo.py --synthetic --epochs 20 --levels 5 --spp --out outputs/train_yolo_5l_spp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.data import GlueDataset, SyntheticGlueDataset, read_manifest  # noqa: E402
from algo.net.yolo import YOLO5FL  # noqa: E402
from algo.train import TrainConfig, evaluate_yolo, train_yolo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 YOLOv3-5FL + SPP")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--labels", default="data/labels/reflection")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--levels", type=int, default=5, choices=[3, 5])
    parser.add_argument("--spp", dest="spp", action="store_true", default=True)
    parser.add_argument("--no-spp", dest="spp", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_yolo")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    if args.synthetic:
        train_ds = SyntheticGlueDataset("reflection", img_size=img_size, length=32, seed=args.seed)
        val_ds = SyntheticGlueDataset("reflection", img_size=img_size, length=12, seed=args.seed + 100)
    else:
        if not args.manifest:
            print("非合成模式需要 --manifest", file=sys.stderr)
            return 2
        records = read_manifest(args.manifest)
        train_ds = GlueDataset(records, args.image_root, args.labels, img_size)
        val_ds = train_ds

    model = YOLO5FL(num_levels=args.levels, use_spp=args.spp)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device,
                      out_dir=args.out, seed=args.seed)
    result = train_yolo(model, train_ds, cfg)
    metrics = evaluate_yolo(model, val_ds, device_name=args.device)

    print(f"checkpoint: {result['checkpoint']}")
    print(f"验证集 recall={metrics['recall']:.3f} meanIoU={metrics['mean_iou']:.4f} "
          f"(n={metrics['n']}, levels={args.levels}, spp={args.spp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
