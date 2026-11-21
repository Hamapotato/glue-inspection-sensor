"""用演示数据集训练 RPN / YOLO 权重（合成数据，仅用于展示流程）。

读取 `make_demo_dataset.py` 生成的 labels.json（图像 + 框），letterbox 到模型输入后训练；
输出权重与指标到 `outputs/demo/<task>/`，并由 `export_onnx.py --rpn-weights` 导出 ONNX。

用法：
  .venv\\Scripts\\python.exe scripts\\train_demo_weights.py --task rpn
  .venv\\Scripts\\python.exe scripts\\train_demo_weights.py --task yolo --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algo.data import letterbox, scale_boxes  # noqa: E402
from algo.net.rpn import RPN, RPNConfig, anchors_from_kmeans  # noqa: E402
from algo.net.yolo import YOLO5FL, YOLOConfig  # noqa: E402
from algo.train import TrainConfig, evaluate_rpn, evaluate_yolo, train_rpn, train_yolo  # noqa: E402


class DemoDataset(Dataset):
    """演示数据集：读取 PNG + 标签，返回 letterbox 后的张量与框（roi 用胶条框，reflection 用反光框）。"""

    def __init__(self, labels_path: Path, image_root: Path, task: str, img_size: tuple[int, int], limit: int = 0):
        self.records = json.loads(labels_path.read_text(encoding="utf-8"))
        if limit and limit > 0:
            self.records = self.records[:limit]
        self.image_root = image_root
        self.task = task
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        path = self.image_root / r["file"]
        raw = np.fromfile(str(path), dtype=np.uint8)  # 支持非 ASCII 路径
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
        if gray is None:
            raise FileNotFoundError(path)
        canvas, scale, pad = letterbox(gray, self.img_size)
        key = "reflection_boxes" if self.task == "reflection" else "boxes"
        boxes = np.asarray(r.get(key, []), dtype=np.float32).reshape(-1, 4)
        boxes = scale_boxes(boxes, scale, pad) if boxes.size else np.zeros((0, 4), np.float32)
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return {"image": tensor, "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
                "image_id": r["file"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "outputs" / "demo_dataset"))
    ap.add_argument("--task", default="rpn", choices=["rpn", "yolo"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=20, help="每 N 轮验证一次；0 表示不验证")
    ap.add_argument("--val-count", type=int, default=20, help="验证集只用前 N 张（限速）")
    ap.add_argument("--kmeans-anchors", action="store_true", help="用训练集胶条框做 K-means 生成 9 个先验框（论文 3.1.2）")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--no-spp", dest="spp", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo"))
    args = ap.parse_args()

    data = Path(args.data)
    img_size = (args.img_size, args.img_size)
    if args.task == "rpn":
        train_ds = DemoDataset(data / "train_labels.json", data, "roi", img_size)
        val_ds = DemoDataset(data / "val_labels.json", data, "roi", img_size, args.val_count)
        cfg_rpn = RPNConfig(in_ch=1, width=args.width, stride=16)
        if args.kmeans_anchors:
            gt = []
            for i in range(len(train_ds)):
                b = train_ds[i]["boxes"]
                if b.numel():
                    gt.extend(b.tolist())
            cfg_rpn.anchors = anchors_from_kmeans(gt, k=cfg_rpn.num_anchors, seed=args.seed)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "anchors.json").write_text(
                json.dumps(cfg_rpn.anchors, ensure_ascii=False, indent=2), encoding="utf-8")
            print("K-means 先验框:", [(round(w, 1), round(h, 1)) for w, h in cfg_rpn.anchors])
        model = RPN(cfg_rpn)
        cfg = TrainConfig(epochs=args.epochs, lr=args.lr, out_dir=str(Path(args.out) / "rpn"), seed=args.seed,
                          val_every=max(1, args.val_every))
        result = train_rpn(
            model, train_ds, cfg, iou_mode="diou",
            val_dataset=val_ds if args.val_every > 0 else None,
        )
        best = RPN.load(result["best"]) if Path(result["best"]).exists() else model
        metrics = evaluate_rpn(best, val_ds)
        summary = {"task": "rpn", "weights": result["best"], "best_epoch": result["best_epoch"],
                   "best_val": result["best_val"], "val": metrics,
                   "note": "合成演示数据，仅用于流程展示，不代表真实精度"}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "rpn_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"best: {result['best']} (epoch {result['best_epoch']})")
        print(f"验证集 mean IoU={metrics['mean_iou']:.4f} hit>=0.5={metrics['hit_rate']:.3f} n={metrics['n']}")
        return 0

    train_ds = DemoDataset(data / "train_labels.json", data, "reflection", (128, 128))
    val_ds = DemoDataset(data / "val_labels.json", data, "reflection", (128, 128))
    model = YOLO5FL(YOLOConfig(num_levels=args.levels, use_spp=args.spp))
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, out_dir=str(Path(args.out) / "yolo"), seed=args.seed)
    result = train_yolo(model, train_ds, cfg, val_dataset=val_ds)
    best = YOLO5FL.load(result["best"]) if Path(result["best"]).exists() else model
    metrics = evaluate_yolo(best, val_ds, conf_thr=0.3)
    summary = {"task": "yolo", "weights": result["best"], "best_epoch": result["best_epoch"],
               "best_val": result["best_val"], "val": metrics,
               "note": "合成演示数据，仅用于流程展示，不代表真实精度"}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "yolo_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best: {result['best']} (epoch {result['best_epoch']})")
    print(f"验证集: {json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
