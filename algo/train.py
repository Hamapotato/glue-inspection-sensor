"""训练循环（B1 RPN / B2 YOLO5FL）。

仅做编排：模型来自 ``algo.net``，数据来自 ``algo.data``，损失来自各自模型模块。
依赖 torch（可选依赖）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate_one
from .net.rpn import RPN, ProposalConfig, propose, rpn_loss
from .net.yolo import YOLO5FL, yolo_loss, yolo_targets


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    out_dir: str = "outputs/train"
    log_every: int = 1
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _loader(dataset, cfg: TrainConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_one, num_workers=0)


def train_rpn(model: RPN, dataset, cfg: TrainConfig, iou_mode: str = "diou") -> dict:
    """训练 RPN；返回 history 并保存 checkpoint。"""
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _loader(dataset, cfg)
    history: list[dict] = []
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        totals = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0}
        n = 0
        for item in loader:
            image = item["image"].to(device)
            boxes = item["boxes"].to(device)
            cls_logits, reg, anchors, _ = model(image)
            losses = rpn_loss(cls_logits[0], reg[0], anchors, boxes, mode=iou_mode)
            optim.zero_grad()
            losses["loss"].backward()
            optim.step()
            for k in totals:
                totals[k] += float(losses[k])
            n += 1
        record = {"epoch": epoch, **{k: v / max(1, n) for k, v in totals.items()},
                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
        history.append(record)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"[rpn] epoch {epoch}/{cfg.epochs} loss={record['loss']:.4f} "
                  f"cls={record['loss_cls']:.4f} reg={record['loss_reg']:.4f} {record['ms']}ms")

    ckpt = out_dir / "rpn.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "iou_mode": iou_mode}, ckpt)
    (out_dir / "rpn_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"checkpoint": str(ckpt), "history": history}


def train_yolo(model: YOLO5FL, dataset, cfg: TrainConfig) -> dict:
    """训练 YOLO5FL；返回 history 并保存 checkpoint。"""
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _loader(dataset, cfg)
    history: list[dict] = []
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        totals = {"loss": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        n = 0
        for item in loader:
            image = item["image"].to(device)
            boxes = item["boxes"].to(device)
            img_size = (image.shape[2], image.shape[3])
            outputs = model(image)
            targets = yolo_targets(outputs, boxes, img_size, anchors=model.anchors, num_classes=model.num_classes)
            losses = yolo_loss(outputs, targets, img_size, anchors=model.anchors)
            optim.zero_grad()
            losses["loss"].backward()
            optim.step()
            for k in totals:
                totals[k] += float(losses[k])
            n += 1
        record = {"epoch": epoch, **{k: v / max(1, n) for k, v in totals.items()},
                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
        history.append(record)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"[yolo] epoch {epoch}/{cfg.epochs} loss={record['loss']:.4f} "
                  f"obj={record['loss_obj']:.4f} cls={record['loss_cls']:.4f} box={record['loss_box']:.4f} {record['ms']}ms")

    ckpt = out_dir / "yolo5fl.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "num_levels": model.num_levels, "use_spp": model.use_spp}, ckpt)
    (out_dir / "yolo_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"checkpoint": str(ckpt), "history": history}


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #

def evaluate_rpn(
    model: RPN,
    dataset,
    proposal_cfg: ProposalConfig | None = None,
    device_name: str = "cpu",
) -> dict:
    """在数据集上评估 RoI 定位平均 IoU（M1 口径）。"""
    from .metrics import mean_iou

    device = _device(device_name)
    model.to(device).eval()
    preds, gts = [], []
    with torch.no_grad():
        for item in dataset:
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            if boxes.numel() == 0:
                continue
            cls_logits, reg, anchors, _ = model(image)
            p_boxes, _ = propose(cls_logits[0], reg[0], anchors, (image.shape[2], image.shape[3]), proposal_cfg)
            preds.append(p_boxes.cpu().numpy())
            gts.append(boxes.cpu().numpy())
    return {"mean_iou": mean_iou(preds, gts), "n": len(preds)}


def evaluate_yolo(model: YOLO5FL, dataset, conf_thr: float = 0.5, device_name: str = "cpu") -> dict:
    """在数据集上评估反光检测（M2 样本召回 + M3 框 IoU）。"""
    from .metrics import best_iou

    device = _device(device_name)
    model.to(device).eval()
    ious: list[float] = []
    hits = 0
    with torch.no_grad():
        for item in dataset:
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            if boxes.numel() == 0:
                continue
            from .net.yolo import decode_yolo

            outputs = model(image)
            decoded = decode_yolo(outputs, (image.shape[2], image.shape[3]), anchors=model.anchors, conf_thr=conf_thr)
            pred = decoded[0][0].cpu().numpy()
            gt = boxes.cpu().numpy()
            iou = best_iou(pred, gt)
            ious.append(iou)
            hits += int(iou >= 0.5)
    n = max(1, len(ious))
    return {"recall": hits / n, "mean_iou": float(sum(ious) / n), "n": len(ious)}
