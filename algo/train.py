"""训练与评估循环（B1 RPN / B2 YOLO5FL）。

只做编排：模型来自 ``algo.net``，数据来自 ``algo.data``，损失来自各模型模块。
依赖 torch（可选依赖，见 AGENTS.md §4.1）。

训练约定
--------
- 单帧（batch=1）训练，与工业现场"一帧一次触发"一致；
- 每 ``val_every`` 轮在验证集上评估，按验证指标保存 ``best.pt``，最后一轮另存
  ``last.pt``；
- 训练曲线写入 ``<out_dir>/history.json``，训练配置写入 ``<out_dir>/train_config.json``，
  便于报告复现（AGENTS.md §6.2）。
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
from .net.yolo import YOLO5FL, detect, yolo_loss, yolo_targets


@dataclass
class TrainConfig:
    """训练超参（论文未给完整配置，见 p0-spec A8）。"""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    out_dir: str = "outputs/train"
    log_every: int = 1
    val_every: int = 1
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _loader(dataset, cfg: TrainConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_one, num_workers=0)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def train_rpn(
    model: RPN,
    dataset,
    cfg: TrainConfig | None = None,
    iou_mode: str | None = None,
    val_dataset=None,
) -> dict:
    """训练 RPN；返回 history 并保存 checkpoint。"""
    cfg = cfg or TrainConfig()
    iou_mode = iou_mode or model.cfg.iou_mode
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs))
    loader = _loader(dataset, cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_metric = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        totals = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0}
        n = 0
        for item in loader:
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            cls_logits, reg, anchors, _ = model(image)
            losses = rpn_loss(cls_logits[0], reg[0], anchors, boxes, mode=iou_mode,
                              pos_thr=model.cfg.pos_thr, neg_thr=model.cfg.neg_thr,
                              samples_per_image=model.cfg.samples_per_image,
                              positive_fraction=model.cfg.positive_fraction)
            optim.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()
            for k in totals:
                totals[k] += float(losses[k].detach())
            n += 1
        scheduler.step()

        record = {"epoch": epoch, **{k: v / max(1, n) for k, v in totals.items()},
                  "lr": float(optim.param_groups[0]["lr"]),
                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if val_dataset is not None and epoch % cfg.val_every == 0:
            record["val"] = evaluate_rpn(model, val_dataset, proposal_cfg=ProposalConfig.from_rpn(model.cfg),
                                         device_name=cfg.device)
            metric = record["val"]["mean_iou"]
            if metric > best_metric:
                best_metric, best_epoch = metric, epoch
                model.save(out_dir / "best.pt")
        history.append(record)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            extra = f" val_iou={record['val']['mean_iou']:.4f}" if "val" in record else ""
            print(f"[rpn] epoch {epoch}/{cfg.epochs} loss={record['loss']:.4f} "
                  f"cls={record['loss_cls']:.4f} reg={record['loss_reg']:.4f}{extra} {record['ms']}ms")

    model.save(out_dir / "last.pt")
    _write_json(out_dir / "rpn_history.json", history)
    _write_json(out_dir / "train_config.json", {"task": "rpn", "config": asdict(cfg), "iou_mode": iou_mode,
                                                 "best_epoch": best_epoch, "best_val": best_metric})
    return {"checkpoint": str(out_dir / "last.pt"), "best": str(out_dir / "best.pt"),
            "history": history, "best_epoch": best_epoch, "best_val": best_metric}


def train_yolo(
    model: YOLO5FL,
    dataset,
    cfg: TrainConfig | None = None,
    val_dataset=None,
) -> dict:
    """训练 YOLO5FL；返回 history 并保存 checkpoint。"""
    cfg = cfg or TrainConfig()
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs))
    loader = _loader(dataset, cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_metric = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        totals = {"loss": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        n = 0
        for item in loader:
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            img_size = (image.shape[2], image.shape[3])
            outputs = model(image)
            targets = yolo_targets(outputs, boxes, img_size, anchors=model.anchors,
                                   num_classes=model.num_classes)
            losses = yolo_loss(outputs, targets, img_size, anchors=model.anchors)
            optim.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()
            for k in totals:
                totals[k] += float(losses[k].detach())
            n += 1
        scheduler.step()

        record = {"epoch": epoch, **{k: v / max(1, n) for k, v in totals.items()},
                  "lr": float(optim.param_groups[0]["lr"]),
                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if val_dataset is not None and epoch % cfg.val_every == 0:
            record["val"] = evaluate_yolo(model, val_dataset, conf_thr=model.cfg.conf_thr, device_name=cfg.device)
            metric = record["val"]["recall"]
            if metric > best_metric:
                best_metric, best_epoch = metric, epoch
                model.save(out_dir / "best.pt")
        history.append(record)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            extra = f" val_recall={record['val']['recall']:.4f}" if "val" in record else ""
            print(f"[yolo] epoch {epoch}/{cfg.epochs} loss={record['loss']:.4f} "
                  f"obj={record['loss_obj']:.4f} cls={record['loss_cls']:.4f} box={record['loss_box']:.4f}{extra} {record['ms']}ms")

    model.save(out_dir / "last.pt")
    _write_json(out_dir / "yolo_history.json", history)
    _write_json(out_dir / "train_config.json", {"task": "yolo5fl", "config": asdict(cfg),
                                                 "num_levels": model.num_levels, "use_spp": model.use_spp,
                                                 "best_epoch": best_epoch, "best_val": best_metric})
    return {"checkpoint": str(out_dir / "last.pt"), "best": str(out_dir / "best.pt"),
            "history": history, "best_epoch": best_epoch, "best_val": best_metric}


# --------------------------------------------------------------------------- #
# 评估（口径见 p0-spec M1/M2/M3；合成数据结果不作为论文指标）
# --------------------------------------------------------------------------- #

def evaluate_rpn(
    model: RPN,
    dataset,
    proposal_cfg: ProposalConfig | None = None,
    device_name: str = "cpu",
) -> dict:
    """在数据集上评估 RoI 定位：平均 IoU（M1 口径）与 IoU≥0.5 命中率。"""
    from .metrics import mean_iou

    device = _device(device_name)
    model.to(device).eval()
    proposal_cfg = proposal_cfg or ProposalConfig.from_rpn(model.cfg)
    preds, gts = [], []
    hits = 0
    skipped = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            if boxes.numel() == 0:
                skipped += 1
                continue
            cls_logits, reg, anchors, _ = model(image)
            p_boxes, _ = propose(cls_logits[0], reg[0], anchors,
                                 (image.shape[2], image.shape[3]), proposal_cfg)
            preds.append(p_boxes.cpu().numpy())
            gts.append(boxes.cpu().numpy())
            if p_boxes.numel():
                from .metrics import best_iou
                hits += int(best_iou(p_boxes.cpu().numpy(), boxes.cpu().numpy()) >= 0.5)
    return {"mean_iou": mean_iou(preds, gts) if preds else 0.0,
            "hit_rate": hits / max(1, len(preds)),
            "n": len(preds), "skipped": skipped}


def evaluate_yolo(model: YOLO5FL, dataset, conf_thr: float = 0.5, device_name: str = "cpu") -> dict:
    """在数据集上评估反光检测：样本召回（M2）与最佳框 IoU（M3）。"""
    from .metrics import best_iou

    device = _device(device_name)
    model.to(device).eval()
    ious: list[float] = []
    hits = 0
    false_alarms = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            image = item["image"].to(device).unsqueeze(0)
            boxes = item["boxes"].to(device)
            gt = boxes.cpu().numpy()
            result = detect(model, image, conf_thr=conf_thr)[0]
            pred = result["boxes"].cpu().numpy()
            if gt.size == 0:
                # 无反光样本：检出即误报（M2 口径之外，单独统计）
                false_alarms += int(pred.size > 0)
                continue
            iou = best_iou(pred, gt) if pred.size else 0.0
            ious.append(iou)
            hits += int(iou >= 0.5)
    n = max(1, len(ious))
    return {"recall": hits / n, "mean_best_iou": float(sum(ious) / n),
            "n": len(ious), "false_alarms": false_alarms}
