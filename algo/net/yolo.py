"""B2：YOLOv3-5FL + SPP 反光检测（论文 3.3）。

对应关系
--------
- 3.2  YOLO 单阶段检测原理
- 3.3.2 增加网络特征层的 YOLOv3-5FL：在 3 个尺度基础上再上采样两级，共 5 个输出尺度
- 3.3.3 嵌入 SPP（4×4 / 2×2 / 1×1，图 3-21）
- 3.3.4 只判"是否存在反光 + 定位"，不做修复

结构
----
输入 → 轻量 Darknet 骨干（/2,/4,/8,/16,/32）→ [可选 SPP] FPN 5 级 →
每级 1×1 检测头 → (B, A, H*W, 5+nc)

消融开关（对应论文表 3-4 的四组）：``num_levels`` ∈ {3,5}、``use_spp`` ∈ {True,False}。

说明
----
- 训练与推理按 **batch=1** 实现（工业单帧场景），批量训练待数据到位后扩展。
- 正样本分配：论文未给细节，本实现把每个 GT 分给"全层 anchor 中宽高 IoU 最大"的
  那一层与 anchor（YOLOv3 惯例），属复现假设 A8。
- 推理含按类别的 NMS（``detect``），训练/评估脚本统一使用它。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SPP, giou_diou_ciou, nms_torch

# 自粗到细：/32、/16、/8、/4、/2
STRIDES: tuple[int, ...] = (32, 16, 8, 4, 2)

DEFAULT_ANCHORS: list[tuple[tuple[float, float], ...]] = [
    ((200, 60), (280, 80), (360, 100)),  # /32
    ((110, 40), (160, 56), (210, 70)),   # /16
    ((60, 26), (90, 36), (120, 48)),     # /8
    ((40, 20), (60, 28), (84, 38)),      # /4
    ((28, 16), (40, 22), (56, 30)),      # /2
]


@dataclass
class YOLOConfig:
    """YOLO5FL 结构与推理参数（来自配置文件，见 interfaces.md §5）。"""

    in_ch: int = 1
    num_classes: int = 1
    num_levels: int = 5
    use_spp: bool = True
    anchors: list[list[list[float]]] | None = None
    conf_thr: float = 0.5
    nms_thr: float = 0.5
    max_det: int = 100
    extra: dict = field(default_factory=dict)

    def anchor_levels(self) -> list[tuple[tuple[float, float], ...]]:
        if self.anchors:
            return [tuple(tuple(float(v) for v in pair) for pair in level) for level in self.anchors]
        return DEFAULT_ANCHORS


def anchors_from_kmeans_yolo(
    gt_boxes: torch.Tensor | list[tuple[float, float, float, float]],
    k: int = 15,
    seed: int = 0,
    num_levels: int = 5,
) -> list[tuple[tuple[float, float], ...]]:
    """由真实框宽高做 K-means，生成分层的 15 个先验框（论文 5.3.2.1）。

    - 论文：K-means 得到 15 个先验框，avgIoU = 90.81%；
    - 本实现：聚类后按面积从小到大排序，每 3 个一组分配给从细到粗的输出层
      （/2 最小、/32 最大），层内顺序即 anchor 索引。
    """
    from ..geometry import kmeans_anchors

    if isinstance(gt_boxes, torch.Tensor):
        boxes = gt_boxes.detach().cpu().numpy()
    else:
        boxes = torch.tensor(gt_boxes, dtype=torch.float32).numpy()
    if boxes.size == 0:
        raise ValueError("gt_boxes 为空，无法聚类")
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    wh = wh[(wh[:, 0] > 1) & (wh[:, 1] > 1)]
    if len(wh) < k:
        raise ValueError(f"有效框数量 {len(wh)} 少于 k={k}")
    centers = kmeans_anchors(wh, k=k, seed=seed)
    centers = np.asarray(centers, dtype=np.float32)
    centers = centers[np.argsort(centers[:, 0] * centers[:, 1])]  # 面积升序
    per_level = k // num_levels
    if per_level * num_levels != k:
        raise ValueError("k 必须能被 num_levels 整除")
    # 自细到粗：/2, /4, /8, /16, /32
    levels: list[tuple[tuple[float, float], ...]] = []
    for i in range(num_levels):
        chunk = centers[i * per_level:(i + 1) * per_level]
        levels.append(tuple(tuple(float(v) for v in pair) for pair in chunk))
    return list(reversed(levels))  # 返回自粗到细，与 DEFAULT_ANCHORS 一致


class ConvBlock(nn.Module):
    """Conv + BN + SiLU。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """1×1 降维 + 3×3 卷积 + 残差。"""

    def __init__(self, ch: int):
        super().__init__()
        self.cv1 = ConvBlock(ch, ch // 2, k=1)
        self.cv2 = ConvBlock(ch // 2, ch, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class DarkBackbone(nn.Module):
    """轻量 Darknet：输出 /8、/16、/32 三级特征（论文 3.2 的 Darknet-53 简化版）。"""

    def __init__(self, in_ch: int = 1, chans: tuple[int, ...] = (32, 64, 128, 256, 512), n_blocks: int = 2):
        super().__init__()
        self.stem = ConvBlock(in_ch, chans[0], 3, 2)
        self.stage1 = self._stage(chans[0], chans[1], n_blocks)
        self.stage2 = self._stage(chans[1], chans[2], n_blocks)
        self.stage3 = self._stage(chans[2], chans[3], n_blocks)
        self.stage4 = self._stage(chans[3], chans[4], n_blocks)
        self.out_ch = chans

    @staticmethod
    def _stage(in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [ConvBlock(in_ch, out_ch, 3, 2)]
        layers += [ResBlock(out_ch) for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)   # /8
        c4 = self.stage3(c3)  # /16
        c5 = self.stage4(c4)  # /32
        return c3, c4, c5


class YOLO5FL(nn.Module):
    """YOLOv3 的 5 层特征版本（可关掉多余两层与 SPP，用于消融）。"""

    def __init__(self, cfg: YOLOConfig | None = None, **overrides):
        super().__init__()
        self.cfg = cfg or YOLOConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"未知的 YOLO 配置项：{key}")
            setattr(self.cfg, key, value)
        if self.cfg.num_levels not in (3, 5):
            raise ValueError("num_levels 只支持 3 或 5（对应 YOLOv3 与 YOLOv3-5FL）")

        self.num_levels = self.cfg.num_levels
        self.use_spp = self.cfg.use_spp
        self.num_classes = self.cfg.num_classes
        self.anchors = self.cfg.anchor_levels()
        self.num_anchors = len(self.anchors[0])

        self.backbone = DarkBackbone(in_ch=self.cfg.in_ch)
        c3_ch, c4_ch, c5_ch = self.backbone.out_ch[2], self.backbone.out_ch[3], self.backbone.out_ch[4]
        neck_ch = 128
        self.spp = SPP(c5_ch, neck_ch) if self.use_spp else ConvBlock(c5_ch, neck_ch, 1)
        self.lat4 = ConvBlock(c4_ch, neck_ch, 1)
        self.lat3 = ConvBlock(c3_ch, neck_ch, 1)
        self.neck5 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck4 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck3 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck2 = ConvBlock(neck_ch, neck_ch // 2, 3)
        self.neck1 = ConvBlock(neck_ch // 2, neck_ch // 2, 3)

        per = 5 + self.num_classes
        self.head5 = nn.Conv2d(neck_ch, self.num_anchors * per, 1)
        self.head4 = nn.Conv2d(neck_ch, self.num_anchors * per, 1)
        self.head3 = nn.Conv2d(neck_ch, self.num_anchors * per, 1)
        self.head2 = nn.Conv2d(neck_ch // 2, self.num_anchors * per, 1)
        self.head1 = nn.Conv2d(neck_ch // 2, self.num_anchors * per, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for head in (self.head1, self.head2, self.head3, self.head4, self.head5):
            nn.init.constant_(head.bias[4], -3.0)  # obj 分支初始偏负，抑制初期误检

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回自粗到细的检测头输出：每个 (B, A, H*W, 5+nc)。"""
        c3, c4, c5 = self.backbone(x)
        p5 = self.neck5(self.spp(c5))
        p4 = self.neck4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.neck3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))

        heads = [(self.head5, p5), (self.head4, p4), (self.head3, p3)]
        if self.num_levels == 5:
            p2 = self.neck2(F.interpolate(p3, scale_factor=2, mode="nearest"))
            p1 = self.neck1(F.interpolate(p2, scale_factor=2, mode="nearest"))
            heads += [(self.head2, p2), (self.head1, p1)]

        outputs = []
        per = 5 + self.num_classes
        for head, feat in heads:
            out = head(feat)
            b, _, h, w = out.shape
            out = out.view(b, self.num_anchors, per, h, w)
            outputs.append(out.permute(0, 1, 3, 4, 2).reshape(b, self.num_anchors, h * w, per))
        return outputs

    def strides_for(self, num_levels: int | None = None) -> tuple[int, ...]:
        return STRIDES[: (num_levels or self.num_levels)]

    # -- 序列化 ---------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "dsh-yolo5fl-v1", "config": asdict(self.cfg),
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "YOLO5FL":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("format") != "dsh-yolo5fl-v1":
            raise ValueError(f"不是 YOLO5FL 权重文件：{path}")
        cfg = YOLOConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# --------------------------------------------------------------------------- #
# 解码与推理
# --------------------------------------------------------------------------- #

def _grid(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


def _level_sizes(hw: int, img_size: tuple[int, int], stride: int) -> tuple[int, int]:
    w = max(1, img_size[1] // stride)
    h = max(1, img_size[0] // stride)
    if h * w != hw:
        raise ValueError(f"特征图大小 {hw} 与图像尺寸 {img_size} / stride {stride} 不匹配")
    return h, w


@torch.no_grad()
def decode_yolo(
    outputs: Sequence[torch.Tensor],
    img_size: tuple[int, int],
    anchors: list[tuple[tuple[float, float], ...]] | None = None,
    conf_thr: float = 0.5,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """解码为逐图 (boxes, scores)，坐标在输入图像尺度。batch 需为 1。"""
    anchors = anchors or DEFAULT_ANCHORS
    n_levels = len(outputs)
    strides = list(STRIDES[:n_levels])
    anc_levels = anchors[-n_levels:]

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for out, stride, anc in zip(outputs, strides, anc_levels):
        b, a, hw, per = out.shape
        h, w = _level_sizes(hw, img_size, stride)
        out5 = out.view(b, a, h, w, per)
        gx, gy = _grid(h, w, out.device)
        aw = torch.tensor([p[0] for p in anc], device=out.device).view(1, a, 1, 1)
        ah = torch.tensor([p[1] for p in anc], device=out.device).view(1, a, 1, 1)
        cx = (torch.sigmoid(out5[..., 0]) + gx.view(1, 1, h, w)) * stride
        cy = (torch.sigmoid(out5[..., 1]) + gy.view(1, 1, h, w)) * stride
        bw = torch.exp(out5[..., 2].clamp(-8, 8)) * aw
        bh = torch.exp(out5[..., 3].clamp(-8, 8)) * ah
        obj = torch.sigmoid(out5[..., 4])
        cls = torch.sigmoid(out5[..., 5:])
        score = obj * (cls.max(dim=-1).values if cls.numel() else 1.0)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
        all_boxes.append(boxes.reshape(b, -1, 4))
        all_scores.append(score.reshape(b, -1))

    boxes_all = torch.cat(all_boxes, dim=1)
    scores_all = torch.cat(all_scores, dim=1)
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(boxes_all.shape[0]):
        keep = scores_all[i] >= conf_thr
        results.append((boxes_all[i][keep], scores_all[i][keep]))
    return results


@torch.no_grad()
def detect(
    model: YOLO5FL,
    image: torch.Tensor,
    conf_thr: float | None = None,
    nms_thr: float | None = None,
    max_det: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """端到端推理：前向 → 解码 → NMS，返回每张图的 boxes/scores。

    image: (B,1,H,W) 已归一化；B 必须为 1。
    """
    if image.dim() != 4:
        raise ValueError("image 需为 (B,C,H,W)")
    if image.shape[0] != 1:
        raise ValueError("当前实现仅支持 batch=1")
    conf_thr = model.cfg.conf_thr if conf_thr is None else conf_thr
    nms_thr = model.cfg.nms_thr if nms_thr is None else nms_thr
    max_det = model.cfg.max_det if max_det is None else max_det

    was_training = model.training
    model.eval()
    outputs = model(image)
    img_size = (image.shape[2], image.shape[3])
    decoded = decode_yolo(outputs, img_size, anchors=model.anchors, conf_thr=conf_thr)
    boxes, scores = decoded[0]
    if boxes.numel() == 0:
        return [{"boxes": boxes, "scores": scores}]
    keep = nms_torch(boxes, scores, nms_thr)
    boxes, scores = boxes[keep], scores[keep]
    if boxes.shape[0] > max_det:
        idx = scores.topk(max_det).indices
        boxes, scores = boxes[idx], scores[idx]
    if was_training:
        model.train()
    # 裁剪到图像范围
    h, w = img_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return [{"boxes": boxes, "scores": scores}]


# --------------------------------------------------------------------------- #
# 目标构造与损失（batch=1）
# --------------------------------------------------------------------------- #

def _anchor_wh(anchors, device) -> torch.Tensor:
    return torch.tensor([[a[0], a[1]] for a in anchors], dtype=torch.float32, device=device)


def _wh_iou(wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """wh (2,) 与 anchors (N,2) 的宽高 IoU。"""
    inter = torch.minimum(wh[0], anchors[:, 0]) * torch.minimum(wh[1], anchors[:, 1])
    area = wh[0] * wh[1] + anchors[:, 0] * anchors[:, 1] - inter
    return inter / area.clamp(min=1e-7)


def yolo_targets(
    outputs: Sequence[torch.Tensor],
    gt_boxes: torch.Tensor,
    img_size: tuple[int, int],
    anchors: list[tuple[tuple[float, float], ...]] | None = None,
    num_classes: int = 1,
) -> list[dict[str, torch.Tensor]]:
    """为每个尺度构造目标（batch=1）。

    正样本分配：把每个 GT 分到"全层所有 anchor 中宽高 IoU 最大"的那一层与 anchor，
    落在其中心所在网格（YOLOv3 惯例；论文未给细节，属假设 A8）。
    """
    anchors = anchors or DEFAULT_ANCHORS
    n_levels = len(outputs)
    strides = list(STRIDES[:n_levels])
    anc_levels = anchors[-n_levels:]
    h_img, w_img = img_size
    per = 5 + num_classes
    targets: list[dict[str, torch.Tensor]] = []

    # 各层网格尺寸
    sizes: list[tuple[int, int]] = []
    for out, stride in zip(outputs, strides):
        b, a, hw, _ = out.shape
        h, w = _level_sizes(hw, img_size, stride)
        sizes.append((h, w))

    device = outputs[0].device
    per_level = []
    for out in outputs:
        _, a, hw, _ = out.shape
        per_level.append((a, hw))

    obj = [torch.zeros((a, hw), device=device) for a, hw in per_level]
    cls = [torch.zeros((a, hw, num_classes), device=device) for a, hw in per_level]
    box = [torch.zeros((a, hw, 4), device=device) for a, hw in per_level]
    mask = [torch.zeros((a, hw), dtype=torch.bool, device=device) for a, hw in per_level]

    # 全层 anchor 表：索引 -> (level, anchor)
    flat: list[tuple[int, int, float, float]] = []
    for li, anc in enumerate(anc_levels):
        for ai, (aw, ah) in enumerate(anc):
            flat.append((li, ai, aw, ah))
    flat_t = torch.tensor([[f[2], f[3]] for f in flat], dtype=torch.float32, device=device)

    for gt in gt_boxes:
        gw = float((gt[2] - gt[0]).clamp(min=1.0))
        gh = float((gt[3] - gt[1]).clamp(min=1.0))
        wh = torch.tensor([gw, gh], dtype=torch.float32, device=device)
        fi = int(_wh_iou(wh, flat_t).argmax())
        li, ai, aw, ah = flat[fi]
        stride = strides[li]
        h, w = sizes[li]
        cx = float((gt[0] + gt[2]) / 2 / stride)
        cy = float((gt[1] + gt[3]) / 2 / stride)
        gi, gj = int(cx), int(cy)
        if not (0 <= gi < w and 0 <= gj < h):
            continue
        idx = gj * w + gi
        obj[li][ai, idx] = 1.0
        mask[li][ai, idx] = True
        cls[li][ai, idx, 0] = 1.0
        box[li][ai, idx, 0] = cx - gi
        box[li][ai, idx, 1] = cy - gj
        box[li][ai, idx, 2] = torch.log(torch.tensor(gw / aw, device=device))
        box[li][ai, idx, 3] = torch.log(torch.tensor(gh / ah, device=device))

    for li in range(n_levels):
        targets.append({"obj": obj[li], "cls": cls[li], "box": box[li], "mask": mask[li]})
    assert per == targets[0]["cls"].shape[-1] + 5
    return targets


def _decode_level(box_pred: torch.Tensor, stride: int, anc, h: int, w: int) -> torch.Tensor:
    a = box_pred.shape[0]
    device = box_pred.device
    gx, gy = _grid(h, w, device)
    aw = torch.tensor([p[0] for p in anc], device=device).view(a, 1)
    ah = torch.tensor([p[1] for p in anc], device=device).view(a, 1)
    cx = (torch.sigmoid(box_pred[..., 0]) + gx.view(1, h * w)) * stride
    cy = (torch.sigmoid(box_pred[..., 1]) + gy.view(1, h * w)) * stride
    bw = torch.exp(box_pred[..., 2].clamp(-8, 8)) * aw
    bh = torch.exp(box_pred[..., 3].clamp(-8, 8)) * ah
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


def _target_boxes(box_t: torch.Tensor, stride: int, anc, h: int, w: int) -> torch.Tensor:
    a = box_t.shape[0]
    device = box_t.device
    gx, gy = _grid(h, w, device)
    aw = torch.tensor([p[0] for p in anc], device=device).view(a, 1)
    ah = torch.tensor([p[1] for p in anc], device=device).view(a, 1)
    cx = (box_t[..., 0] + gx.view(1, h * w)) * stride
    cy = (box_t[..., 1] + gy.view(1, h * w)) * stride
    bw = torch.exp(box_t[..., 2]) * aw
    bh = torch.exp(box_t[..., 3]) * ah
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


def yolo_loss(
    outputs: Sequence[torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
    img_size: tuple[int, int],
    anchors: list[tuple[tuple[float, float], ...]] | None = None,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """YOLO 损失：obj BCE + cls BCE + 正样本 (1 - CIoU)。batch=1。"""
    anchors = anchors or DEFAULT_ANCHORS
    n_levels = len(outputs)
    strides = list(STRIDES[:n_levels])
    anc_levels = anchors[-n_levels:]
    device = outputs[0].device
    loss_obj = torch.zeros((), device=device)
    loss_cls = torch.zeros((), device=device)
    loss_box = torch.zeros((), device=device)
    n_pos = 0

    for out, tgt, stride, anc in zip(outputs, targets, strides, anc_levels):
        _, a, hw, _ = out.shape
        h, w = _level_sizes(hw, img_size, stride)
        logits = out[0]  # (A, HW, per)
        obj_logit = logits[..., 4]
        cls_logit = logits[..., 5:]
        box_pred = logits[..., :4]

        # 正样本极少（一张图通常 1~2 个）：用 pos_weight 且按正样本数归一化，
        # 否则 5000+ 个背景 anchor 会把梯度稀释成"全判中等分数"的塌陷解。
        n_pos_l = float(tgt["obj"].sum())
        n_neg_l = float(tgt["obj"].numel() - n_pos_l)
        pos_weight = torch.tensor(min(50.0, max(1.0, n_neg_l / max(1.0, n_pos_l))), device=device)
        loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
            obj_logit, tgt["obj"], reduction="sum", pos_weight=pos_weight)
        m = tgt["mask"]
        if m.any():
            n_pos += int(m.sum())
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(cls_logit[m], tgt["cls"][m])
            pred = _decode_level(box_pred, stride, anc, h, w)
            gt = _target_boxes(tgt["box"], stride, anc, h, w)
            sim = giou_diou_ciou(pred[m], gt[m], mode="ciou")
            loss_box = loss_box + (1.0 - sim).mean()

    denom = max(1, n_pos)
    total = obj_weight * loss_obj / denom + cls_weight * loss_cls / denom + box_weight * loss_box / denom
    return {"loss_obj": loss_obj, "loss_cls": loss_cls, "loss_box": loss_box, "loss": total}
