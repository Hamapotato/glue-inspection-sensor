"""B1：区域生成网络（论文 3.1）。

对应关系
--------
- 3.1.1 RPN 在特征图上滑窗 → 1 维向量 → 两个分支（2k 分类 / 4k 回归）
- 3.1.2 K-means 生成先验框（本模块支持自定义 anchor，默认 9 个）
- 3.1.3 回归损失可选 iou/giou/diou/ciou（论文最终选 DIoU）
- 3.1.4 候选框筛选可选 NMS / CIoU-NMS

结构（与论文一致的最小实现）：
    输入 (1,H,W) → 4 层 stride=2 卷积（stride 16）→ 3×3 卷积 →
    1×1 分类头 (A) + 1×1 回归头 (4A) → 解码 → 排序 → NMS → Top-300
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ciou_nms_torch, giou_diou_ciou, make_anchors, nms_torch


class RPNBackbone(nn.Module):
    """轻量骨干：4 次 stride=2，输出 stride 16、256 通道。

    论文未给 backbone（假设 A6，见 p0-spec §5）；这里用小网络，便于 CPU 冒烟。
    """

    def __init__(self, in_ch: int = 1, width: int = 32):
        super().__init__()
        c = width
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.SiLU(inplace=True))
        self.block3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, 2, 1, bias=False), nn.BatchNorm2d(c * 4), nn.SiLU(inplace=True))
        self.block4 = nn.Sequential(nn.Conv2d(c * 4, c * 8, 3, 2, 1, bias=False), nn.BatchNorm2d(c * 8), nn.SiLU(inplace=True))
        self.out_ch = c * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class RPN(nn.Module):
    """区域生成网络主体。"""

    def __init__(
        self,
        in_ch: int = 1,
        num_anchors: int = 9,
        scales: tuple[int, ...] = (32, 64, 128),
        ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        width: int = 32,
    ):
        super().__init__()
        self.stride = 16
        self.num_anchors = num_anchors
        self.scales = scales
        self.ratios = ratios
        self.backbone = RPNBackbone(in_ch=in_ch, width=width)
        ch = self.backbone.out_ch
        self.rpn_conv = nn.Conv2d(ch, 256, 3, 1, 1)
        self.cls = nn.Conv2d(256, num_anchors, 1)
        self.reg = nn.Conv2d(256, num_anchors * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (cls_logits, reg_deltas, anchors, feat)。"""
        feat = self.backbone(x)
        feat = F.silu(self.rpn_conv(feat))
        b, _, h, w = feat.shape
        cls = self.cls(feat).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors)
        reg = self.reg(feat).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, 4)
        anchors = make_anchors(h, w, self.stride, self.scales, self.ratios, device=x.device)
        return cls, reg, anchors, feat


# --------------------------------------------------------------------------- #
# 编解码与损失
# --------------------------------------------------------------------------- #

def encode_boxes(anchors: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """anchor → (tx,ty,tw,th)，与论文 RPN 回归参数定义一致。"""
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-7)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-7)
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    gw = (targets[:, 2] - targets[:, 0]).clamp(min=1e-7)
    gh = (targets[:, 3] - targets[:, 1]).clamp(min=1e-7)
    gx = (targets[:, 0] + targets[:, 2]) / 2
    gy = (targets[:, 1] + targets[:, 3]) / 2
    return torch.stack([(gx - ax) / aw, (gy - ay) / ah, torch.log(gw / aw), torch.log(gh / ah)], dim=1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """回归参数 → 预测框。"""
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-7)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-7)
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    cx = deltas[:, 0] * aw + ax
    cy = deltas[:, 1] * ah + ay
    w = torch.exp(deltas[:, 2].clamp(-10, 10)) * aw
    h = torch.exp(deltas[:, 3].clamp(-10, 10)) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def rpn_loss(
    cls_logits: torch.Tensor,
    reg_deltas: torch.Tensor,
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    mode: str = "diou",
    pos_thr: float = 0.7,
    neg_thr: float = 0.3,
) -> dict[str, torch.Tensor]:
    """RPN 训练损失：分类（前景/背景）+ 定位（1 - 相似度）。

    gt_boxes: (N,4)；无目标时可传空张量。
    """
    device = cls_logits.device
    if gt_boxes.numel() == 0:
        cls_target = torch.zeros_like(cls_logits)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
        return {"loss_cls": loss_cls, "loss_reg": torch.zeros((), device=device), "loss": loss_cls}

    ious = torch.cat([box_iou_1vN(anchors, gt) for gt in gt_boxes], dim=1)  # (A_boxes, N_gt)
    best_iou, best_idx = ious.max(dim=1)
    pos = best_iou >= pos_thr
    neg = best_iou < neg_thr

    cls_target = torch.zeros_like(cls_logits)
    cls_target[pos] = 1.0
    valid = pos | neg
    loss_cls = F.binary_cross_entropy_with_logits(cls_logits[valid], cls_target[valid])

    if pos.any():
        matched = gt_boxes[best_idx[pos]]
        target_delta = encode_boxes(anchors[pos], matched)
        pred_delta = reg_deltas[pos]
        # 用预测框与真实框的相似度作为定位损失（论文 3.1.3 的做法）
        pred_boxes = decode_boxes(anchors[pos], pred_delta)
        sim = giou_diou_ciou(pred_boxes, matched, mode=mode)
        loss_reg = (1.0 - sim).mean()
    else:
        loss_reg = torch.zeros((), device=device)

    return {"loss_cls": loss_cls, "loss_reg": loss_reg, "loss": loss_cls + loss_reg}


def box_iou_1vN(boxes: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(N,4) 与单个 (4,) 的 IoU，返回 (N,1)。"""
    from .common import box_iou

    return box_iou(boxes, gt.unsqueeze(0))


# --------------------------------------------------------------------------- #
# 推理：候选框生成
# --------------------------------------------------------------------------- #

@dataclass
class ProposalConfig:
    pre_nms_topk: int = 6000
    post_nms_topk: int = 300
    nms_thr: float = 0.7
    nms_type: str = "nms"  # nms | ciou_nms
    min_size: float = 8.0
    score_thr: float = 0.0


@torch.no_grad()
def propose(
    cls_logits: torch.Tensor,
    reg_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_size: tuple[int, int],
    cfg: ProposalConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由 RPN 输出生成候选框，返回 (boxes (K,4), scores (K,))。"""
    cfg = cfg or ProposalConfig()
    scores = torch.sigmoid(cls_logits)
    boxes = decode_boxes(anchors, reg_deltas)

    h, w = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

    keep = (boxes[:, 2] - boxes[:, 0] >= cfg.min_size) & (boxes[:, 3] - boxes[:, 1] >= cfg.min_size)
    keep &= scores >= cfg.score_thr
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        return boxes, scores

    topk = min(cfg.pre_nms_topk, scores.numel())
    idx = scores.topk(topk).indices
    boxes, scores = boxes[idx], scores[idx]

    keep_idx = (
        ciou_nms_torch(boxes, scores, cfg.nms_thr)
        if cfg.nms_type == "ciou_nms"
        else nms_torch(boxes, scores, cfg.nms_thr)
    )
    boxes, scores = boxes[keep_idx], scores[keep_idx]
    topk = min(cfg.post_nms_topk, scores.numel())
    idx = scores.topk(topk).indices
    return boxes[idx], scores[idx]
