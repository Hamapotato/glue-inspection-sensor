"""B1：区域生成网络（论文 3.1）。

对应关系
--------
- 3.1.1 RPN 在特征图上滑窗 → 1 维向量 → 两个分支（2k 分类 / 4k 回归）
- 3.1.2 K-means 生成先验框（论文用 K-means 替换固定 anchor；本模块默认 9 个，
  可用 ``anchors_from_kmeans`` 由训练集真实框统计得到）
- 3.1.3 回归损失可选 iou/giou/diou/ciou（论文最终选 DIoU）
- 3.1.4 候选框筛选可选 NMS / CIoU-NMS（论文选定阈值 0.5，见表 3-2）

结构（与论文一致的最小实现）：
    输入 (1,H,W) → 4 层 stride=2 卷积（stride 16）→ 3×3 卷积 →
    1×1 分类头 (A) + 1×1 回归头 (4A) → 解码 → 排序 → NMS → Top-300

说明
----
- 骨干网络论文未给出（假设 A6，见 p0-spec §5），这里用轻量卷积便于 CPU 训练与
  RK3588 部署。
- 训练与推理均按 **batch=1** 实现（工业单帧场景）；批量训练留待数据到位后扩展。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import box_iou, ciou_nms_torch, giou_diou_ciou, make_anchors, nms_torch


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

@dataclass
class RPNConfig:
    """RPN 结构与推理参数（全部来自配置文件，见 interfaces.md §5）。"""

    in_ch: int = 1
    width: int = 32
    num_anchors: int = 9
    scales: tuple[int, int, int] = (32, 64, 128)
    ratios: tuple[float, float, float] = (0.5, 1.0, 2.0)
    anchors: list[tuple[float, float]] | None = None  # 显式 anchor 宽高（K-means 结果）
    stride: int = 16
    iou_mode: str = "diou"
    nms_type: str = "ciou_nms"
    nms_thr: float = 0.5
    pre_nms_topk: int = 3000
    post_nms_topk: int = 300
    min_size: float = 8.0
    score_thr: float = 0.0
    pos_thr: float = 0.4
    neg_thr: float = 0.3
    samples_per_image: int = 256      # RPN 每图采样 anchor 数（正负合计）
    positive_fraction: float = 0.5
    extra: dict = field(default_factory=dict)

    def anchor_wh(self) -> torch.Tensor:
        """返回 (A,2) 的 anchor 宽高。"""
        if self.anchors:
            return torch.tensor(self.anchors, dtype=torch.float32)
        out = []
        for scale in self.scales:
            for ratio in self.ratios:
                out.append((scale * math.sqrt(ratio), scale / math.sqrt(ratio)))
        return torch.tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #

class RPNBackbone(nn.Module):
    """轻量骨干：4 次 stride=2，输出 stride 16、width*8 通道。

    论文未给 backbone（假设 A6，见 p0-spec §5）；这里用小网络，便于 CPU 冒烟与
    后续 RK3588 上的 NPU 转换。
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

    def __init__(self, cfg: RPNConfig | None = None, **overrides):
        super().__init__()
        self.cfg = cfg or RPNConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"未知的 RPN 配置项：{key}")
            setattr(self.cfg, key, value)
        self.stride = self.cfg.stride
        self.num_anchors = self.cfg.num_anchors
        self.backbone = RPNBackbone(in_ch=self.cfg.in_ch, width=self.cfg.width)
        ch = self.backbone.out_ch
        self.rpn_conv = nn.Conv2d(ch, 256, 3, 1, 1)
        self.cls = nn.Conv2d(256, self.num_anchors, 1)
        self.reg = nn.Conv2d(256, self.num_anchors * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 分类头偏置初始化为负值，避免训练初期全部 anchor 判为前景
        nn.init.constant_(self.cls.bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (cls_logits, reg_deltas, anchors, feat)。

        cls_logits: (B, H*W*A)；reg_deltas: (B, H*W*A, 4)；anchors: (H*W*A, 4)。
        """
        feat = self.backbone(x)
        feat = F.silu(self.rpn_conv(feat))
        b, _, h, w = feat.shape
        cls = self.cls(feat).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors)
        reg = self.reg(feat).permute(0, 2, 3, 1).reshape(b, h * w * self.num_anchors, 4)
        anchors = make_anchors(h, w, self.stride, self.cfg.scales, self.cfg.ratios,
                               anchors=self.cfg.anchor_wh() if self.cfg.anchors else None,
                               device=x.device)
        return cls, reg, anchors, feat

    # -- 序列化 ---------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "dsh-rpn-v1", "config": asdict(self.cfg),
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "RPN":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("format") != "dsh-rpn-v1":
            raise ValueError(f"不是 RPN 权重文件：{path}")
        cfg = RPNConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


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
    pos_thr: float = 0.4,
    neg_thr: float = 0.3,
    samples_per_image: int = 256,
    positive_fraction: float = 0.5,
) -> dict[str, torch.Tensor]:
    """RPN 训练损失：签名（前景/背景）+ 定位（1 - 相似度）。

    - 正样本：与任一 GT 的 IoU ≥ ``pos_thr``，或该 GT 的最佳 anchor（保证正样本非空）；
    - 负样本：与所有 GT 的 IoU < ``neg_thr``；其余为忽略样本；
    - 从正负样本中按 ``positive_fraction`` 采样 ``samples_per_image`` 个 anchor 计算损失，
      避免背景 anchor 数量占绝对优势导致分类塌陷（标准的 RPN 训练做法，论文未给细节）。
    """
    if cls_logits.dim() == 2:
        if cls_logits.shape[0] != 1:
            raise ValueError("rpn_loss 目前只支持 batch=1")
        cls_logits = cls_logits[0]
    if reg_deltas.dim() == 3:
        if reg_deltas.shape[0] != 1:
            raise ValueError("rpn_loss 目前只支持 batch=1")
        reg_deltas = reg_deltas[0]
    device = cls_logits.device
    zero = torch.zeros((), device=device)
    if gt_boxes.numel() == 0:
        target = torch.zeros_like(cls_logits)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, target)
        return {"loss_cls": loss_cls, "loss_reg": zero, "loss": loss_cls}

    ious = torch.cat([box_iou(anchors, gt.unsqueeze(0)) for gt in gt_boxes], dim=1)  # (N_anchor, N_gt)
    best_iou, best_idx = ious.max(dim=1)
    pos = best_iou >= pos_thr
    forced = ious.argmax(dim=0)      # 每个 GT 的最佳 anchor
    pos[forced] = True
    neg = best_iou < neg_thr

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    n_samples = min(samples_per_image, n_pos + n_neg)
    if n_samples == 0:
        return {"loss_cls": zero, "loss_reg": zero, "loss": zero}

    num_pos = min(int(round(positive_fraction * n_samples)), n_pos)
    num_neg = min(n_samples - num_pos, n_neg)

    pos_idx = pos.nonzero(as_tuple=True)[0]
    # 正样本按 IoU 从高到低取（保证高质量正样本优先）
    if num_pos < n_pos:
        order = torch.argsort(best_iou[pos_idx], descending=True)
        pos_idx = pos_idx[order[:num_pos]]
    elif num_pos < n_pos:
        pos_idx = pos_idx[:num_pos]
    neg_idx = neg.nonzero(as_tuple=True)[0]
    if num_neg < n_neg:
        perm = torch.randperm(n_neg, device=device)[:num_neg]
        neg_idx = neg_idx[perm]

    sel = torch.cat([pos_idx, neg_idx])
    labels = torch.cat([torch.ones(len(pos_idx), device=device),
                        torch.zeros(len(neg_idx), device=device)])
    loss_cls = F.binary_cross_entropy_with_logits(cls_logits[sel], labels)

    if len(pos_idx):
        matched = gt_boxes[best_idx[pos_idx]]
        pred_boxes = decode_boxes(anchors[pos_idx], reg_deltas[pos_idx])
        sim = giou_diou_ciou(pred_boxes, matched, mode=mode)
        loss_reg = (1.0 - sim).mean()
    else:
        loss_reg = zero

    return {"loss_cls": loss_cls, "loss_reg": loss_reg, "loss": loss_cls + loss_reg}


# --------------------------------------------------------------------------- #
# 推理：候选框生成
# --------------------------------------------------------------------------- #

@dataclass
class ProposalConfig:
    """候选框筛选参数，默认值与论文表 3-2 的选定值一致。"""

    pre_nms_topk: int = 3000
    post_nms_topk: int = 300
    nms_thr: float = 0.5
    nms_type: str = "ciou_nms"
    min_size: float = 8.0
    score_thr: float = 0.0

    @classmethod
    def from_rpn(cls, cfg: RPNConfig) -> "ProposalConfig":
        return cls(pre_nms_topk=cfg.pre_nms_topk, post_nms_topk=cfg.post_nms_topk,
                   nms_thr=cfg.nms_thr, nms_type=cfg.nms_type,
                   min_size=cfg.min_size, score_thr=cfg.score_thr)


@torch.no_grad()
def propose(
    cls_logits: torch.Tensor,
    reg_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_size: tuple[int, int],
    cfg: ProposalConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由 RPN 输出生成候选框，返回 (boxes (K,4), scores (K,))，按 score 降序。"""
    cfg = cfg or ProposalConfig()
    scores = torch.sigmoid(cls_logits).reshape(-1)
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

    keep_idx = (ciou_nms_torch(boxes, scores, cfg.nms_thr)
                if cfg.nms_type == "ciou_nms" else nms_torch(boxes, scores, cfg.nms_thr))
    boxes, scores = boxes[keep_idx], scores[keep_idx]
    topk = min(cfg.post_nms_topk, scores.numel())
    idx = scores.topk(topk).indices
    return boxes[idx], scores[idx]


# --------------------------------------------------------------------------- #
# K-means 先验框（论文 3.1.2）
# --------------------------------------------------------------------------- #

def anchors_from_kmeans(
    gt_boxes: torch.Tensor | list[tuple[float, float, float, float]],
    k: int = 9,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """由真实框宽高做 K-means，返回 (k,2) 的 anchor 宽高。

    对应论文 3.1.2：用训练集统计替代固定 anchor。距离用 1 - wh-IoU，
    初始化用 k-means++ 以保证稳定（复现假设 A7）。
    """
    from ..geometry import kmeans_anchors

    if isinstance(gt_boxes, torch.Tensor):
        boxes = gt_boxes.detach().cpu().numpy()
    else:
        boxes = torch.tensor(gt_boxes, dtype=torch.float32).numpy()
    if boxes.size == 0:
        raise ValueError("gt_boxes 为空，无法聚类")
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    centers = kmeans_anchors(wh, k=k, seed=seed)
    return [(float(w), float(h)) for w, h in centers]

