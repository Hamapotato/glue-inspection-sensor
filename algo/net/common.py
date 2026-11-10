"""模型公共组件：卷积块、SPP、anchor、框运算与 NMS（纯 torch 实现）。

对应论文
--------
- 3.1.1 RPN 在特征图上滑窗生成 9 个 anchor
- 3.3.3 SPP 的 4×4 / 2×2 / 1×1 多尺度池化（图 3-21）
- 3.1.3 / 3.1.4 IoU 族与 CIoU-NMS
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础块
# --------------------------------------------------------------------------- #

class ConvBNSiLU(nn.Module):
    """Conv2d + BN + SiLU（等价于论文图中的 CBR/CBL 单元）。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SPP(nn.Module):
    """空间金字塔池化（论文 3.3.3，图 3-21）。

    对同一特征层做 4×4、2×2、1×1 三种自适应最大池化，再上采样回原尺寸并拼接，
    输出通道为输入的 4 倍。等价于论文中的 16+4+1=21 个 bin 的多尺度池化。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, mid, k=1)
        self.cv2 = ConvBNSiLU(mid * 4, out_ch, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        h, w = x.shape[-2:]
        feats = [x]
        for bins in (4, 2, 1):
            pooled = F.adaptive_max_pool2d(x, (bins, bins))
            feats.append(F.interpolate(pooled, size=(h, w), mode="nearest"))
        return self.cv2(torch.cat(feats, dim=1))


# --------------------------------------------------------------------------- #
# anchor
# --------------------------------------------------------------------------- #

def make_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    scales: tuple[int, ...] = (8, 16, 32),
    ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    device: torch.device | None = None,
) -> torch.Tensor:
    """在特征图上铺设 anchor，返回图像坐标 (H*W*A, 4)。"""
    anchors_wh = []
    for scale in scales:
        for ratio in ratios:
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            anchors_wh.append((w, h))
    aw = torch.tensor([a[0] for a in anchors_wh], dtype=torch.float32, device=device)
    ah = torch.tensor([a[1] for a in anchors_wh], dtype=torch.float32, device=device)

    ys = (torch.arange(feat_h, dtype=torch.float32, device=device) + 0.5) * stride
    xs = (torch.arange(feat_w, dtype=torch.float32, device=device) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx = gx.reshape(-1, 1)
    cy = gy.reshape(-1, 1)

    x1 = cx - aw.view(1, -1) / 2
    y1 = cy - ah.view(1, -1) / 2
    x2 = cx + aw.view(1, -1) / 2
    y2 = cy + ah.view(1, -1) / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
    return boxes


def wh_anchors(
    scales: tuple[int, ...] = (8, 16, 32),
    ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """返回 (A,2) 的宽高 anchor（YOLO 用）。"""
    out = []
    for scale in scales:
        for ratio in ratios:
            out.append((scale * math.sqrt(ratio), scale / math.sqrt(ratio)))
    return torch.tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 框运算
# --------------------------------------------------------------------------- #

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """两批框的 IoU 矩阵 (N,M)。"""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


def giou_diou_ciou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    mode: str = "ciou",
    eps: float = 1e-7,
) -> torch.Tensor:
    """返回逐对相似度 (N,)，与论文式 (3-9)/(3-10)/(3-12) 一致（越大越好）。

    box1/box2 形状为 (N,4) 或 (4,)；当一方为单框时会自动广播。
    """
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)
    if box1.shape[0] == 1 and box2.shape[0] > 1:
        box1 = box1.expand(box2.shape[0], 4)
    if box2.shape[0] == 1 and box1.shape[0] > 1:
        box2 = box2.expand(box1.shape[0], 4)
    if box1.shape[0] != box2.shape[0]:
        raise ValueError("box1 与 box2 数量不一致且无法广播")

    iou = box_iou(box1, box2, eps).diagonal()
    if mode == "iou":
        return iou

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    c_area = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)
    c_diag2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

    if mode == "giou":
        area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
        area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        return iou - (c_area - union) / c_area.clamp(min=eps)

    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    if mode == "diou":
        return iou - rho2 / c_diag2.clamp(min=eps)

    # ciou
    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)
    v = (4 / math.pi**2) * (torch.atan(w1 / h1) - torch.atan(w2 / h2)) ** 2
    alpha = v / (1 - iou + v).clamp(min=eps)
    return iou - (rho2 / c_diag2.clamp(min=eps) + alpha * v)


def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """标准 NMS，返回保留索引。"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def ciou_nms_torch(boxes: torch.Tensor, scores: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """CIoU-NMS（论文 3.1.4）：用 CIoU 代替 IoU 做抑制判断。"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        c = giou_diou_ciou(boxes[i].unsqueeze(0).expand(order.numel() - 1, 4), boxes[order[1:]], mode="ciou")
        order = order[1:][c <= thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
