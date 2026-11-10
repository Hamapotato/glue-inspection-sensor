"""B2：YOLOv3-5FL + SPP 反光检测（论文 3.3）。

对应关系
--------
- 3.2  YOLO 单阶段检测原理
- 3.3.2 增加网络特征层的 YOLOv3-5FL：在 3 个尺度基础上再上采样两级，共 5 个输出尺度
- 3.3.3 嵌入 SPP（4×4 / 2×2 / 1×1，图 3-21）
- 3.3.4 只判"是否存在反光 + 定位"，不做修复

结构：
    输入 → Darknet 轻量骨干（/2,/4,/8,/16,/32）→ [可选 SPP] FPN 5 级 →
    每级 1×1 检测头 → (B, A, H*W, 5+nc)

消融开关（对应表 3-4 的四组）：``num_levels`` ∈ {3,5}、``use_spp`` ∈ {True,False}。

说明：训练与损失按 **batch=1** 实现（工业单帧推理场景，冒烟足够）。批量训练留待
数据到位后扩展（见 tasks.md B-08）。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SPP, giou_diou_ciou


class ConvBlock(nn.Module):
    """Conv + BN + SiLU。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
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
    """轻量 Darknet：输出 /8、/16、/32 三级特征。"""

    def __init__(self, in_ch: int = 1, chans: tuple[int, ...] = (32, 64, 128, 256, 512), n_blocks: int = 2):
        super().__init__()
        self.stem = ConvBlock(in_ch, chans[0], 3, 2)  # /2
        self.stage1 = self._stage(chans[0], chans[1], n_blocks)  # /4
        self.stage2 = self._stage(chans[1], chans[2], n_blocks)  # /8  (C3)
        self.stage3 = self._stage(chans[2], chans[3], n_blocks)  # /16 (C4)
        self.stage4 = self._stage(chans[3], chans[4], n_blocks)  # /32 (C5)
        self.out_ch = chans

    @staticmethod
    def _stage(in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [ConvBlock(in_ch, out_ch, 3, 2)]
        layers += [ResBlock(out_ch) for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


# 自粗到细：/32、/16、/8、/4、/2
DEFAULT_ANCHORS: list[tuple[tuple[float, float], ...]] = [
    ((200, 60), (280, 80), (360, 100)),  # /32
    ((110, 40), (160, 56), (210, 70)),   # /16
    ((60, 26), (90, 36), (120, 48)),     # /8
    ((40, 20), (60, 28), (84, 38)),      # /4
    ((28, 16), (40, 22), (56, 30)),      # /2
]

STRIDES = (32, 16, 8, 4, 2)


class YOLO5FL(nn.Module):
    """YOLOv3 的 5 层特征版本（可关掉多余两层与 SPP，用于消融）。"""

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 1,
        num_levels: int = 5,
        use_spp: bool = True,
        anchors: list[tuple[tuple[float, float], ...]] | None = None,
    ):
        super().__init__()
        if num_levels not in (3, 5):
            raise ValueError("num_levels 只支持 3 或 5（对应 YOLOv3 与 YOLOv3-5FL）")
        self.num_levels = num_levels
        self.use_spp = use_spp
        self.num_classes = num_classes
        self.anchors = anchors or DEFAULT_ANCHORS
        self.num_anchors = len(self.anchors[0])

        self.backbone = DarkBackbone(in_ch=in_ch)
        c3_ch, c4_ch, c5_ch = self.backbone.out_ch[2], self.backbone.out_ch[3], self.backbone.out_ch[4]
        neck_ch = 128

        self.spp = SPP(c5_ch, neck_ch) if use_spp else ConvBlock(c5_ch, neck_ch, 1)
        self.lat4 = ConvBlock(c4_ch, neck_ch, 1)
        self.lat3 = ConvBlock(c3_ch, neck_ch, 1)
        self.neck5 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck4 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck3 = ConvBlock(neck_ch, neck_ch, 3)
        self.neck2 = ConvBlock(neck_ch, neck_ch // 2, 3)
        self.neck1 = ConvBlock(neck_ch // 2, neck_ch // 2, 3)

        per_anchor = 5 + num_classes
        self.head5 = nn.Conv2d(neck_ch, self.num_anchors * per_anchor, 1)
        self.head4 = nn.Conv2d(neck_ch, self.num_anchors * per_anchor, 1)
        self.head3 = nn.Conv2d(neck_ch, self.num_anchors * per_anchor, 1)
        self.head2 = nn.Conv2d(neck_ch // 2, self.num_anchors * per_anchor, 1)
        self.head1 = nn.Conv2d(neck_ch // 2, self.num_anchors * per_anchor, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def level_strides(self, img_size: tuple[int, int]) -> list[int]:
        """按输入尺寸返回各输出层实际步长（自粗到细）。"""
        h = img_size[0]
        strides = []
        for i in range(self.num_levels):
            stride = STRIDES[i]
            strides.append(stride if h // stride >= 1 else stride)
        return strides

    def head_sizes(self, img_size: tuple[int, int]) -> list[tuple[int, int]]:
        h, w = img_size
        return [(max(1, h // STRIDES[i]), max(1, w // STRIDES[i])) for i in range(self.num_levels)]


# --------------------------------------------------------------------------- #
# 解码
# --------------------------------------------------------------------------- #

def _grid(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


@torch.no_grad()
def decode_yolo(
    outputs: Sequence[torch.Tensor],
    img_size: tuple[int, int],
    anchors: list[tuple[tuple[float, float], ...]] | None = None,
    conf_thr: float = 0.5,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """解码为逐图 (boxes, scores)，坐标在输入图像尺度。batch 需为 1 或逐图调用。"""
    anchors = anchors or DEFAULT_ANCHORS
    h_img, w_img = img_size
    n_levels = len(outputs)
    strides = list(STRIDES[:n_levels])
    anc_levels = anchors[-n_levels:]

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for out, stride, anc in zip(outputs, strides, anc_levels):
        b, a, hw, per = out.shape
        h = max(1, h_img // stride)
        w = max(1, w_img // stride)
        out5 = out.view(b, a, h, w, per)
        gx, gy = _grid(h, w, out.device)
        aw = torch.tensor([p[0] for p in anc], device=out.device).view(1, a, 1)
        ah = torch.tensor([p[1] for p in anc], device=out.device).view(1, a, 1)
        cx = (torch.sigmoid(out5[..., 0]) + gx.view(1, 1, h * w)) * stride
        cy = (torch.sigmoid(out5[..., 1]) + gy.view(1, 1, h * w)) * stride
        bw = torch.exp(out5[..., 2].clamp(-8, 8)) * aw
        bh = torch.exp(out5[..., 3].clamp(-8, 8)) * ah
        obj = torch.sigmoid(out5[..., 4])
        cls = torch.sigmoid(out5[..., 5:])
        score = obj * (cls.max(dim=-1).values if cls.numel() else 1.0)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
        all_boxes.append(boxes.reshape(b, -1, 4))
        all_scores.append(score.reshape(b, -1))

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    boxes_all = torch.cat(all_boxes, dim=1)
    scores_all = torch.cat(all_scores, dim=1)
    for i in range(boxes_all.shape[0]):
        keep = scores_all[i] >= conf_thr
        results.append((boxes_all[i][keep], scores_all[i][keep]))
    return results


# --------------------------------------------------------------------------- #
# 目标构造与损失（batch=1）
# --------------------------------------------------------------------------- #

def _anchor_wh(anchors, device) -> torch.Tensor:
    return torch.tensor([[a[0], a[1]] for a in anchors], dtype=torch.float32, device=device)


def _wh_iou(wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    w = torch.minimum(wh[0], anchors[:, 0])
    h = torch.minimum(wh[1], anchors[:, 1])
    inter = w * h
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

    简化版正样本分配：每个 GT 与各层 anchor 做宽高 IoU，取最匹配的层与 anchor，
    落在其中心所在网格。论文未给分配细节，属假设 A8（p0-spec §5）。
    """
    anchors = anchors or DEFAULT_ANCHORS
    n_levels = len(outputs)
    strides = list(STRIDES[:n_levels])
    anc_levels = anchors[-n_levels:]
    h_img, w_img = img_size
    targets: list[dict[str, torch.Tensor]] = []

    for out, stride, anc in zip(outputs, strides, anc_levels):
        _, a, _, per = out.shape
        h = max(1, h_img // stride)
        w = max(1, w_img // stride)
        device = out.device
        obj = torch.zeros((a, h * w), device=device)
        cls = torch.zeros((a, h * w, num_classes), device=device)
        box = torch.zeros((a, h * w, 4), device=device)
        mask = torch.zeros((a, h * w), dtype=torch.bool, device=device)

        if gt_boxes.numel() > 0:
            anc_t = _anchor_wh(anc, device).view(a, 2)
            for gt in gt_boxes:
                gw = (gt[2] - gt[0]).clamp(min=1.0).item()
                gh = (gt[3] - gt[1]).clamp(min=1.0).item()
                iou_wh = _wh_iou(torch.tensor([gw, gh], device=device), anc_t)
                ai = int(iou_wh.argmax())
                cx = float((gt[0] + gt[2]) / 2 / stride)
                cy = float((gt[1] + gt[3]) / 2 / stride)
                gi, gj = int(cx), int(cy)
                if not (0 <= gi < w and 0 <= gj < h):
                    continue
                idx = gj * w + gi
                obj[ai, idx] = 1.0
                mask[ai, idx] = True
                cls[ai, idx, 0] = 1.0
                box[ai, idx, 0] = cx - gi
                box[ai, idx, 1] = cy - gj
                box[ai, idx, 2] = torch.log(torch.tensor(gw / anc_t[ai, 0].item(), device=device))
                box[ai, idx, 3] = torch.log(torch.tensor(gh / anc_t[ai, 1].item(), device=device))

        targets.append({"obj": obj, "cls": cls, "box": box, "mask": mask})
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
    h_img, w_img = img_size

    for out, tgt, stride, anc in zip(outputs, targets, strides, anc_levels):
        _, a, hw, _ = out.shape
        h = max(1, h_img // stride)
        w = max(1, w_img // stride)
        logits = out[0]  # (A, HW, per)
        obj_logit = logits[..., 4]
        cls_logit = logits[..., 5:]
        box_pred = logits[..., :4]

        loss_obj = loss_obj + F.binary_cross_entropy_with_logits(obj_logit, tgt["obj"])
        m = tgt["mask"]
        if m.any():
            n_pos += int(m.sum())
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(cls_logit[m], tgt["cls"][m])
            pred = _decode_level(box_pred, stride, anc, h, w)
            gt = _target_boxes(tgt["box"], stride, anc, h, w)
            sim = giou_diou_ciou(pred[m], gt[m], mode="ciou")
            loss_box = loss_box + (1.0 - sim).mean()

    denom = max(1, n_pos)
    total = obj_weight * loss_obj / n_levels + cls_weight * loss_cls / denom + box_weight * loss_box / denom
    return {"loss_obj": loss_obj, "loss_cls": loss_cls, "loss_box": loss_box, "loss": total}
