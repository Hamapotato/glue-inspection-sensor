"""框运算：IoU 家族、anchor 编解码、NMS / CIoU-NMS、K-means 先验框。

对应论文
--------
- 3.1.2 K-means 聚类生成先验框
- 3.1.3 改进定位损失函数（GIoU / DIoU / CIoU，式 3-8 ~ 3-15）
- 3.1.4 改进 NMS（CIoU-NMS，表 3-2）

约定：``iou/gious/dious/cious`` 返回"相似度"（越大越好），对应损失为 ``1 - 值``，
与论文式 (3-15) ``L_CIoU = 1 - IoU + ρ²/c² + αυ`` 一致。
"""

from __future__ import annotations

import math

import numpy as np


def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """单个框与一组框的 IoU。

    box: (4,) ; boxes: (N,4) ; 返回 (N,)
    """
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _enclosing(box: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_x1 = np.minimum(box[0], boxes[:, 0])
    c_y1 = np.minimum(box[1], boxes[:, 1])
    c_x2 = np.maximum(box[2], boxes[:, 2])
    c_y2 = np.maximum(box[3], boxes[:, 3])
    c_area = np.clip(c_x2 - c_x1, 0, None) * np.clip(c_y2 - c_y1, 0, None)
    c_diag2 = np.clip(c_x2 - c_x1, 0, None) ** 2 + np.clip(c_y2 - c_y1, 0, None) ** 2
    return c_area, c_diag2


def bbox_giou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """式 (3-9)：GIoU = IoU - (C - U) / C。"""
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    iou = bbox_iou(box, boxes)
    union = _union_area(box, boxes)
    c_area, _ = _enclosing(box, boxes)
    return iou - np.where(c_area > 0, (c_area - union) / np.maximum(c_area, 1e-12), 0.0)


def bbox_diou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """式 (3-10)：DIoU = IoU - ρ²(b, b_gt) / c²。"""
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    iou = bbox_iou(box, boxes)
    _, c_diag2 = _enclosing(box, boxes)
    rho2 = (_center(box)[0] - _center(boxes)[0]) ** 2 + (_center(box)[1] - _center(boxes)[1]) ** 2
    return iou - np.where(c_diag2 > 0, rho2 / np.maximum(c_diag2, 1e-12), 0.0)


def bbox_ciou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """式 (3-12)~(3-14)：CIoU = IoU - (ρ²/c² + α·υ)。"""
    box = np.asarray(box, dtype=np.float64).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    iou = bbox_iou(box, boxes)
    diou_part = bbox_diou(box, boxes)
    return diou_part - _alpha_v(box, boxes) * _v(box, boxes)


def _center(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2


def _union_area(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return area_a + area_b - _inter(box, boxes)


def _inter(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _v(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """长宽比相似性 υ，式 (3-14)。"""
    w_gt = np.clip(box[2] - box[0], 1e-12, None)
    h_gt = np.clip(box[3] - box[1], 1e-12, None)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 1e-12, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 1e-12, None)
    return (4 / math.pi**2) * (np.arctan(w_gt / h_gt) - np.arctan(w / h)) ** 2


def _alpha_v(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """权重函数 α，式 (3-13)。"""
    iou = bbox_iou(box, boxes)
    v = _v(box, boxes)
    denom = (1 - iou) + v
    return np.where(denom > 0, v / np.maximum(denom, 1e-12), 0.0)


# --------------------------------------------------------------------------- #
# anchor 编解码（RPN 边界框回归参数）
# --------------------------------------------------------------------------- #

def encode_boxes(anchors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """anchor -> 回归参数 (tx, ty, tw, th)。"""
    aw = np.clip(anchors[:, 2] - anchors[:, 0], 1e-12, None)
    ah = np.clip(anchors[:, 3] - anchors[:, 1], 1e-12, None)
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    gw = np.clip(targets[:, 2] - targets[:, 0], 1e-12, None)
    gh = np.clip(targets[:, 3] - targets[:, 1], 1e-12, None)
    gx = (targets[:, 0] + targets[:, 2]) / 2
    gy = (targets[:, 1] + targets[:, 3]) / 2
    return np.stack([(gx - ax) / aw, (gy - ay) / ah, np.log(gw / aw), np.log(gh / ah)], axis=1)


def decode_boxes(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """回归参数 -> 预测框。"""
    aw = np.clip(anchors[:, 2] - anchors[:, 0], 1e-12, None)
    ah = np.clip(anchors[:, 3] - anchors[:, 1], 1e-12, None)
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    cx = deltas[:, 0] * aw + ax
    cy = deltas[:, 1] * ah + ay
    w = np.exp(np.clip(deltas[:, 2], -10, 10)) * aw
    h = np.exp(np.clip(deltas[:, 3], -10, 10)) * ah
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """标准 NMS，返回保留索引（按分数降序）。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-np.asarray(scores))
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        iou = bbox_iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def ciou_nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """CIoU-NMS（论文 3.1.4）：用 CIoU 代替 IoU 做抑制判断。

    注意 CIoU 取值范围 [-1, 1]，重叠度高时接近 1；阈值语义与标准 NMS 相同。
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-np.asarray(scores))
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        c = bbox_ciou(boxes[i], boxes[order[1:]])
        order = order[1:][c <= thr]
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
# K-means 先验框（论文 3.1.2）
# --------------------------------------------------------------------------- #

def _wh_iou(matrix: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """宽高形式的 IoU 矩阵，(N,2) x (K,2) -> (N,K)。"""
    w = np.minimum(matrix[:, None, 0], anchors[None, :, 0])
    h = np.minimum(matrix[:, None, 1], anchors[None, :, 1])
    inter = w * h
    area1 = matrix[:, 0] * matrix[:, 1]
    area2 = anchors[:, 0] * anchors[:, 1]
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-12)


def kmeans_anchors(
    wh: np.ndarray,
    k: int = 9,
    iters: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """对标注框的 (w, h) 做 K-means，距离为 1 - IoU（论文 3.1.2）。

    返回按面积升序排列的 (k, 2) 先验框。
    """
    wh = np.asarray(wh, dtype=np.float64).reshape(-1, 2)
    if len(wh) < k:
        raise ValueError(f"样本数 {len(wh)} 少于聚类数 {k}")
    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(len(wh), k, replace=False)].copy()
    for _ in range(iters):
        d = 1.0 - _wh_iou(wh, centers)
        assign = d.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            members = wh[assign == j]
            if len(members) > 0:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centers, atol=1e-6):
            centers = new
            break
        centers = new
    return centers[np.argsort(centers[:, 0] * centers[:, 1])]


def generate_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    anchors_wh: np.ndarray,
) -> np.ndarray:
    """在特征图上铺 anchor，返回原图坐标的 (N,4)，顺序为 (y, x, anchor)。"""
    anchors_wh = np.asarray(anchors_wh, dtype=np.float64).reshape(-1, 2)
    shifts_x = np.arange(0, feat_w) * stride + stride / 2
    shifts_y = np.arange(0, feat_h) * stride + stride / 2
    grid_x, grid_y = np.meshgrid(shifts_x, shifts_y)
    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    out = []
    for w, h in anchors_wh:
        out.append(np.stack([centers[:, 0] - w / 2, centers[:, 1] - h / 2,
                             centers[:, 0] + w / 2, centers[:, 1] + h / 2], axis=1))
    return np.concatenate(out, axis=0)
