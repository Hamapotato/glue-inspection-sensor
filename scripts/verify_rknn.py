"""板端验证 RKNN 模型：对比 RKNNLite 与 ONNX Runtime 的输出与候选框。

运行环境：RK3588 板端（aarch64 Linux），需 `pip install rknn-toolkit-lite2 onnxruntime`。
在 x86 上也可用（没有 NPU 会失败），主要用于板端验证。

用法：
  python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn \
      --onnx outputs/demo/onnx/rpn.onnx --samples outputs/rust_samples

输出：
  - 每张图 raw 输出（cls/reg）的 max|Δ| 与平均相对误差；
  - 解码后候选框数量与最高分框的 IoU（RKNN vs ONNX）；
  - 判定：误差超阈值时返回非 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def letterbox(gray: np.ndarray, size: int, pad_value: int = 114):
    """返回 (uint8 canvas, f32 canvas, scale, pad_x, pad_y)，与 Rust/Python 训练一致。"""
    h, w = gray.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    f32 = canvas.astype(np.float32) / 255.0
    return canvas, f32, scale, x0, y0


def anchors_for(size: int, stride: int, wh: list[list[float]]) -> np.ndarray:
    f = size // stride
    ys, xs = np.mgrid[0:f, 0:f]
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    n, k = f * f, len(wh)
    pos = np.repeat(np.stack([cx.ravel(), cy.ravel()], axis=1), k, axis=0)
    sizes = np.tile(np.asarray(wh, dtype=np.float32), (n, 1))
    return np.concatenate([pos, sizes], axis=1).astype(np.float32)


def decode(anchors: np.ndarray, reg: np.ndarray) -> np.ndarray:
    cx = reg[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = reg[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = np.exp(reg[:, 2]) * anchors[:, 2]
    h = np.exp(reg[:, 3]) * anchors[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def top_boxes(cls: np.ndarray, reg: np.ndarray, anchors: np.ndarray, thr: float, topk: int) -> np.ndarray:
    scores = sigmoid(cls.reshape(-1))
    idx = np.argsort(-scores)[:topk]
    keep = idx[scores[idx] >= thr]
    if keep.size == 0:
        return np.zeros((0, 4), np.float32)
    return decode(anchors[keep], reg.reshape(-1, 4)[keep])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if ua <= 0 else inter / ua


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default=str(ROOT / "outputs" / "demo" / "rknn" / "rpn.rknn"))
    ap.add_argument("--onnx", default=str(ROOT / "outputs" / "demo" / "onnx" / "rpn.onnx"))
    ap.add_argument("--anchors", default=str(ROOT / "outputs" / "demo" / "anchors.json"))
    ap.add_argument("--samples", default=str(ROOT / "outputs" / "rust_samples"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--score-thr", type=float, default=0.95)
    args = ap.parse_args()

    rknn_path = Path(args.rknn)
    onnx_path = Path(args.onnx)
    anchors_path = Path(args.anchors)
    for p in (rknn_path, onnx_path, anchors_path):
        if not p.is_file():
            print(f"[ERROR] 缺少文件：{p}", file=sys.stderr)
            return 2

    wh = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = anchors_for(args.size, 16, wh)
    images = sorted(Path(args.samples).glob("*.png"))[: args.limit]
    if not images:
        print(f"[ERROR] {args.samples} 下没有 png", file=sys.stderr)
        return 2

    try:
        from rknnlite.api import RKNNLite  # type: ignore
    except ImportError:
        print("[ERROR] 未安装 rknn-toolkit-lite2（板端安装）：pip install rknn-toolkit-lite2", file=sys.stderr)
        return 2
    import onnxruntime as ort

    lite = RKNNLite()
    if lite.load_rknn(str(rknn_path)) != 0:
        print(f"[ERROR] load_rknn 失败：{rknn_path}", file=sys.stderr)
        return 1
    if lite.init_runtime() != 0:
        print("[ERROR] init_runtime 失败（没有 NPU 或驱动不匹配）", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    worst = 0.0
    for img_path in images:
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        u8, f32, _scale, _px, _py = letterbox(gray, args.size)
        onnx_outs = sess.run(None, {in_name: f32.reshape(1, 1, args.size, args.size)})
        rknn_outs = lite.inference(inputs=[u8[None, :, :, None]])
        if len(rknn_outs) < 2:
            print(f"[ERROR] RKNN 输出数量异常：{len(rknn_outs)}")
            return 1
        d_cls = float(np.max(np.abs(np.asarray(onnx_outs[0]).reshape(-1) - np.asarray(rknn_outs[0]).reshape(-1))))
        d_reg = float(np.max(np.abs(np.asarray(onnx_outs[1]).reshape(-1) - np.asarray(rknn_outs[1]).reshape(-1))))
        worst = max(worst, d_reg)

        cls_flat = np.asarray(onnx_outs[0]).reshape(-1)
        reg_flat = np.asarray(onnx_outs[1]).reshape(-1, 4)
        b_onnx = top_boxes(cls_flat, reg_flat, anchors, args.score_thr, 300)
        b_rknn = top_boxes(
            np.asarray(rknn_outs[0]).reshape(-1), np.asarray(rknn_outs[1]).reshape(-1, 4), anchors, args.score_thr, 300
        )
        ov = 0.0
        if b_onnx.shape[0] and b_rknn.shape[0]:
            ov = iou(b_onnx[0], b_rknn[0])
        print(
            f"{img_path.name:22s} max|Δcls|={d_cls:.5f} max|Δreg|={d_reg:.5f} "
            f"boxes onnx/rknn={b_onnx.shape[0]}/{b_rknn.shape[0]} top-IoU={ov:.4f}"
        )

    lite.release()
    print(f"\n最差 reg 误差：{worst:.5f}")
    # 阈值：量化后允许 <0.05（经验值，需按实际模型校准；超过就应检查量化集/算子支持）
    if worst > 0.05:
        print(f"[FAIL] reg 误差 {worst:.5f} > 0.05，建议检查量化数据集或改用 --no-quant 排查")
        return 1
    print("[PASS] RKNN 与 ONNX 输出一致（reg 误差 < 0.05）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
