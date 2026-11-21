"""板端 AidLite 跑 RPN（先 ONNX 后端，验证 API 与模型兼容性）。

运行环境：AidLux Ubuntu（aarch64），已安装 aidlite（Python 3.10）。
用法：
  python3 scripts/aidlite_rpn_test.py --image outputs/rust_samples/0000_normal.png
  python3 scripts/aidlite_rpn_test.py --accel gpu --iters 10
  python3 scripts/aidlite_rpn_test.py --model outputs/demo/onnx/rpn-static.onnx

说明：
- 这是"验证脚本"，输出模型张量信息、单帧耗时与解码后的前几个 RoI；
- ONNX 后端走 AidLite 的 CPU/GPU；QNN(NPU) 需要 .ctx.bin 模型，本脚本暂不支持；
- 判定是否与 CPU 参考一致：看输出的 RoI 是否覆盖胶条（例如正常样例约 y≈420~540、x≈0~1280）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def letterbox_gray(gray: np.ndarray, size: int, pad_value: int = 114):
    import cv2

    h, w = gray.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def anchors_for(size: int, stride: int, wh: list[list[float]]) -> np.ndarray:
    f = size // stride
    ys, xs = np.mgrid[0:f, 0:f]
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    return np.concatenate(
        [np.stack([cx.ravel(), cy.ravel(), np.full(f * f, aw), np.full(f * f, ah)], axis=1) for aw, ah in wh],
        axis=0,
    ).astype(np.float32)


def decode(anchors: np.ndarray, reg: np.ndarray) -> np.ndarray:
    cx = reg[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = reg[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = np.exp(reg[:, 2]) * anchors[:, 2]
    h = np.exp(reg[:, 3]) * anchors[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "outputs" / "demo" / "onnx" / "rpn-static.onnx"))
    ap.add_argument("--image", default=str(ROOT / "outputs" / "rust_samples" / "0000_normal.png"))
    ap.add_argument("--anchors", default=str(ROOT / "outputs" / "demo" / "anchors.json"))
    ap.add_argument("--accel", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--score-thr", type=float, default=0.5)
    args = ap.parse_args()

    import cv2

    model_path = Path(args.model)
    for p in (model_path, Path(args.image), Path(args.anchors)):
        if not p.is_file():
            print(f"[ERROR] 缺少文件：{p}", file=sys.stderr)
            return 2

    import aidlite  # type: ignore

    print(f"[aidlite] lib={aidlite.get_library_version()} py={aidlite.get_py_library_version()} abi={aidlite.get_abi_version()}")
    aidlite.set_log_level(aidlite.LogLevel.INFO)

    config = aidlite.Config.create_instance()
    config.accelerate_type = aidlite.AccelerateType.TYPE_GPU if args.accel == "gpu" else aidlite.AccelerateType.TYPE_CPU
    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    config.framework_type = aidlite.FrameworkType.TYPE_ONNX

    model = aidlite.Model.create_instance(str(model_path))
    interp = aidlite.InterpreterBuilder.build_interpreter_from_model_and_config(model, config)
    if interp is None:
        print("[ERROR] build_interpreter 返回 None", file=sys.stderr)
        return 1
    interp.init()
    ret = interp.load_model()
    print(f"[aidlite] load_model ret={ret}")

    in_info = interp.get_input_tensor_info()
    out_info = interp.get_output_tensor_info()
    for gi, g in enumerate(in_info):
        for ti, t in enumerate(g):
            print(f"[in ] G{gi}T{ti} name={t.name} shape={t.shape} type={t.element_type}")
    for gi, g in enumerate(out_info):
        for ti, t in enumerate(g):
            print(f"[out] G{gi}T{ti} name={t.name} shape={t.shape} type={t.element_type}")

    # 准备输入（灰度 letterbox + /255，NCHW f32）
    raw = np.fromfile(str(args.image), dtype=np.uint8)
    gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    canvas, scale, px, py = letterbox_gray(gray, args.size)
    inp = (canvas.astype(np.float32) / 255.0)[None, None, :, :]
    data = inp.reshape(-1).tobytes()

    # 输入名：优先按 info 名字找
    in_name = in_info[0][0].name if in_info and in_info[0] else ""
    try:
        in_idx = interp.input_tensor_name_to_index(in_name)
        print(f"[aidlite] input name={in_name} index={in_idx}")
        ret = interp.set_input_tensor(in_idx, data)
    except Exception:  # noqa: BLE001
        ret = interp.set_input_tensor(0, data)
    if ret != 0:
        print("[ERROR] set_input_tensor 失败（检查输入 dtype/布局）", file=sys.stderr)
        return 1

    # 输出索引：按名字；找不到就用 0/1
    def out_index(name: str, fallback: int) -> int:
        try:
            return interp.output_tensor_name_to_index(name)
        except Exception:  # noqa: BLE001
            return fallback

    o_names = [out_info[0][0].name if out_info and out_info[0] else "", out_info[0][1].name if out_info and len(out_info[0]) > 1 else ""]
    idx_cls = out_index(o_names[0], 0)
    idx_reg = out_index(o_names[1], 1)

    times = []
    for _ in range(max(1, args.iters)):
        t0 = time.perf_counter()
        ret = interp.invoke()
        if ret != 0:
            print(f"[ERROR] invoke 失败 ret={ret}", file=sys.stderr)
            return 1
        cls = np.asarray(interp.get_output_tensor(idx_cls, aidlite.DataType.TYPE_FLOAT32), dtype=np.float32).reshape(-1)
        reg = np.asarray(interp.get_output_tensor(idx_reg, aidlite.DataType.TYPE_FLOAT32), dtype=np.float32).reshape(-1, 4)
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"[aidlite] invoke x{max(1, args.iters)}: p50≈{sorted(times)[len(times)//2]:.2f} ms (含取输出)")

    anchors = anchors_for(args.size, 16, json.loads(Path(args.anchors).read_text(encoding="utf-8")))
    scores = sigmoid(cls)
    top = np.argsort(-scores)[:5]
    boxes = decode(anchors, reg)
    print(f"[result] cls={cls.shape} reg={reg.shape} anchors={anchors.shape}")
    for rank, i in enumerate(top):
        b = boxes[i]
        # 还原到原图：去 padding、除以 scale
        x1 = (b[0] - px) / scale
        y1 = (b[1] - py) / scale
        x2 = (b[2] - px) / scale
        y2 = (b[3] - py) / scale
        print(f"  top{rank}: score={scores[i]:.4f}  box(原图)=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

    interp.destroy()
    print("[ok] AidLite ONNX 后端跑通（若 box 覆盖胶条区域，说明模型与预处理一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
