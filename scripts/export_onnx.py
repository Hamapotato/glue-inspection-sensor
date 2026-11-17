"""导出 RPN / YOLO5FL 到 ONNX，并用 onnxruntime 与 PyTorch 做数值对拍。

用法：
    .venv\\Scripts\\python.exe scripts\\export_onnx.py
    .venv\\Scripts\\python.exe scripts\\export_onnx.py --rpn-weights outputs/rpn.pt --yolo-weights outputs/yolo.pt

说明：
- 未提供权重时导出随机初始化模型，仅用于验证"算子可导出 + ONNX 推理与 PyTorch 一致"；
- 指标只用于一致性检查，不代表任何检测精度。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algo.net.rpn import RPN, RPNConfig  # noqa: E402
from algo.net.yolo import YOLO5FL, YOLOConfig  # noqa: E402


class RPNExport(nn.Module):
    """RPN 推理子图：只输出 cls 与 reg；anchors 由宿主机按输入尺寸确定性生成。"""

    def __init__(self, model: RPN) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, reg, _anchors, _feat = self.model(x)
        return cls, reg


class YOLOExport(nn.Module):
    """YOLO5FL：把 5 个尺度的输出拼接成 (B, N, 5+nc) 单张量，便于 ONNX 输出。"""

    def __init__(self, model: YOLO5FL) -> None:
        super().__init__()
        self.model = model
        self.per = 5 + model.cfg.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.model(x)
        return torch.cat([o.reshape(o.shape[0], -1, self.per) for o in outs], dim=1)


def export_rpn(out_dir: Path, weights: Path | None, img: int = 320) -> dict:
    model = RPN(RPNConfig()).eval()
    if weights and weights.exists():
        model = RPN.load(weights)
    wrapper = RPNExport(model).eval()
    dummy = torch.randn(1, 1, img, img)
    onnx_path = out_dir / "rpn.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["image"],
            output_names=["cls", "reg"],
            dynamic_axes={"image": {0: "batch"}, "cls": {0: "batch"}, "reg": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    return {"model": model, "wrapper": wrapper, "dummy": dummy, "path": onnx_path, "names": ["cls", "reg"]}


def export_yolo(out_dir: Path, weights: Path | None, img: int = 128) -> dict:
    model = YOLO5FL(YOLOConfig()).eval()
    if weights and weights.exists():
        model = YOLO5FL.load(weights)
    wrapper = YOLOExport(model).eval()
    dummy = torch.randn(1, 1, img, img)
    onnx_path = out_dir / "yolo5fl.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["image"],
            output_names=["out"],
            dynamic_axes={"image": {0: "batch"}, "out": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    return {"model": model, "wrapper": wrapper, "dummy": dummy, "path": onnx_path, "names": ["out"]}


def verify(job: dict, batch: int = 2) -> float:
    import onnxruntime as ort

    dummy = job["dummy"]
    x = dummy.repeat(batch, 1, 1, 1)
    with torch.no_grad():
        torch_outs = job["wrapper"](x)
    if isinstance(torch_outs, torch.Tensor):
        torch_outs = (torch_outs,)
    sess = ort.InferenceSession(str(job["path"]), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    ort_outs = sess.run(names, {sess.get_inputs()[0].name: x.numpy()})
    diffs = []
    for t, o in zip(torch_outs, ort_outs):
        a = t.detach().numpy()
        diffs.append(float(np.max(np.abs(a - o))))
    return max(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "onnx"))
    ap.add_argument("--rpn-weights", default=None)
    ap.add_argument("--yolo-weights", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import onnx

    results = []
    for name, fn, weights in [
        ("rpn", export_rpn, args.rpn_weights),
        ("yolo5fl", export_yolo, args.yolo_weights),
    ]:
        job = fn(out_dir, Path(weights) if weights else None)
        model = onnx.load(str(job["path"]))
        onnx.checker.check_model(model)
        # 默认导出会写 .onnx.data 外部权重；这里内联成单文件，便于交付与端侧加载
        onnx.save(model, str(job["path"]))
        data_file = Path(str(job["path"]) + ".data")
        if data_file.exists():
            data_file.unlink()
        diff = verify(job)
        size_kb = job["path"].stat().st_size / 1024.0
        nparam = sum(p.numel() for p in job["model"].parameters())
        results.append((name, nparam, size_kb, diff))
        print(f"[{name}] params={nparam/1e6:.2f}M  onnx={size_kb:.0f}KB  max|torch-onnx|={diff:.3e}")

    worst = max(r[3] for r in results)
    print(f"[done] worst diff={worst:.3e} ({'OK' if worst < 1e-4 else 'CHECK'})")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
