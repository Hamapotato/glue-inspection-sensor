"""glue-infer 与 onnxruntime 的数值对拍（E-06 验收脚本）。

流程：
1. 用固定随机种子生成输入（f32 行优先），写 raw 文件；
2. 调 Rust `glue-infer run` 得到原始输出（raw + meta JSON）；
3. 用 onnxruntime 跑同一模型/输入，与 Rust 输出逐张量比较 max|diff|。

用法：
    .venv\\Scripts\\python.exe scripts\\check_infer_parity.py
    .venv\\Scripts\\python.exe scripts\\check_infer_parity.py --exe host/target/release/glue-infer.exe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_case(name: str, model: Path, exe: Path, size: int, tol: float = 1e-4) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, size, size)).astype(np.float32)
    work = ROOT / "outputs" / "infer_parity"
    work.mkdir(parents=True, exist_ok=True)
    raw_in = work / f"{name}_in.f32"
    raw_out = work / f"{name}_rust.f32"
    meta = work / f"{name}_rust.json"
    raw_in.write_bytes(x.tobytes())

    cmd = [
        str(exe), "run",
        "--model", str(model),
        "--input", str(raw_in),
        "--in-shape", f"{size}x{size}",
        "--out", str(raw_out),
        "--meta", str(meta),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{name}] Rust 推理失败：\n{proc.stdout}\n{proc.stderr}")
        raise SystemExit(1)
    info = json.loads(meta.read_text(encoding="utf-8"))
    rust = np.frombuffer(raw_out.read_bytes(), dtype=np.float32)

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: x}
    refs = sess.run(None, feed)

    diffs = []
    for out_info, ref in zip(info["outputs"], refs):
        off, ln = out_info["offset"], out_info["len"]
        got = rust[off: off + ln].reshape(ref.shape)
        diff = float(np.max(np.abs(got - ref)))
        diffs.append(diff)
        print(f"[{name}] {out_info['shape']}  max|rust-ort|={diff:.3e}")
    worst = max(diffs)
    status = "OK" if worst < tol else "FAIL"
    print(f"[{name}] worst={worst:.3e} ({status}, tol={tol:g})")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "host" / "target" / "release" / "glue-infer.exe"))
    ap.add_argument("--models", default=str(ROOT / "outputs" / "onnx"))
    args = ap.parse_args()
    exe = Path(args.exe)
    models = Path(args.models)
    if not exe.exists():
        print(f"找不到 glue-infer：{exe}")
        return 1
    worst = 0.0
    for name, file, size in [("rpn", "rpn.onnx", 320), ("yolo5fl", "yolo5fl.onnx", 128)]:
        model = models / file
        if not model.exists():
            print(f"跳过 {name}：缺少模型 {model}（先跑 scripts/export_onnx.py）")
            continue
        worst = max(worst, run_case(name, model, exe, size))
    print(f"[done] worst diff={worst:.3e} ({'OK' if worst < 1e-4 else 'CHECK'})")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
