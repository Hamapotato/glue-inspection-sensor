"""RKNN 前置检查：判断当前机器能否做"转换"或"板端推理"，并检查所需文件。

用法（构建机 / 板端都可以跑）：
    python3 scripts/rknn_preflight.py                 # 自动判断角色
    python3 scripts/rknn_preflight.py --role convert  # 只检查转换条件（x86_64 Linux）
    python3 scripts/rknn_preflight.py --role runtime  # 只检查板端推理条件（aarch64）

检查项：
  1. 平台：转换需 Linux x86_64；推理需 Linux aarch64；
  2. Python 包：转换需 rknn-toolkit2（rknn.api）；推理需 rknn-toolkit-lite2（rknnlite.api）；
  3. ONNX：默认 outputs/demo/onnx/rpn-static.onnx，要求静态 batch 且输入 320×320；
  4. 校准集与 anchors：outputs/demo_dataset/train、outputs/demo/anchors.json；
  5. 运行库：librknnrt.so（板端）与 /dev/rknpu 或 /dev/dri 设备；
  6. 产物：.rknn 是否存在（推理时必需）。

退出码：0 = 当前角色所需条件全部满足；1 = 有阻塞项（会打印具体缺口）。
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "outputs" / "demo" / "onnx" / "rpn-static.onnx"
DEFAULT_RKNN = ROOT / "outputs" / "demo" / "rknn" / "rpn.rknn"
DEFAULT_ANCHORS = ROOT / "outputs" / "demo" / "anchors.json"
DEFAULT_CALIB = ROOT / "outputs" / "demo_dataset" / "train"

LIB_CANDIDATES = [
    "/usr/lib/librknnrt.so",
    "/usr/lib/aarch64-linux-gnu/librknnrt.so",
    "/usr/local/lib/librknnrt.so",
    "/opt/rknn/lib/librknnrt.so",
]


def check_platform(role: str) -> list[tuple[str, bool, str]]:
    system = platform.system()
    machine = platform.machine()
    is_linux = system == "Linux"
    ok = False
    if role == "convert":
        ok = is_linux and machine in ("x86_64", "AMD64")
        hint = "转换需要 x86_64 Linux（rknn-toolkit2 官方只提供 x86_64）"
    else:
        ok = is_linux and machine in ("aarch64", "arm64")
        hint = "板端推理需要 aarch64 Linux"
    detail = f"{system}/{machine}"
    if not ok:
        detail += f"（{hint}）"
    return [("平台", ok, detail)]


def check_python_pkg(role: str) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    if role == "convert":
        try:
            import rknn  # noqa: F401

            out.append(("rknn-toolkit2", True, "已安装（可转换）"))
        except ImportError:
            out.append(("rknn-toolkit2", False, "未安装：pip install rknn-toolkit2（x86_64 Linux）"))
    else:
        try:
            import rknnlite  # noqa: F401

            out.append(("rknn-toolkit-lite2", True, "已安装（可推理）"))
        except ImportError:
            out.append(("rknn-toolkit-lite2", False, "未安装：pip install rknn-toolkit-lite2"))
    return out


def check_onnx(path: Path) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    if not path.is_file():
        out.append(("ONNX 文件", False, f"不存在：{path}（先运行 scripts/export_onnx.py --rpn-static）"))
        return out
    out.append(("ONNX 文件", True, f"{path.name}（{path.stat().st_size / 1024 / 1024:.2f} MB）"))

    # 优先用 onnx 检查静态形状；没有 onnx 时退回 onnxruntime
    static = None
    shape = None
    ops: list[str] = []
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path))
        dims = model.graph.input[0].type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        static = all(isinstance(d.dim_value, int) and d.dim_value > 0 for d in dims)
        ops = sorted({n.op_type for n in model.graph.node})
    except ImportError:
        try:
            import onnxruntime as ort  # type: ignore

            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            static = all(isinstance(d, int) for d in shape)
        except ImportError:
            pass

    if shape is not None:
        expected = [1, 1, 320, 320]
        out.append(("ONNX 输入", shape == expected, f"{shape}（期望 {expected}，静态 batch）"))
    if ops:
        out.append(("ONNX 算子", True, ", ".join(ops)))
    return out


def check_calib(calib: Path, anchors: Path) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    images = sorted(calib.glob("*.png")) if calib.is_dir() else []
    out.append(("校准集", len(images) >= 20, f"{len(images)} 张（{calib}）"))
    out.append(("anchors.json", anchors.is_file(), str(anchors)))
    return out


def find_rknn_lib() -> str | None:
    for p in LIB_CANDIDATES:
        if Path(p).is_file():
            return p
    env = os.environ.get("LD_LIBRARY_PATH", "")
    for d in env.split(os.pathsep):
        if d:
            for name in ("librknnrt.so", "librknnrt.so.1"):
                fp = Path(d) / name
                if fp.is_file():
                    return str(fp)
    try:
        ctypes.CDLL("librknnrt.so")
        return "librknnrt.so（系统搜索路径可加载）"
    except OSError:
        return None


def check_runtime(rknn: Path) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    lib = find_rknn_lib()
    out.append(("librknnrt.so", lib is not None, lib or "未找到：安装 RKNN 运行库或设置 GLUE_RKNN_LIB"))
    devs = glob.glob("/dev/rknpu*") + glob.glob("/dev/dri/renderD*")
    out.append(("NPU 设备", bool(devs), ", ".join(devs) if devs else "未发现 /dev/rknpu* 或 /dev/dri/renderD*"))
    out.append(("RKNN 模型", rknn.is_file(), str(rknn)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["auto", "convert", "runtime"], default="auto")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--rknn", default=str(DEFAULT_RKNN))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    args = ap.parse_args()

    machine = platform.machine()
    role = args.role
    if role == "auto":
        role = "convert" if machine in ("x86_64", "AMD64") else "runtime"
    print(f"== RKNN 前置检查（角色：{'转换' if role == 'convert' else '板端推理'}）==")

    checks: list[tuple[str, bool, str]] = []
    checks += check_platform(role)
    checks += check_python_pkg(role)
    if role == "convert":
        checks += check_onnx(Path(args.onnx))
        checks += check_calib(Path(args.calib), Path(args.anchors))
    else:
        checks += check_onnx(Path(args.onnx))
        checks += check_runtime(Path(args.rknn))

    failed = 0
    for name, ok, detail in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name:<16} {detail}")
        failed += 0 if ok else 1

    print()
    if failed:
        print(f"结论：{failed} 项未满足。")
        if role == "convert":
            print("下一步（在 x86_64 Linux 上）：")
            print("  pip install rknn-toolkit2")
            print("  python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn-static.onnx \\")
            print("      --out outputs/demo/rknn/rpn.rknn")
        else:
            print("下一步（板端）：")
            print("  # 安装 RKNN 运行库（librknnrt.so，来自 RKNN runtime / BSP），或：")
            print("  export GLUE_RKNN_LIB=/path/to/librknnrt.so")
            print("  python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn \\")
            print("      --onnx outputs/demo/onnx/rpn-static.onnx")
        return 1
    print("结论：当前角色所需条件全部满足。")
    if role == "convert":
        print("可直接执行：python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn-static.onnx")
    else:
        print("可执行：python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn --onnx outputs/demo/onnx/rpn-static.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
