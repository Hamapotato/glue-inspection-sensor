"""ONNX → RKNN 转换（RK3588 NPU），需在 x86_64 Linux 上运行 rknn-toolkit2。

RKNN-Toolkit2 官方只提供 x86_64 Linux 的转换工具（板端 aarch64 只能跑 rknn-toolkit-lite2 做推理），
因此本脚本不期望在 Windows / RK3588 板端直接运行。转换完成后把 .rknn 拷到板端。

流程：
1) 从样例图生成 320×320 letterbox 校准图（uint8，114 填充，与训练/推理预处理一致）；
2) 生成 dataset.txt 校准列表；
3) rknn.config(mean=0, std=255) → load_onnx → build(do_quantization) → export_rknn。

用法（x86_64 Ubuntu，已 pip install rknn-toolkit2）：
  python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn.onnx --out outputs/demo/rknn/rpn.rknn

板端验证：
  python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn --onnx outputs/demo/onnx/rpn.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def letterbox_u8(gray: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """等比缩放 + 114 填充到 size×size（与 algo.data.letterbox 的 uint8 版本一致）。

    letterbox 常量与 host/crates/infer/src/lib.rs 保持一致：填充值 114。
    """
    h, w = gray.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def build_calib(images: list[Path], out_dir: Path, size: int, limit: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, path in enumerate(images[:limit]):
        raw = np.fromfile(str(path), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
        if gray is None:
            print(f"[WARN] 跳过无法读取的图：{path}", file=sys.stderr)
            continue
        canvas = letterbox_u8(gray, size)
        dst = out_dir / f"calib_{i:04d}.png"
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError(f"编码失败：{dst}")
        buf.tofile(str(dst))
        lines.append(str(dst))
    list_path = out_dir / "dataset.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"校准集：{len(lines)} 张 -> {list_path}")
    return list_path


def collect_images(dataset_dir: Path, samples_dir: Path) -> list[Path]:
    if dataset_dir.is_dir():
        imgs = sorted(dataset_dir.glob("*.png"))
        if imgs:
            return imgs
    if samples_dir.is_dir():
        return sorted(samples_dir.glob("*.png")) + sorted(samples_dir.glob("*.jpg"))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "outputs" / "demo" / "onnx" / "rpn.onnx"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo" / "rknn" / "rpn.rknn"))
    ap.add_argument("--dataset", default=str(ROOT / "outputs" / "demo_dataset" / "train"))
    ap.add_argument("--samples", default=str(ROOT / "outputs" / "rust_samples"))
    ap.add_argument("--calib-count", type=int, default=50)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--no-quant", action="store_true", help="不量化（仅结构转换，便于排查）")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    out = Path(args.out)
    if not onnx.is_file():
        print(f"[ERROR] 找不到 ONNX：{onnx}", file=sys.stderr)
        print("        请先生成演示权重：scripts/build-demo-weights.ps1（或手动训练+导出）", file=sys.stderr)
        return 2

    images = collect_images(Path(args.dataset), Path(args.samples))
    if not images:
        print("[ERROR] 没有校准图片：请先运行 scripts/make_demo_dataset.py 生成数据集", file=sys.stderr)
        return 2

    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError:
        print("[ERROR] 未安装 rknn-toolkit2（转换需要 x86_64 Linux）：", file=sys.stderr)
        print("        pip install rknn-toolkit2   # 版本需与板端 librknnrt.so 匹配", file=sys.stderr)
        print("        板端（aarch64）只能装 rknn-toolkit-lite2 做推理，不能转换。", file=sys.stderr)
        return 2

    calib_list = build_calib(images, ROOT / "outputs" / "demo" / "rknn" / "calib", args.size, args.calib_count)

    print(f"== 加载 ONNX: {onnx}")
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0.0]],
        std_values=[[255.0]],
        target_platform=args.target,
        optimization_level=3,
        quantized_algorithm="normal",
    )
    ret = rknn.load_onnx(model=str(onnx))
    if ret != 0:
        print(f"[ERROR] load_onnx 失败 ret={ret}", file=sys.stderr)
        return 1

    print(f"== 构建（量化={not args.no_quant}）")
    ret = rknn.build(do_quantization=not args.no_quant, dataset=str(calib_list))
    if ret != 0:
        print(f"[ERROR] build 失败 ret={ret}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(out))
    if ret != 0:
        print(f"[ERROR] export_rknn 失败 ret={ret}", file=sys.stderr)
        return 1
    rknn.release()
    print(f"[OK] {out}  ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print("下一步：把 .rknn 和 anchors_wh 配置拷到板端，用 configs/experiments/demo-rknn.yaml 运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
