"""下载演示权重（ONNX + anchors）到本仓库的 outputs/demo/。

权重与 outputs/ 都在 .gitignore 中（红线 R4：不把模型权重提交进 Git），
所以 `git pull` 之后需要一个命令把权重取回来。本脚本支持：

  # 从构建机的 HTTP 服务下载（推荐，PC 上：py -3 -m http.server 8000 --directory outputs/demo）
  python3 scripts/fetch_demo_weights.py --url http://192.168.1.20:8000

  # 直接给单个文件 URL（会写到 outputs/demo/onnx/rpn.onnx）
  python3 scripts/fetch_demo_weights.py --url http://192.168.1.20:8000/onnx/rpn.onnx

  # 校验 SHA256（可选，防止传坏）
  python3 scripts/fetch_demo_weights.py --url http://... --sha256 f85cc1e3...

  # 也可以从本地目录/文件复制
  python3 scripts/fetch_demo_weights.py --url /mnt/usb/demo
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "outputs" / "demo"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    scheme = urllib.parse.urlparse(url).scheme
    if scheme in ("http", "https"):
        print(f"[fetch] {url} -> {dst}")
        with urllib.request.urlopen(url, timeout=60) as r, dst.open("wb") as f:
            shutil.copyfileobj(r, f)
    else:
        src = Path(url)
        if not src.is_file():
            raise SystemExit(f"[ERROR] 本地文件不存在：{src}")
        print(f"[fetch] {src} -> {dst}")
        shutil.copyfile(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="HTTP 基地址（.../onnx/rpn.onnx）或本地目录/文件")
    ap.add_argument("--dst", default=str(DEFAULT_DST), help="目标目录（默认 outputs/demo）")
    ap.add_argument("--sha256", default=None, help="可选：rpn.onnx 的期望 SHA256")
    args = ap.parse_args()

    dst = Path(args.dst)
    base = args.url.rstrip("/")
    is_file_url = base.endswith(".onnx")
    if is_file_url:
        onnx_url = base
        anchors_url = base.rsplit("/", 1)[0] + "/anchors.json"
        # 若直接给 onnx 文件，anchors 可能不存在：允许失败并提示用配置里已内置的 anchors
    else:
        onnx_url = base + "/onnx/rpn.onnx"
        anchors_url = base + "/anchors.json"

    onnx_dst = dst / "onnx" / "rpn.onnx"
    try:
        fetch(onnx_url, onnx_dst)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 下载 ONNX 失败：{e}", file=sys.stderr)
        return 2

    size = onnx_dst.stat().st_size
    if size < 1024 * 1024:
        print(f"[ERROR] rpn.onnx 只有 {size} 字节，可能下载到了错误内容", file=sys.stderr)
        return 2
    actual = sha256_file(onnx_dst)
    print(f"[ok] rpn.onnx {size / 1024 / 1024:.2f} MB  sha256={actual}")
    if args.sha256 and actual.lower() != args.sha256.lower():
        print("[ERROR] SHA256 不匹配！请重新下载", file=sys.stderr)
        return 2

    try:
        fetch(anchors_url, dst / "anchors.json")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] anchors.json 未下载（{e}）；若配置里已写死 anchors_wh 可忽略", file=sys.stderr)

    # 静态 batch ONNX（RKNN 转换用）；构建机没有时忽略
    static_url = (onnx_url.rsplit("/", 1)[0] + "/rpn-static.onnx") if is_file_url else (base + "/onnx/rpn-static.onnx")
    try:
        fetch(static_url, dst / "onnx" / "rpn-static.onnx")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] rpn-static.onnx 未下载（{e}）；RKNN 转换前需在构建机执行 export_onnx.py --rpn-static", file=sys.stderr)

    print("\n完成。下一步：")
    print("  cd host && cargo build --release -p glue-agent --features infer")
    print("  cd .. && ./host/target/release/glue-agent --config configs/experiments/camera-rpn.yaml \\")
    print("      --camera --frames 100 --mm-per-px 0.0833 --state outputs/camera.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
