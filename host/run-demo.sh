#!/usr/bin/env bash
# 启动"推理演示"版上位机（Linux/板端）：RPN 定位 + 传统量测/质量门控
# 在仓库根目录或 host/ 下执行均可：host/run-demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="host/target/release/glue-app"
if [ ! -x "$BIN" ]; then
  echo "[ERROR] 缺少 GUI 二进制 $BIN，请先构建："
  echo "        cd host && cargo build --release -p glue-app"
  exit 1
fi
if [ ! -f "outputs/demo/onnx/rpn.onnx" ]; then
  echo "[ERROR] 缺少演示权重 outputs/demo/onnx/rpn.onnx（权重不入库）"
  echo "        方式 A：从构建机下载（构建机执行 py -3 -m http.server 8000 --directory outputs/demo）："
  echo "          mkdir -p outputs/demo/onnx && wget -O outputs/demo/onnx/rpn.onnx http://<构建机IP>:8000/onnx/rpn.onnx"
  echo "        方式 B：在构建机生成后拷贝：scripts/build-demo-weights.ps1（需 Python+torch）"
  exit 1
fi

export GLUE_CONFIG="configs/experiments/demo-rpn.yaml"
export GLUE_SAMPLES="outputs/rust_samples"
export GLUE_DB="outputs/host.db"

echo "[demo] 配置: $GLUE_CONFIG"
echo "[demo] 权重: outputs/demo/onnx/rpn.onnx（RPN 定位）"
echo "[demo] 样例: $GLUE_SAMPLES"
exec "$BIN" "$@"
