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
  echo "[ERROR] 缺少演示权重 outputs/demo/onnx/rpn.onnx"
  echo "        请先在仓库根目录执行："
  echo "        python3 scripts/make_demo_dataset.py --with-reflection"
  echo "        python3 scripts/train_demo_weights.py --task rpn --kmeans-anchors --out outputs/demo"
  echo "        python3 scripts/export_onnx.py --rpn-weights outputs/demo/rpn/best.pt --out outputs/demo/onnx"
  exit 1
fi

export GLUE_CONFIG="configs/experiments/demo-rpn.yaml"
export GLUE_SAMPLES="outputs/rust_samples"
export GLUE_DB="outputs/host.db"

echo "[demo] 配置: $GLUE_CONFIG"
echo "[demo] 权重: outputs/demo/onnx/rpn.onnx（RPN 定位）"
echo "[demo] 样例: $GLUE_SAMPLES"
exec "$BIN" "$@"
