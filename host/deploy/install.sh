#!/usr/bin/env bash
# 安装 glue-agent 到 Ubuntu 板子：sudo ./install.sh
# 目录：/opt/glue/{bin,configs,samples,logs}
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${GLUE_PREFIX:-/opt/glue}"
ARCH="$(uname -m)"

case "$ARCH" in
  aarch64|arm64) BIN="$SRC/bin/aarch64/glue-agent" ;;
  x86_64|amd64)  BIN="$SRC/bin/x86_64/glue-agent" ;;
  *) echo "[ERROR] 不支持的架构：$ARCH（包内提供 aarch64 / x86_64）" >&2; exit 1 ;;
esac

if [ ! -f "$BIN" ]; then
  echo "[ERROR] 找不到对应架构的二进制：$BIN" >&2
  exit 1
fi

echo "[1/4] 创建目录 $PREFIX"
install -d "$PREFIX/bin" "$PREFIX/configs" "$PREFIX/configs/experiments" "$PREFIX/samples" "$PREFIX/logs" "$PREFIX/scripts"

echo "[2/4] 安装二进制（$ARCH）"
install -m 0755 "$BIN" "$PREFIX/bin/glue-agent"
cp -f "$SRC/configs/default.yaml" "$PREFIX/configs/default.yaml"
cp -f "$SRC"/configs/experiments/*.yaml "$PREFIX/configs/experiments/" 2>/dev/null || true
cp -f "$SRC"/scripts/*.py "$PREFIX/scripts/" 2>/dev/null || true
if [ -d "$SRC/samples" ]; then
  cp -f "$SRC"/samples/*.png "$PREFIX/samples/" 2>/dev/null || true
fi
# 自检脚本与说明放到安装目录，便于板端直接运行
cp -f "$SRC/deploy/verify.sh" "$PREFIX/verify.sh" 2>/dev/null || true
cp -f "$SRC/README.txt" "$PREFIX/README.txt" 2>/dev/null || true
cp -f "$SRC/deploy/CAMERA-MVS.md" "$PREFIX/CAMERA-MVS.md" 2>/dev/null || true
cp -f "$SRC/deploy/RKNN-CHECKLIST.md" "$PREFIX/RKNN-CHECKLIST.md" 2>/dev/null || true
chmod +x "$PREFIX/verify.sh" 2>/dev/null || true

echo "[3/4] 创建系统用户 glue（已存在则跳过）"
if ! id glue >/dev/null 2>&1; then
  useradd -r -s /usr/sbin/nologin glue
fi
chown -R glue:glue "$PREFIX"

echo "[4/4] 安装 systemd 服务（可选，按需启用）"
if [ -f "$SRC/deploy/glue-agent.service" ]; then
  cp -f "$SRC/deploy/glue-agent.service" /etc/systemd/system/glue-agent.service
  systemctl daemon-reload || true
fi

cat <<EOF

安装完成：$PREFIX
  二进制：$PREFIX/bin/glue-agent（$(du -h "$PREFIX/bin/glue-agent" | cut -f1)）
  配置：  $PREFIX/configs/default.yaml
  样例：  $PREFIX/samples/

快速自检：
  cd $PREFIX && ./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out logs/result.json

启用服务（常驻扫描 samples，每 5 秒）：
  sudo systemctl enable --now glue-agent
  journalctl -u glue-agent -f

相机模式（需先装 MVS SDK，见 deploy/CAMERA-MVS.md）：
  cd $PREFIX && ./bin/glue-agent --config configs/experiments/camera-mvs.yaml \
      --camera --frames 100 --mm-per-px 0.0833 --state logs/camera.jsonl

EOF
