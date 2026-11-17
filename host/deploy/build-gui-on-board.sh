#!/usr/bin/env bash
# 板端构建 GUI（需要网络与显示环境）：sudo ./build-gui-on-board.sh
# 在仓库根目录（含 host/ 的目录）执行。
set -euo pipefail

if [ ! -d host ]; then
  echo "[ERROR] 请在仓库根目录执行（应能看到 host/）"
  exit 1
fi

echo "== 1. 安装系统依赖（Ubuntu/Debian）=="
sudo apt-get update
sudo apt-get install -y build-essential pkg-config cmake curl \
  libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxkbcommon-dev \
  libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libasound2-dev \
  fonts-wqy-microhei

echo "== 2. 安装 Rust（已安装则跳过）=="
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
cargo --version

echo "== 3. 构建 glue-app =="
cd host
cargo build --release -p glue-app

echo "== 4. 运行提示 =="
cat <<'EOF'

构建完成：host/target/release/glue-app
有桌面时直接运行：
  GLUE_CONFIG=/opt/glue/configs/default.yaml \
  GLUE_SAMPLES=/opt/glue/samples \
  GLUE_DB=/opt/glue/logs/host.db \
  ./target/release/glue-app

无桌面时请用 glue-agent（见 install.sh），或用 X11 转发：
  ssh -X user@board 后运行 GUI

中文字体缺失时：
  export GLUE_FONT=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
EOF
