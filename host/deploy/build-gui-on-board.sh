#!/usr/bin/env bash
# 板端构建 GUI（需要网络与显示环境）：sudo ./build-gui-on-board.sh
# 在仓库根目录（含 host/ 的目录）执行。
#
# 常用参数：
#   --skip-deps   跳过 apt 安装（已装好依赖/离线环境）
#   --force-x11   运行时提示强制 X11 后端（Wayland 桌面显示异常时用）
set -euo pipefail

SKIP_DEPS=0
FORCE_X11=0
for arg in "$@"; do
  case "$arg" in
    --skip-deps) SKIP_DEPS=1 ;;
    --force-x11) FORCE_X11=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "[WARN] 未知参数：$arg" ;;
  esac
done

if [ ! -d host ]; then
  echo "[ERROR] 请在仓库根目录执行（应能看到 host/）"
  exit 1
fi

echo "== 1. 检查系统依赖（GUI：X11/Wayland/GL/ALSA）=="
MISSING=()
for bin in gcc pkg-config cmake curl; do
  command -v "$bin" >/dev/null 2>&1 || MISSING+=("$bin")
done
for pc in x11 xkbcommon wayland-client gl egl alsa; do
  pkg-config --exists "$pc" 2>/dev/null || MISSING+=("lib(pc:$pc)")
done

if [ ${#MISSING[@]} -eq 0 ]; then
  echo "依赖齐全"
else
  echo "缺少：${MISSING[*]}"
  if [ "$SKIP_DEPS" -eq 1 ]; then
    echo "[ERROR] 已指定 --skip-deps，但依赖不齐；请先手动安装后重试"
    exit 1
  fi
  echo "== 1.1 安装系统依赖（Ubuntu/Debian）=="
  sudo apt-get update
  sudo apt-get install -y build-essential pkg-config cmake curl \
    libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxkbcommon-dev \
    libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libasound2-dev \
    fonts-wqy-microhei
  STILL=()
  for bin in gcc pkg-config cmake curl; do
    command -v "$bin" >/dev/null 2>&1 || STILL+=("$bin")
  done
  for pc in x11 xkbcommon wayland-client gl egl alsa; do
    pkg-config --exists "$pc" 2>/dev/null || STILL+=("lib(pc:$pc)")
  done
  if [ ${#STILL[@]} -gt 0 ]; then
    echo "[ERROR] 安装后仍缺少：${STILL[*]}"
    echo "        请检查 apt 源或改用离线安装；只跑无界面服务可跳过 GUI。"
    exit 1
  fi
  echo "依赖安装完成"
fi

echo "== 2. 检查显示环境 =="
if [ -n "${WAYLAND_DISPLAY:-}" ]; then
  echo "检测到 Wayland 会话（WAYLAND_DISPLAY=$WAYLAND_DISPLAY）"
elif [ -n "${DISPLAY:-}" ]; then
  echo "检测到 X11 会话（DISPLAY=$DISPLAY）"
else
  echo "[WARN] 未检测到 DISPLAY / WAYLAND_DISPLAY：当前是纯 SSH 无桌面环境，"
  echo "       GUI 无法打开。请改用 glue-agent，或先用 VNC / ssh -X 准备显示环境。"
fi

echo "== 3. 安装 Rust（已安装则跳过）=="
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
cargo --version
rustc --version

echo "== 4. 构建 glue-app（eframe: glow + x11 + wayland）=="
cd host
cargo build --release -p glue-app
BIN="target/release/glue-app"
if [ ! -x "$BIN" ]; then
  echo "[ERROR] 构建结束但没有找到 $BIN"
  exit 1
fi
echo "产物：$(pwd)/$BIN  ($(du -h "$BIN" | cut -f1))"

echo "== 5. 运行提示 =="
cat <<'EOF'

有桌面时运行：
  GLUE_CONFIG=/opt/glue/configs/default.yaml \
  GLUE_SAMPLES=/opt/glue/samples \
  GLUE_DB=/opt/glue/logs/host.db \
  ./target/release/glue-app

Wayland 桌面显示异常时，强制 X11（需有 XWayland）：
  WINIT_UNIX_BACKEND=x11 ./target/release/glue-app

无桌面时请用 glue-agent（见 install.sh），或用 VNC / ssh -X。

中文字体缺失时：
  export GLUE_FONT=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
EOF

if [ "$FORCE_X11" -eq 1 ]; then
  echo "提示：按 --force-x11 要求，运行时优先使用 WINIT_UNIX_BACKEND=x11"
fi
