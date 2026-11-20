# 在 RK3588 上部署上位机

> 目标平台：RK3588（aarch64，Debian/Ubuntu 系 BSP），无显示器也能跑检测服务；需要界面时按第 5 节构建 GUI。
> 当前交付形态：**无界面检测服务 `glue-agent`（静态 musl，单文件，无系统依赖）** + 可选 GUI。

---

## 0. 打包与上板（推荐，先看这里）

```powershell
# 一条命令生成部署包（aarch64 + x86_64 两个静态二进制 + 配置 + 样例 + 脚本）
powershell -ExecutionPolicy Bypass -File scripts\build-dist.ps1
# 产物：dist/glue-board-<日期>.tar.gz
```

上板步骤、预期输出、故障排查见 **[deploy/UBUNTU-CHECKLIST.md](deploy/UBUNTU-CHECKLIST.md)**（30 分钟清单）。

---

## 1. 交叉编译（在 Windows 开发机）

```powershell
cd host
powershell -ExecutionPolicy Bypass -File scripts\build-aarch64.ps1
```

产物在仓库根目录 `dist/rk3588/`：

```
dist/rk3588/
├─ bin/glue-agent              # aarch64 静态可执行文件（约 1.5 MB）
├─ configs/default.yaml        # 算法参数
├─ samples/                    # 放待检测图片
├─ logs/
└─ glue-agent.service          # systemd 单元
```

技术上：目标三元组 `aarch64-unknown-linux-musl`，链接器 `rust-lld`，**静态链接、不依赖 glibc**，直接拷到板上即可运行（已验证 ELF 头：64 位 AArch64）。

## 2. 拷贝到板子

```bash
scp -r dist/rk3588/* user@<board-ip>:/opt/glue/
ssh user@<board-ip>
sudo chmod +x /opt/glue/bin/glue-agent
```

## 3. 运行（无界面）

```bash
cd /opt/glue
# 单次检测一个目录，输出 JSON
./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out logs/result.json

# 常驻：每 5 秒扫描一次 samples，结果追加到 JSONL
./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 \
                 --state logs/results.jsonl --watch 5
```

结果 JSON 字段与 Windows 版一致（verdict / width_mm / timings_ms / rois）。

## 4. 开机自启（systemd）

```bash
sudo cp /opt/glue/glue-agent.service /etc/systemd/system/
sudo useradd -r -s /usr/sbin/nologin glue 2>/dev/null || true
sudo chown -R glue:glue /opt/glue
sudo systemctl daemon-reload
sudo systemctl enable --now glue-agent
journalctl -u glue-agent -f
```

服务定义见 `host/deploy/glue-agent.service`（按需修改路径与扫描间隔）。

## 4.1 可选：板端启用推理

`glue-agent` 默认不编译推理（`tract` 交叉编译需要 aarch64 C 工具链）。在板子上原生构建时可开启：

```bash
cd host
cargo build --release -p glue-agent --features infer
```

- 需要 `gcc`（或 aarch64 交叉工具链）与约 2 GB 磁盘；
- 需要把训练导出的 ONNX 放到板端，并在配置里设置 `locate.enabled/weights/anchors_wh`（参考 `configs/experiments/demo-rpn.yaml`）；
- 当前演示权重是合成数据训练的，仅用于链路展示。

## 4.2 RK3588 NPU（RKNN，可选）

1. **转换（x86_64 Linux，不是板端）**：
   ```bash
   pip install rknn-toolkit2   # 版本需与板端 librknnrt.so 匹配
   python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn.onnx --out outputs/demo/rknn/rpn.rknn
   ```
   校准图由脚本自动从 `outputs/demo_dataset/train` 生成（320×320 letterbox，uint8）。
2. **拷到板端**：`.rknn` + `configs/experiments/demo-rknn.yaml`（含 `anchors_wh`）。
3. **板端构建带 NPU 的二进制**：
   ```bash
   cd host
   cargo build --release -p glue-agent --features rknn
   ```
4. **运行**：
   ```bash
   ./target/release/glue-agent --config ../configs/experiments/demo-rknn.yaml \
       --input ../outputs/rust_samples --mm-per-px 0.0833 --out /tmp/rknn.json
   ```
   若运行库不在默认搜索路径：`export GLUE_RKNN_LIB=/usr/lib/librknnrt.so`。
5. **验证**：`python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn --onnx outputs/demo/onnx/rpn.onnx`（板端，RKNN 与 ONNX 输出误差 < 0.05 视为通过）。

> 注意：当前仓库未包含 `.rknn` 产物，也未在真实板端实测性能；转换与验证需在 x86 Linux + 板端分别执行。

## 4.3 相机（海康 MVS / USB3 Vision）

- 步骤、udev、脚本与排障见 [deploy/CAMERA-MVS.md](deploy/CAMERA-MVS.md)；
- 快速验证：`python3 scripts/camera_mvs.py --device 0 --fps 10 > /tmp/raw.bin`；
- 运行：`./bin/glue-agent --config configs/experiments/camera-mvs.yaml --camera --frames 100 --state logs/camera.jsonl`；
- GUI：左侧导航「相机」→「启动相机」（需桌面环境）。

## 5. 构建 GUI 版（需要显示环境）

`glue-app`（egui）依赖 X11/Wayland 与 OpenGL ES，建议**直接在板子上编译**。
`eframe` 已启用 `glow` + `x11` + `wayland` feature（`host/crates/app/Cargo.toml`），X11 与 Wayland 桌面均可直接运行。

```bash
# 1) 装 Rust（一次）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# 2) 系统依赖（Ubuntu/Debian）
sudo apt update
sudo apt install -y build-essential pkg-config cmake \
    libx11-dev libxi-dev libxcursor-dev libxrandr-dev libxkbcommon-dev \
    libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libasound2-dev

# 3) 编译
cd host
cargo build --release -p glue-app

# 4) 运行（有桌面/X11 时）
GLUE_CONFIG=/opt/glue/configs/default.yaml \
GLUE_SAMPLES=/opt/glue/samples \
GLUE_DB=/opt/glue/logs/host.db \
./target/release/glue-app
```

> 只想远程看界面时，可在板子上装轻量桌面（`xorg` + `openbox`）或 VNC / `x11vnc`。

## 6. 中文字体

GUI 会自动查找 `/usr/share/fonts/...` 下的 Noto CJK、文泉驿等；若无中文字体（界面显示方块）：

```bash
sudo apt install -y fonts-wqy-microhei        # 或 fonts-noto-cjk
export GLUE_FONT=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
```

## 7. 路径与环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GLUE_CONFIG` | `../configs/default.yaml` | 算法配置 |
| `GLUE_SAMPLES` | `../outputs/rust_samples` | 检测图片目录（GUI 默认目录） |
| `GLUE_DB` | `../outputs/host.db` | SQLite 历史库（GUI） |
| `GLUE_RECIPE` | `../outputs/recipe.json` | 配方文件（GUI） |
| `GLUE_FONT` | 自动查找 | 中文字体路径 |

`glue-agent` 全部参数走命令行，不读上述环境变量。

## 8. 性能验证

```bash
# 板上测传统链路耗时
time ./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out /tmp/r.json
```

预期 1280×960 单帧 20–40 ms（A76 单核约为 x86 的 0.5–0.7 倍，x86 实测 p50 ≈ 8 ms）。
若明显偏慢，先看 CPU 调频（`cpufreq-info`）是否被降频。

## 9. 已知限制

1. **无相机**：当前只做文件回放；接 MVS 相机需要 `glue-camera`（MVS aarch64 SDK + BSP 内核驱动，需单独验证）；
2. **无 NPU 推理**：RPN/YOLO 尚未导出 ONNX/RKNN，板端仅传统算法链路；
3. **GUI 交叉编译**：目前只给"板上原生编译"；Windows 交叉编译 GUI 需要 aarch64 glibc sysroot，未配置；
4. **字体格式**：`.ttc` 集合字体若加载失败，改用 `.ttf`（如 `wqy-microhei.ttf`）并通过 `GLUE_FONT` 指定。

## 10. 后续

- `glue-camera`：MVS FFI，三路采集与硬触发；
- `glue-infer`：ONNX Runtime → RKNN，接 NPU；
- 打包：agent + configs + systemd 打成 tar.gz 或 deb 包。
