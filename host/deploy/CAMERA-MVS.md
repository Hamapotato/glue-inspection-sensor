# 海康（HIKROBOT）相机接入指南（RK3588 / Ubuntu）

> 相机：MV-CB016-10UM-S（USB3 Vision，1.6MP 黑白），USB ID `2bdf:0001`。
> 结论：**不是 UVC**，`/dev/video*` 不是它；必须通过 **MVS SDK** 取图。
> 本仓库通过"命令管道"接入：Rust 启动采集脚本，从 stdout 读 GLUE 帧协议。

## 1. 安装 MVS Linux aarch64 SDK

从海康机器人官网下载中心取 **Linux aarch64** 包（如 `MVS-5.1.0_Linux_aarch64_*`）：

```bash
tar -xzf MVS-*_Linux_aarch64_*.tar.gz
cd MVS-*_Linux_aarch64_*
sudo ./install.sh            # 或 sudo dpkg -i MVS-*.deb

# 环境变量（建议写入 ~/.bashrc）
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH

# udev 规则 + 权限（关键）
sudo find /opt/MVS -name '*.rules' -exec cp {} /etc/udev/rules.d/ \;
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video $USER
```

## 2. 先验证 SDK 能取图（不依赖本仓库）

```bash
# 确认相机被识别（MVS 枚举，不是 /dev/video*）
lsusb | grep -i 2bdf

# SDK 自带示例
find /opt/MVS/Samples -name 'GrabImage*.py' | head
python3 <示例目录>/GrabImage.py     # 有桌面时会弹窗；无桌面可参考示例改成存图
```

如果示例跑不通，先解决 USB3 口/线、udev、设备占用问题，再继续。

## 3. 用本仓库脚本验证

```bash
# 仓库根目录执行；stdout 是帧协议，图片/日志在 stderr
python3 scripts/camera_mvs.py --device 0 --fps 10 > /tmp/raw.bin
# 另开终端看是否持续增长：
ls -l /tmp/raw.bin
```

脚本输出协议（小端）：

```text
b"GLUE" | width:u32 | height:u32 | len:u32 | gray[len]   # Mono8
```

## 4. 接入软件

**配置**（`configs/experiments/camera-mvs.yaml`）：

```yaml
camera:
  enabled: true
  source: command
  command: "python3 scripts/camera_mvs.py --device 0 --fps 10"
```

**无界面连续检测**（结果逐帧写 JSONL）：

```bash
cd /opt/glue     # 或仓库根目录
./bin/glue-agent --config configs/experiments/camera-mvs.yaml \
    --camera --frames 100 --mm-per-px 0.0833 --state logs/camera.jsonl
```

**GUI**：检测页左侧导航新增「相机」页 → 「启动相机」即可预览 + 实时判定
（需要桌面环境，构建方式见 DEPLOY-RK3588.md §5）。

**没有相机时联调**（合成帧，验证链路）：

```bash
./bin/glue-agent --config configs/experiments/camera-synthetic.yaml --camera --frames 5 --state /tmp/syn.jsonl
```

## 4.1 结合 ROI 推理（可选）

相机帧可以直接走 RPN 推理定位，而不是启发式定位：

| 路线 | 配置 | 构建要求 |
| --- | --- | --- |
| CPU（ONNX / tract） | `configs/experiments/camera-rpn.yaml` | 板端原生 `cargo build --release -p glue-agent --features infer`；GUI 默认已编入 |
| NPU（RKNN） | `configs/experiments/camera-rknn.yaml` | 先按 `docs/algo/reports/npu-rknn.md` 转换出 `.rknn`，再 `--features rknn` 构建 |

```bash
# CPU 推理 + 相机
cd /opt/glue
./bin/glue-agent --config configs/experiments/camera-rpn.yaml \
    --camera --frames 100 --mm-per-px 0.0833 --state logs/camera.jsonl
```

前提：`outputs/demo/onnx/rpn.onnx` 与 `anchors_wh` 放在板端可访问路径（配置里的路径相对运行目录）。

> 注意：当前仓库的演示权重是**合成数据**训练的，真实相机图像分布不同，定位效果需要真实数据重新训练/微调后再评估；没有真实数据前，不要把演示权重的检测结果当成现场精度。

## 4.2 获取演示权重（ONNX，不入库）

模型权重与 `outputs/` 全部在 `.gitignore` 里，**clone 仓库后不会有**，需要从构建机传输：

权重不在 git 里（红线 R4），`git pull` 之后用仓库自带脚本取回：

```bash
# 构建机（Windows，仓库根目录）起 HTTP：
py -3 -m http.server 8000 --directory outputs\demo

# 板端（仓库根目录）：
python3 scripts/fetch_demo_weights.py --url http://<构建机IP>:8000
# 会自动下载 outputs/demo/onnx/rpn.onnx 与 anchors.json，并打印校验值与下一步命令
```

也可以给 `run-demo.sh` 传环境变量自动拉取：

```bash
GLUE_DEMO_WEIGHTS_URL=http://<构建机IP>:8000 host/run-demo.sh
```

构建机上的 SHA256：`F85CC1E3B13A8D936B980F159709EE2B55CA3B6CADFE655627418D822C492C91`
（本仓库演示权重，合成数据训练；重新训练会变化）

也可以把 `best.pt` 一起拷走，再用本仓库脚本重新导出 ONNX。

## 5. 常见问题

| 现象 | 处理 |
| --- | --- |
| 枚举到 0 台 | USB3 口/线、直连不用 Hub、`lsusb` 是否出现 `2bdf` |
| Permission denied | udev 规则未装/未 trigger；重插；确认 `video` 组 |
| 打开失败 | MVS 客户端占用相机；先关闭 MVS |
| 脚本 import 失败 | `--sdk` 指定 `MvImport` 目录，或设置 `MVS_PYTHON_PATH` |
| 帧率不足 | 降分辨率/`--fps`；检查 USB3 带宽、线材 |
| 采集脚本退出 | agent 会报"采集失败/EOF"并退出；GUI 状态栏显示错误 |

## 6. 已知限制

1. 当前是**命令管道**后端（Rust 不直接链接 `libMvCameraControl.so`），每帧经 stdout 拷贝；10 fps / 1.6MP 下约 15 MB/s，实测本机 1280×960 合成流整链路 ~16 ms/帧（含检测）；
2. 尚未做硬触发/外部 IO 同步（论文是连续采集 + 在线检测）；
3. 未在真实相机上做长时间稳定性测试（需要板端验证）；
4. 后续可选：直接 MVS FFI（省一次拷贝）或 Aravis/GenICam 后端。
