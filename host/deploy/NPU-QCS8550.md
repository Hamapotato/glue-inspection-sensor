# QCS8550（广翼智联 FV04 / AidLux）NPU 接入说明

> 主控：Qualcomm QCS8550（Snapdragon 8 Gen 2 的 IoT 版本）｜系统：AidLux（Android + Ubuntu 容器）｜架构：aarch64
> 结论：**RKNN 不适用**（那是 Rockchip RK3588 的方案）。本板 NPU 要走 Qualcomm 工具链：
> **AidLite（AidLux 自带）**、**SNPE** 或 **QNN / QAIRT** 三选一。

## 1. 当前状态

| 项 | 状态 |
| --- | --- |
| 相机接入（MVS） | ✅ 已通 |
| CPU 推理（ONNX / tract） | ✅ 已通（板端实测定位 p50 23.81 ms、整帧 p50 31.87 ms，见项目进度报告） |
| NPU 推理 | ❌ 未接入；工具链和模型格式都需按 Qualcomm 重做 |
| 之前的 RKNN 工作 | 仅适用 RK3588 板；本板保留但不再作为主路径 |

## 2. 先探测板端已有组件（在 Ubuntu 里执行）

```bash
uname -m
ls /opt
ls /opt/MVS 2>/dev/null | head

# Qualcomm/AidLux 推理组件
python3 -c "import aidlite; print('aidlite:', aidlite.__file__)" 2>&1
python3 -c "import snpe; print('snpe py:', snpe.__file__)" 2>&1
find / -maxdepth 4 \( -iname '*snpe*' -o -iname '*qnn*' -o -iname '*aidlite*' -o -iname '*aimo*' \) 2>/dev/null | head -30
ls /usr/lib /usr/lib/aarch64-linux-gnu 2>/dev/null | grep -Ei 'snpe|qnn|adsprpc|cdsprpc|fastrpc' | head
ls /dev | grep -Ei 'adsprpc|fastrpc|dsp|kgsl' | head

# Android 侧（辅助确认芯片）
getprop ro.board.platform 2>/dev/null
getprop ro.soc.model 2>/dev/null
```

把输出发我，可直接判断走哪条路线。

## 3. 三条可选路线

### A. AidLite（AidLux 自带，最快）

AidLux 通常自带 **AidLite** 推理框架（C/C++/Python），并在其 AI 工具里提供模型转换。
适合快速验证；具体支持的模型格式与转换入口**以板端实际安装为准**（见上面探测输出）。

### B. SNPE（Qualcomm Snapdragon Neural Processing Engine，成熟）

- 转换：在 **x86_64 Linux** 安装 SNPE SDK（需 Qualcomm 账号）：
  ```bash
  snpe-onnx-to-dlc -i rpn-static.onnx -d input:1,1,320,320
  snpe-dlc-quantize --input_dlc rpn.dlc --input_list calib.txt --output_dlc rpn_quant.dlc
  ```
- 板端：部署 `libSNPE.so` 及 HTP 相关库，用 `snpe-net-run` 先验证；再在 Rust 里 FFI 调 SNPE C API（新增 `snpe` feature，结构可参照现有 `rknn` 模块）。
- 要求：ONNX 静态 batch、输入 320×320（`rpn-static.onnx` 已就绪）。

### C. QNN / QAIRT（Qualcomm AI Engine Direct，新）

- 转换（x86_64 Linux，QAIRT SDK）：`qairt-converter` 把 ONNX 转成 QNN context binary；
- 板端：`libQnnHtp.so` 等运行库 + Rust FFI 或 Python 验证；
- 与 SNPE 二选一，取决于你能拿到的 SDK 版本。

## 4. 与本仓库的衔接

- CPU 路径（现状）不改：`glue-agent --config configs/experiments/camera-rpn.yaml` 继续可用；
- NPU 路径建议新增 `snpe`（或 `aidlite`）后端，接口与 `rknn` 一致：
  `ObjectDetector` → letterbox → 推理 → anchors 解码 → NMS → 原图坐标；
- 配置形态：`locate.type = snpe|aidlite`，权重为 `.dlc`/模型文件，`anchors_wh` 与训练一致；
- 验证口径：与 CPU ONNX 输出对比（cls/reg 误差、RoI IoU）；**未实测前不写 NPU 延迟数字**。

## 5. 与 RKNN 的关系

`host/deploy/RKNN-CHECKLIST.md`、`scripts/export_rknn.py`、`verify_rknn.py`、`crates/infer/src/rknn.rs`
都是 RK3588 方案，**本板（QCS8550）不使用**；保留是为了以后换 RK3588 板时可直接复用。
