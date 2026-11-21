# glue-infer（Rust ONNX 推理）

用 **tract-onnx（纯 Rust）** 加载 B1/B2 导出的 ONNX，在 Rust 侧完成推理与后处理（anchors、解码、NMS），不依赖 onnxruntime 动态库，便于 Windows 与 RK3588（aarch64）两端复用。

## 1. 依赖与说明

- `tract-onnx 0.21`；由于 `kstring 2.0.5` 要求 rustc ≥1.96，本仓库将 `kstring` 固定为 **2.0.2**（见 `Cargo.lock`），适配当前 rustc 1.95；
- 模型由 `scripts/export_onnx.py` 导出（RPN 动态 batch；YOLO5FL **静态 batch**，因为 tract 对动态 `Resize` 的符号推导不完整）；
- 本 crate 只做推理，不做判定；判定仍在 `glue-core`。

## 2. 用法

```powershell
cargo build --release -p glue-infer

# 原始输出（对拍用）：输入为行优先 f32 raw，输出 raw + 元数据 JSON
.\target\release\glue-infer.exe run --model ..\..\outputs\onnx\rpn.onnx --input ..\..\outputs\infer_parity\rpn_in.f32 --in-shape 320x320 --out out.f32 --meta out.json

# RPN：cls/reg → 候选框 JSON
.\target\release\glue-infer.exe rpn --model ..\..\outputs\onnx\rpn.onnx --input in.f32 --in-shape 320x320 --out boxes.json

# YOLO：输出 → 检测框 JSON
.\target\release\glue-infer.exe yolo --model ..\..\outputs\onnx\yolo5fl.onnx --input in.f32 --in-shape 128x128 --out dets.json
```

## 3. 与 onnxruntime 的对拍（E-06 验收）

`.venv` 下运行 `scripts/check_infer_parity.py`（自动生成固定随机输入，分别跑 Rust 与 onnxruntime 并比较）：

| 模型 | 输出形状 | max\|rust-ort\| |
| --- | --- | --- |
| RPN | `[1,3600]` + `[1,3600,4]` | **4.77e-07 / 2.27e-07** |
| YOLO5FL | `[1,16368,6]` | **5.81e-07** |

结论：**远小于 1e-5 对拍判据**（实测 < 1e-6），推理数值一致。

## 4. 接入 pipeline

`glue-core` 定义 `ObjectDetector` trait；`glue-pipeline::Inspector::with_detectors` 注入；
`cli` / `agent` / `app` 通过 `glue_infer::detectors_from_config(&cfg)` 按配置构建，推理不可用时打印告警并回退启发式（原因写入 `FrameResult.notes`）。

`configs/default.yaml` 中打开（默认关闭）：

```yaml
locate:
  enabled: true
  type: rpn
  weights: outputs/onnx/rpn.onnx   # 训练后的权重
  input_size: 320
  score_thr: 0.5
  anchors_wh: [[142.5,30.0], ...]   # K-means 先验框，必须与训练一致（可选）
reflect:
  enabled: true
  type: yolo5fl
  weights: outputs/onnx/yolo5fl.onnx
  input_size: 128
  score_thr: 0.3
```

冒烟结果（**随机权重**，只验证链路，不代表精度）：
- 仅开 RPN：每帧输出约 240–280 个候选框 → 判定 BROKEN（随机模型的预期表现）；
- 仅开 YOLO：正常/过窄/过宽 → `REFLECTION_SUSPECT`，断胶仍 `BROKEN`，无胶仍 `NO_ROI`，坏图仍 `NO_QUALITY`；
- 训练后的权重接入后，才可做"判定级"对拍（E-08）。

## 5. RKNN（可选 feature `rknn`，RK3588 NPU）

`glue-infer` 支持用 RKNN 运行库执行同一套 RPN 推理（预处理、anchors、解码、NMS 与 ONNX 路径共用）：

```bash
# 板端原生构建（需 librknnrt.so 与 aarch64 工具链）
cargo build --release -p glue-agent --features rknn

# 指定运行库路径（可选）
export GLUE_RKNN_LIB=/usr/lib/librknnrt.so
```

配置：`locate.type = rknn`，`weights` 指向 `.rknn`，`anchors_wh` 必须与训练一致（参考 `configs/experiments/demo-rknn.yaml`）。

转换与验证（x86_64 Linux 转换、板端验证）：

```bash
# x86_64 Ubuntu，pip install rknn-toolkit2
python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn.onnx --out outputs/demo/rknn/rpn.rknn

# 板端，pip install rknn-toolkit-lite2 onnxruntime
python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn --onnx outputs/demo/onnx/rpn.onnx
```

失败行为：运行库缺失、模型形状不符时 `detect()` 返回错误，pipeline 回退启发式并把原因写入 `FrameResult.notes`（不静默降级）。

## 5.1 AidLite 后端（feature `aidlite`，QCS8550/AidLux）

AidLite 只有 C++/Python 接口，Rust 侧通过**常驻 Python 子进程**调用（`scripts/aidlite_server.py`，GLUE 二进制协议），
预处理/anchors/解码/NMS 与 ONNX 路径共用，便于直接对比耗时。

```bash
# 板端构建（需 aidlite Python 包）
cargo build --release -p glue-cli --features aidlite
cargo build --release -p glue-agent --features aidlite          # 无界面
cargo build --release -p glue-app --features aidlite            # GUI
```

配置：`locate.type = aidlite`，`weights` 指向静态 ONNX，`anchors_wh` 与训练一致
（`configs/experiments/demo-aidlite.yaml`、`camera-aidlite.yaml`）。

环境变量：
- `GLUE_AIDLITE_CMD`：完整命令（默认 `python3 scripts/aidlite_server.py --model <weights> --accel cpu`）；
- `GLUE_AIDLITE_ACCEL=cpu|gpu`；
- `GLUE_AIDLITE_PYTHON` / `GLUE_AIDLITE_SCRIPT`：分别覆盖解释器与脚本路径。

## 6. 已知限制

1. **未接入 pipeline**：当前 `glue-vision::locate` 仍是启发式；RPN 的 ONNX 特征图尚未替换它（下一步：`Detector` trait + 配置开关）；
2. **YOLO 静态 batch**：端侧按单帧推理，暂不支持动态 batch；
3. **后处理 NMS 为 IoU 版本**：RPN 的 CIoU-NMS（论文 3.1.4）与 YOLO 的 0.45 阈值目前按 IoU 近似，后续可按论文补 CIoU；
4. **未在 RK3588 实测**：tract 纯 Rust 可交叉编译到 aarch64，但性能未测（需板端验证）。
