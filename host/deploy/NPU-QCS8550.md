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

## 2.1 板端探测结果（2026-11-22，板端实测）

| 检查项 | 结果 |
| --- | --- |
| 架构 | aarch64 |
| AidLite Python | ✅ `/usr/local/lib/python3.10/dist-packages/aidlite`，SDK V2.5.1.304（ABI 0x2050100） |
| QNN 运行库 | ✅ `/vendor/lib64/libQnnHtp.so`、`libQnnHtpV73Stub.so`、`libQnnSystem.so` |
| HTP 架构 | V73（对应 QCS8550；V75/V79 也在） |
| AidLite 后端 | `qnn231 / qnn236 / qnn240 / qnn248 / snpe229 / onnx / tflite`（`/usr/local/lib/libaidlite_*.so`） |
| DSP/NPU 通路 | ✅ `libcdsprpc.so`、`adsprpc-smd`、`kgsl-3d0` |
| SNPE Python | ❌ 未安装（不需要，AidLite 内部有 snpe229 后端） |

AidLite 示例（`/usr/local/share/aidlite/examples/`）：

| 示例 | 模型格式 |
| --- | --- |
| `aidlite_onnx` | `.onnx`（CPU/GPU 后端） |
| `aidlite_qnn231` / `qnn248` | `.ctx.bin`（QNN context binary，NPU） |
| `aidlite_snpe229` | SNPE `.dlc` |
| `aidlite_tflite` | `.tflite` |

结论：**AidLite 路线可行**；NPU 需要 QNN `.ctx.bin` 模型（示例里是预编译的 yolov5），
ONNX 直接走 AidLite 是 CPU/GPU 后端，不是 NPU。

## 2.2 下一步需要确认（板端执行后回传）

```bash
sed -n '1,160p' /usr/local/share/aidlite/examples/aidlite_onnx/python/onnx_yolov5_multi.py
sed -n '1,120p' /usr/local/share/aidlite/examples/aidlite_qnn248/python/qnn_yolov5_multi.py
cat /usr/local/share/aidlite/examples/aidlite_onnx/Readme.txt
cat /usr/local/share/aidlite/examples/aidlite_qnn248/Readme.txt
ls /usr/local/bin | head -50
find / -maxdepth 6 \( -iname '*qairt*' -o -iname '*qnn*converter*' -o -iname '*aidgen*' \) 2>/dev/null | head -20
nm -D --defined-only /usr/local/lib/libaidlite.so 2>/dev/null | head -40
```

重点确认：
1. AidLite Python API 的调用顺序（构造 → init → set_input → run → get_output）；
2. 输入布局与数值范围（NCHW/NHWC、0-255 还是 0-1）；
3. **是否有本机 ONNX → QNN `.ctx.bin` 的转换工具**（`aidgen`/`qairt-converter`），
   若没有，则 NPU 模型需在 x86_64 Linux + QNN SDK 上转换。

## 2.3 进一步确认（2026-11-22 第二批）

| 发现 | 含义 |
| --- | --- |
| `aidlite` C++ API：`Aidlux::Aidlite::Interpreter`（init/load_model/set_input_tensor/invoke/get_output_tensor） | 运行时 API 明确；Python 为其绑定 |
| `/usr/local/include/aidlux/aidgen/aidgen.hpp` + `libaidgen*.so` | AidLux 自带 **aidgen** 模型生成/转换组件（疑似 ONNX→QNN 转换能力） |
| `/usr/local/lib/aidlux/qairt` | 板端自带 **QAIRT/QNN** 组件目录 |
| `/usr/local/bin/oma` | AidLux 的模型工具 CLI（需 `oma --help` 确认功能） |
| `libaidlite_qnn248.so` / `qnn231` / `snpe229` | AidLite 已有 QNN HTP 后端 |

仍待确认（决定"板端能否直接转 NPU 模型"）：

## 2.4 第三批结论（2026-11-22）

| 项 | 结论 |
| --- | --- |
| Python API | 确认：`Config/Model/InterpreterBuilder.build_interpreter_from_model_and_config` → `init/load_model` → `set_input_tensor(name_to_index(...))` → `invoke` → `get_output_tensor(name_to_index(...), TYPE_FLOAT32)` → `destroy` |
| QNN 运行时 | 板端 `qairt/qnn236|qnn240|qnn248` 目录下是完整运行时（含 `libQnnHtpV73Skel.so` 等），**可直接跑 `.ctx.bin`** |
| 板端转换工具 | ❌ 未发现：`oma` 是运维代理（Operation Maintenance Agent），`aidgen` 是 LLM 组件（t2t/t2e），qairt 目录只有运行时 |
| ONNX → ctx.bin | 需要 x86_64 Linux + QNN SDK（qairt-converter / qnn-context-binary-generator），或 AidLux 提供的在线转换服务 |
| AidLite ONNX 后端 | ✅ 可直接跑 `rpn-static.onnx`（CPU/GPU），用于先验证 API 与模型 |

新增验证脚本：`scripts/aidlite_rpn_test.py`；完整测试流程见 [AIDLITE-TEST-PLAN.md](AIDLITE-TEST-PLAN.md)（可直接交给板端智能体执行）。（板端运行，打印张量信息、单帧耗时与解码后的前 5 个 RoI）。

## 2.5 与 RK3588 的关系（澄清 GPU/NPU）

| 平台 | GPU | NPU | 说明 |
| --- | --- | --- | --- |
| RK3588 | **有**：ARM Mali-G610 MP4（OpenGL ES 3.2 / OpenCL 2.2 / Vulkan 1.2） | 6 TOPS，RKNN 工具链 | "RK3588 没有 GPU"是误解；只是没有 CUDA，GPU 用 OpenCL/Vulkan/GL |
| QCS8550（FV04） | Adreno 740（OpenCL/Vulkan） | Hexagon HTP V73，SNPE/QNN/AidLite | 本板当前用于验证 NPU 可行性与软件链路 |

结论：
1. 本板（QCS8550）验证的是 **"NPU 推理链路可行 + 软件架构可复用"**；
2. 迁移到 RK3588 时，模型转换/运行库换成 RKNN（仓库已有 `rknn` feature 备用），
   `ObjectDetector` 接口、相机链路、判定逻辑、配置形态都不变；
3. 两块板的 GPU 都可用，但本项目实时性瓶颈在推理，GPU 不是主路径。

## 2.6 NPU 可行性验证实验（官方模型，CPU vs GPU vs NPU）

板端自带同一 YOLOv5 的 ONNX 与 QNN ctx.bin 示例，可直接对比：

```bash
# CPU / GPU（AidLite ONNX 后端）
cd /usr/local/share/aidlite/examples/aidlite_onnx/python
python3 onnx_yolov5_multi.py --cpu    # 若脚本支持参数；否则按 Readme 调用
python3 onnx_yolov5_multi.py --gpu

# NPU（AidLite QNN248 后端，ctx.bin）
cp -r /usr/local/share/aidlite/examples/aidlite_qnn248 /tmp/
cd /tmp/aidlite_qnn248/python && python3 qnn_yolov5_multi.py
```

记录每个示例打印的 `invoke cost time`（单位 ms），即可得到本板
"CPU / GPU / NPU" 的推理耗时对比——这是"NPU 是否可行"的直接证据。
（注意：不同模型/输入尺寸不能直接比，必须是同一模型。）

## 2.7 板端实测结果（2026-11-22，首次执行）

| 测试 | 结果 | 说明 |
| --- | --- | --- |
| A. 我们的 RPN（AidLite ONNX CPU） | `rpn-static.onnx` load ret=0；输入 `[1,1,320,320]`，输出 `cls [1,3600]` / `reg [1,3600,4]`；invoke p50 **12.96 ms** | 首次报告的"随机权重"是**测试脚本 anchor 顺序 bug** 导致解码异常，模型本身正常（修正后在 PC 复算 top1 score=0.9997，框 y≈419.8~540.3，正好覆盖胶条） |
| B. 官方 YOLOv5 CPU（AidLite ONNX，640） | 10 次 invoke 合计 1721.75 ms → 平均 **172.18 ms/次** | 检出 4 个目标 |
| C. 官方 YOLOv5 NPU（AidLite QNN248 / DSP） | 10 次 invoke 合计 22.70 ms → 平均 **2.27 ms/次**；检出 4 个目标，与 CPU 一致 | Device `type[0] cores_num[1]`；未使用回退；单次 0.002 ms 是异步提交，按 10 次总和取均值才可靠 |
| GPU（AidLite ONNX） | 失败：`in_tensor_name invalid` | 非关键路径，后续再调 |

**结论**：

1. **NPU 在本板可用**（QNN248 + HTP V73），官方同模型上 NPU 相对 CPU **约 75.9×**（172.18 / 2.27 ms）；
2. **我们的 RPN 可以走 AidLite ONNX 后端**（静态模型 12.96 ms CPU）；真实权重需用 `export_onnx.py --rpn-static` 重新导出静态版（动态 batch 不被 AidLite 接受）；
3. **我们的 RPN 上 NPU 的阻塞点仍是 `.ctx.bin` 转换**（板端无转换工具，需 x86_64 Linux + QNN SDK）。

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
