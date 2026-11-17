# ONNX 导出与数值一致性报告（B1/B2 模型）

> 日期：2026-11-17 ｜ 脚本：`scripts/export_onnx.py` ｜ 代码：`algo/net/rpn.py`、`algo/net/yolo.py`
> 目的：验证两路模型能否导出 ONNX、onnxruntime 推理与 PyTorch 是否一致，为上位机/端侧推理（M3、RK3588）做准备。
> **本报告只验证"可导出 + 数值一致"，不涉及任何检测精度指标。**

## 1. 环境

| 项 | 值 |
| --- | --- |
| PyTorch | 2.15.0+cpu（仓库 `.venv`） |
| onnx | 1.23.2 |
| onnxruntime | 1.32.0（CPUExecutionProvider） |
| onnxscript | 导出器依赖（torch 2.15 新导出路径需要） |
| 导出 opset | 17 |
| 运行设备 | 本机 CPU |

安装（仅导出时需要，不进入运行时依赖）：

```powershell
.venv\Scripts\python.exe -m pip install onnx onnxruntime onnxscript
```

## 2. 导出方式

| 模型 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| RPN | `image (B,1,320,320)` | `cls (B,3600)`、`reg (B,3600,4)` | 只导出网络主体；**anchors 由宿主机按输入尺寸确定性生成**（同 `make_anchors`），不放进图里，避免网格生成算子影响端侧兼容性 |
| YOLO5FL | `image (B,1,128,128)` | `out (B,16368,6)` | 5 个尺度输出在 wrapper 内拼接为单张量；每尺度的 `(A,H*W,5+nc)` 展平后按尺度顺序拼接 |

- 动态轴：仅 batch 维动态（`dynamic_axes`），H/W 固定（端侧按固定分辨率推理）；
- 导出后用 `onnx.checker.check_model` 校验，并把默认的 `.onnx.data` 外部权重**内联成单文件**，便于交付。

## 3. 结果

| 模型 | 参数量 | ONNX 大小 | PyTorch vs onnxruntime 最大绝对误差 |
| --- | --- | --- | --- |
| RPN | 0.99 M | 3.87 MB（单文件） | **2.38 × 10⁻⁶** |
| YOLO5FL（5 层 + SPP） | 6.98 M | 26.6 MB（单文件） | **2.62 × 10⁻⁶** |

- 对拍方式：同一权重、同一随机输入（batch=2，验证动态 batch），比较全部输出张量的 `max|diff|`；
- 结论：**两路模型均可导出 ONNX，onnxruntime 推理与 PyTorch 数值一致（误差 ~1e-6 量级，远小于 1e-4 判据）**；
- 未提供训练权重时导出的是随机初始化模型——本报告只证明"结构可导出、算子兼容、数值一致"，不代表任何检测精度。

## 4. 算子与兼容性观察

- RPN：Conv/BN/SiLU/reshape 全部为标准算子，导出图很小（5 个算子的量级），RKNN 转换风险低；
- YOLO5FL：SPP 中的 `AdaptiveMaxPool2d` 与 `F.interpolate(..., mode="nearest")` 均被导出为静态形状的标准算子（输入 H/W 固定）；FPN 拼接、reshape 正常；
- 输出张量较大（16368×6），端侧建议只取需要的尺度或在后处理中逐尺度处理，避免一次拷贝过多数据。

## 5. 局限与下一步

1. **无训练权重参与**：本次是结构验证。等真实数据训练收敛后，用 `--rpn-weights` / `--yolo-weights` 重新导出并做同样的对拍（脚本已支持）；
2. **未做端侧推理**：Rust 侧 `glue-infer` 仍是空缺；下一步用 `ort` crate 加载 ONNX，在 Rust 中实现解码（RPN 的 anchors 生成与 proposal、YOLO 的 decode/NMS），并与 Python 对拍；
3. **RKNN 未验证**：ONNX → RKNN 转换需在 RK3588 工具链上执行（x86 Linux 上可用 rknn-toolkit2），本机 Windows 未做；
4. **精度评测另算**：ONNX 一致性 ≠ 检测精度，指标评测仍按 `p0-spec.md` 的真实数据流程执行。

## 6. 复现命令

```powershell
# 随机权重结构验证（本次结果）
.venv\Scripts\python.exe scripts\export_onnx.py

# 有训练权重后
.venv\Scripts\python.exe scripts\export_onnx.py --rpn-weights outputs\rpn.pt --yolo-weights outputs\yolo.pt
```

产物默认在 `outputs/onnx/`（已被 `.gitignore` 忽略，不入库）。
