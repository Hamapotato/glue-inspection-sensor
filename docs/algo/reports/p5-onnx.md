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

## 4.1 Rust 端推理与对拍（E-06）

`host/crates/infer`（`glue-infer`，tract-onnx 纯 Rust）加载上述 ONNX 并实现：
anchors 生成、RPN 解码 + NMS、YOLO 五尺度解码 + NMS、命令行原始输出。

对拍方式：`scripts/check_infer_parity.py` 生成固定随机输入，分别跑 Rust 与 onnxruntime，逐张量比较：

| 模型 | 输出 | max\|rust-ort\| |
| --- | --- | --- |
| RPN | `[1,3600]`、`[1,3600,4]` | **4.77e-07 / 2.27e-07** |
| YOLO5FL | `[1,16368,6]` | **5.81e-07** |

结论：**Rust 与 onnxruntime 数值一致（<1e-6），远优于 1e-5 判据**；YOLO 采用静态 batch 导出（tract 不支持动态 Resize 符号推导）。

## 5. 局限与下一步

1. **无训练权重参与**：本次是结构验证。等真实数据训练收敛后，用 `--rpn-weights` / `--yolo-weights` 重新导出并做同样的对拍（脚本已支持）；
2. **已接入 pipeline**：Rust 侧 `glue-infer` 通过 `locate.enabled` / `reflect.enabled` 接入（失败回退启发式并写 `FrameResult.notes`）；判定级对拍待训练权重（E-08）；
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

## 7. 推理性能基线（CPU，tract 纯 Rust）

> 目的：回答"带推理后还能不能实时"。**随机权重**只测架构成本，不代表精度；样本为 20 张合成图 × 10 轮 = 200 帧/组，输入 1280×960 灰度。

环境：Intel i7-13650HX（20 线程）、Windows x86_64、`tract-onnx 0.21`（release 构建）、ONNX 由 `scripts/export_onnx.py` 随机权重导出。
配置：`score_thr=0.1`（随机权重下让候选框路径真正执行，属于偏保守的成本上界）。

| 配置 | p50 | p95 | 最大 | 吞吐 |
| --- | --- | --- | --- | --- |
| 基线（传统链路 + 质量门控） | **10.13 ms** | 20.74 ms | 23.58 ms | **97.8 fps** |
| + RPN 定位（320×320，0.99M 参数） | **34.43 ms** | 41.32 ms | 54.64 ms | 37.8 fps |
| + YOLO 反光（128×128，6.98M 参数） | **50.06 ms** | 63.38 ms | 147.42 ms | 25.5 fps |
| 两者都开 | **74.20 ms** | 81.57 ms | 178.65 ms | 18.2 fps |

成本拆解（与上表吻合）：

| 环节 | 增量 |
| --- | --- |
| 传统链路（质量门控 2.6 + 定位 6.0 + 量测 4.0） | ~10 ms |
| RPN 推理 + 解码/NMS | ~24 ms |
| YOLO 推理 + 解码/NMS | ~40 ms |
| 首帧模型加载（RPN / YOLO） | 45.5 ms / 135.5 ms（含加载） |

结论：

1. **传统链路满足 30 fps**（p50 10 ms、p95 20.7 ms）；
2. **CPU 上带推理不满足 30 fps**：RPN 单独 p95 已 41 ms，双开 p95 82 ms；
3. 板端 aarch64 CPU 通常更慢（约 1.5–3×），**正式实时必须走 RKNN/NPU 或降输入/量化**；
4. 随机权重候选框偏多，训练权重 + `score_thr=0.5` 只能小幅降低解码/NMS 成本，卷积仍是主要开销。

复现（配置文件示例，放在 `outputs/` 下，不入库）：

```yaml
locate:  {enabled: true, type: rpn, weights: outputs/onnx/rpn.onnx, input_size: 320, score_thr: 0.1}
reflect: {enabled: true, type: yolo5fl, weights: outputs/onnx/yolo5fl.onnx, input_size: 128, score_thr: 0.1}
```

```powershell
\host\target\release\glue-cli.exe bench --config outputs\perf-both.yaml --input outputs\rust_samples --iters 10
```

限制：以上为**本机 x86 CPU** 实测，不能直接外推到 RK3588；随机权重与训练权重在同一结构下的卷积耗时应接近，但候选框数量会变，解码/NMS 成本会略降。
