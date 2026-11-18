# RK3588 NPU（RKNN）推理接入

> 日期：2026-11-19 ｜ 目标：把 RPN 从 CPU（tract/ONNX）切到 RK3588 NPU（RKNN）执行。
> 代码：`host/crates/infer/src/rknn.rs`（feature `rknn`）、`scripts/export_rknn.py`、`scripts/verify_rknn.py`；
> 配置：`configs/experiments/demo-rknn.yaml`。

## 1. 架构

```
训练/导出（Windows/PC）:  best.pt → rpn.onnx
转换（x86_64 Linux）    :  rpn.onnx + 校准图 → rknn-toolkit2 → rpn.rknn
板端运行（aarch64）     :  glue-infer(feature=rknn) → librknnrt.so → NPU
                          预处理/anchors/解码/NMS 与 ONNX 路径完全共用
```

- 转换工具 `rknn-toolkit2` **仅支持 x86_64 Linux**；板端 aarch64 只能装 `rknn-toolkit-lite2` 做推理；
- Rust 侧通过 `libloading` 动态加载 `librknnrt.so`，实现 `ObjectDetector`，配置 `locate.type=rknn`；
- 失败（缺库/形状不符）返回错误 → pipeline 回退启发式并写入 `notes`，不静默降级。

## 2. 步骤

1. **生成权重与 ONNX**（Windows，已有演示权重可跳过）：
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\build-demo-weights.ps1
   ```
2. **转换为 RKNN**（x86_64 Ubuntu）：
   ```bash
   pip install rknn-toolkit2      # 版本与板端 librknnrt.so 匹配
   python3 scripts/export_rknn.py --onnx outputs/demo/onnx/rpn.onnx --out outputs/demo/rknn/rpn.rknn
   ```
   - 脚本自动从 `outputs/demo_dataset/train` 生成 50 张 320×320 letterbox 校准图；
   - 量化配置：`mean=0, std=255`，`target_platform=rk3588`，INT8；
   - 排查结构问题时可用 `--no-quant`。
3. **拷到板端**：`rpn.rknn` + `configs/experiments/demo-rknn.yaml`（`anchors_wh` 必须与训练一致）。
4. **板端构建**：
   ```bash
   cd host && cargo build --release -p glue-agent --features rknn
   # 如有需要：export GLUE_RKNN_LIB=/usr/lib/librknnrt.so
   ```
5. **运行**：
   ```bash
   ./target/release/glue-agent --config ../configs/experiments/demo-rknn.yaml \
       --input ../outputs/rust_samples --mm-per-px 0.0833 --out /tmp/rknn.json
   ```
6. **验证**（板端）：
   ```bash
   python3 scripts/verify_rknn.py --rknn outputs/demo/rknn/rpn.rknn --onnx outputs/demo/onnx/rpn.onnx
   ```
   对比 RKNN 与 ONNX 的 cls/reg 原始输出与解码框；判据：reg 最大误差 < 0.05 视为通过（经验阈值，需按模型校准）。

## 3. 当前状态（重要，避免误解）

| 项 | 状态 |
| --- | --- |
| 转换脚本（含校准集生成） | ✅ 已实现，未在真实 x86 Linux 上执行 |
| Rust RKNN 运行时接口 | ✅ 已实现（feature `rknn`），Windows 上通过 `cargo check/clippy --features rknn` |
| 板端验证脚本 | ✅ 已实现，需板端 + `rknn-toolkit-lite2` |
| `.rknn` 产物 | ❌ 未生成（本机没有 x86_64 Linux 与 rknn-toolkit2） |
| 板端实测性能（NPU 延迟） | ❌ 未测量，**不写任何推测数字** |
| 量化精度损失 | ❌ 未验证 |

> 红线：没有板端实测前，不把 NPU 延迟/精度写成结论；本报告只声明"接口与流程已就绪"。

## 4. 风险与排查

1. **算子支持**：RPN 含卷积/上采样/拼接等；若 INT8 量化对某算子不支持，`rknn.build` 会报错。处理：`--no-quant` 先确认结构，再逐步改混合量化（`quantized_dtype`/`quantized_algorithm`）。
2. **动态尺寸**：ONNX 导出时 RPN 是动态 batch；RKNN 转换建议固定 batch=1（脚本按静态 320×320 处理，若报动态维度需先导出静态版）。
3. **输出顺序**：RKNN 输出顺序需与 ONNX 一致（cls、reg）；`verify_rknn.py` 会打印输出数量，异常时检查导出包装器。
4. **运行库版本**：`librknnrt.so` 与 toolkit2 版本必须匹配，否则 `rknn_init` 返回错误（pipeline 会写 `notes` 并回退）。
5. **量化校准集**：必须与真实场景分布接近；当前用合成数据，只能验证流程，不能代表真实精度。

## 5. 下一步

1. 在一台 x86_64 Ubuntu（或 RKNN docker 镜像）执行转换，产出 `rpn.rknn`；
2. 板端跑 `verify_rknn.py`，记录 reg 误差与实际 NPU 延迟（`perf`/自带 profiling）；
3. 若 RPN 量化可用，把 YOLO 反光模型也转成 RKNN（同脚本改输入即可）；
4. 把 NPU 路径接进 systemd 常驻服务与 GUI（配置切换 `locate.type`）。
