# RKNN NPU 验证：前置工作清单

> 目标：把 RPN 从 CPU（ONNX/tract）切到 RK3588 NPU（RKNN）。
> 当前状态：**软件侧前置工作已完成；转换与板端实测受外部环境阻塞**（缺 x86_64 Linux 转换环境、板端缺 `librknnrt.so`）。

## 1. 已完成（仓库内可核查）

| 项 | 内容 | 位置 |
| --- | --- | --- |
| 静态 ONNX | RPN 静态 batch 导出（RKNN 要求），输入 `[1,1,320,320]`；算子 `Conv / Mul / Reshape / Sigmoid / Transpose`（均为 RKNN 常规支持算子） | `scripts/export_onnx.py --rpn-static` |
| 转换脚本 | 自动生成 320×320 letterbox 校准集 → `rknn.config(mean=0,std=255)` → 量化 → `.rknn`；转换前检查静态维并打印算子 | `scripts/export_rknn.py` |
| 板端验证脚本 | RKNN vs ONNX：原始 cls/reg 输出误差 + 解码框 IoU；判据 reg 误差 <0.05 | `scripts/verify_rknn.py` |
| 前置检查 | 按角色（转换/推理）检查平台、Python 包、ONNX 静态性、校准集、`librknnrt.so`、NPU 设备、`.rknn` | `scripts/rknn_preflight.py` |
| Rust 运行时 | `rknn` feature：`dlopen("librknnrt.so")`，输入/anchors/解码/NMS 与 ONNX 路径共用；失败回退启发式并写 `notes` | `host/crates/infer/src/rknn.rs` |
| 配置 | `locate.type=rknn`；演示与相机两套配置 | `configs/experiments/demo-rknn.yaml`、`camera-rknn.yaml` |
| 权重获取 | `fetch_demo_weights.py` 同时拉 `rpn.onnx` 与 `rpn-static.onnx` | `scripts/fetch_demo_weights.py` |

验证方式（任意机器可跑，检查本地缺什么）：

```bash
python3 scripts/rknn_preflight.py              # 自动判断转换/板端角色
python3 scripts/rknn_preflight.py --role convert
python3 scripts/rknn_preflight.py --role runtime
```

## 2. 外部依赖（当前缺口）

| 缺口 | 用途 | 解决方式 |
| --- | --- | --- |
| x86_64 Linux + `rknn-toolkit2` | ONNX → RKNN 转换（官方只提供 x86_64 Linux） | 任意 x86 Ubuntu / RKNN docker 镜像；版本需与板端 `librknnrt.so` 匹配 |
| 板端 `librknnrt.so` | RKNN 运行时 | 来自 RKNN runtime / 板卡 BSP，**不是** MVS SDK；或设置 `GLUE_RKNN_LIB` 指向实际路径 |
| 板端 NPU 设备 | 推理执行 | 确认 `/dev/rknpu*` 或 `/dev/dri/renderD*` 存在、驱动正常 |
| `.rknn` 文件 | 板端推理 | 由转换机生成后拷到 `outputs/demo/rknn/rpn.rknn` |

## 3. 执行顺序

```bash
# ① 构建机（Windows）：导出静态 ONNX（已有演示权重时）
python scripts/export_onnx.py --rpn-weights outputs/demo/rpn/best.pt --rpn-static --out outputs/demo/onnx

# ② 转换机（x86_64 Linux）
pip install rknn-toolkit2
python3 scripts/rknn_preflight.py --role convert          # 全部 OK 再继续
python3 scripts/export_rknn.py \
    --onnx outputs/demo/onnx/rpn-static.onnx \
    --out  outputs/demo/rknn/rpn.rknn

# ③ 拷 .rknn 到板端 outputs/demo/rknn/，然后板端：
python3 scripts/rknn_preflight.py --role runtime
python3 scripts/verify_rknn.py \
    --rknn outputs/demo/rknn/rpn.rknn \
    --onnx outputs/demo/onnx/rpn-static.onnx

# ④ 板端构建带 NPU 的软件并运行
cd host && cargo build --release -p glue-agent --features rknn
./target/release/glue-agent --config ../configs/experiments/camera-rknn.yaml \
    --camera --frames 100 --mm-per-px 0.0833 --state ../outputs/camera.jsonl
```

## 4. 验收标准与风险

| 项 | 标准 |
| --- | --- |
| 转换成功 | `export_rknn.py` 输出 `.rknn`，无算子不支持报错 |
| 数值一致 | `verify_rknn.py` 的 reg 最大误差 <0.05（经验阈值，按实际模型校准） |
| 板端可跑 | `preflight --role runtime` 全 OK；`glue-agent --features rknn` 能初始化并出 RoI |
| 性能 | 实测 NPU 延迟（**尚未测量，不写推测数字**） |

风险：
1. 个别算子量化不支持 → 先 `--no-quant` 排查，再调 `quantized_algorithm`/混合量化；
2. 转换工具与板端运行库版本不一致 → `rknn_init` 失败；
3. 校准集目前为合成数据，真实场景建议换现场图像重新量化；
4. 动态 batch 的 ONNX 不能转换，必须用 `--rpn-static` 版本。
