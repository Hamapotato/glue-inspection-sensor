# 上位机软件架构设计（Rust）

> 版本 v0.1 ｜ 2026-11-15 ｜ 状态：**设计稿，未开始编码**
> 参考对象：老师交付的 GlueInspect 课题验证包（`视觉检测_课题交付_20261102`，.NET 10 / WPF）
> 相关文档：[产品开发路线](product-development-roadmap.md)、[RK3588 部署调研](rk3588-deployment-research.md)、[复现方案](algo/plan.md)

---

## 0. 决策摘要（ADR）

| 编号 | 决策 | 理由 | 备选 |
| --- | --- | --- | --- |
| ADR-1 | 上位机用 **Rust** | 无 GC、单二进制、可交叉编译 aarch64、延迟可预测；算法密集场景收益大 | C# / .NET（参考软件）、Python + Qt |
| ADR-2 | 界面用 **egui**（后续可换 Tauri） | 纯 Rust、单进程、即时模式适合数据密集 HMI；无需 WebView 运行时 | Tauri、Slint、Iced |
| ADR-3 | **训练用 C++（libtorch），部署用 Rust** | 训练侧统一到 C++ 工程，避免 Python 运行时依赖；libtorch 提供 autograd/优化器/序列化，无需自研框架 | 全 Python 训练、全 Rust 自研 autograd |
| ADR-4 | 相机走 **MVS C API FFI** | 沿用现有海康 USB3 相机，不换硬件 | GenICam / Aravis、UVC |
| ADR-5 | 对外通信：**千兆以太网 JSON + 现场总线网关** | 与 roadmap 一致，PLC 协议不自己实现 | 直连 Profinet/DeviceNet |

> **决策状态（2026-11-15 确认）**：ADR-1 Rust ✅、ADR-2 egui ✅、首版范围见 §2.3、不接真实 PLC/机器人 IO（用模拟信号）✅。
> 仍开放：界面是否与 GlueInspect 保持一致、训练机 GPU、训练迁移边界、Python 侧去留。

---

## 1. 背景

### 1.1 项目上下文

涂胶检测传感器：三台海康 USB3 相机（1280×960，24 V 硬触发）+ 环形 LED + 计算单元。算法复现论文（RPN 定位、YOLO 反光、传统量测），最终形态是"传感器内置计算 + 工控机上位机 + 现场总线"。

上位机承担：**实时显示、配方管理、记录与统计、报警与追溯、批量评测**。

### 1.2 参考软件 GlueInspect 是什么

| 项目 | 内容 |
| --- | --- |
| 形态 | Windows 桌面程序，自带 .NET 运行时与 32 张演示图 |
| 技术栈 | .NET 10 + WPF + CommunityToolkit.Mvvm + OpenCvSharp + ONNX Runtime + SQLite |
| 算法 | Python 训练（U-Net 分割 + PatchCore 异常）→ 导出 ONNX；C# 侧规则量测与判定 |
| 功能 | 检测（单张/批量）、配方示教与标定、历史、统计、设置、命令行评测 |
| 架构 | `Core`（纯领域）/ `Vision`（OpenCV）/ `Inference`（ONNX）/ `Pipeline`（编排）/ `Storage`（SQLite）/ `App`（WPF）/ `Bench`（CLI 评测） |
| 结果 | 合成数据：正常检出率 100%、误报 0%、胶宽误差 0.011 mm；变差成像检出率 95.2% |
| 局限 | 只处理图片文件（不接相机/PLC）；5 MP 图 CPU 1.5 s/张，未达 1 s 目标 |

**值得借鉴**：分层与依赖方向、配方/结果数据契约、"演示与评测同一份代码"、批量评测工具、历史/统计页面划分。
**不照搬**：算法链路（U-Net+PatchCore ≠ 我们的 RPN+YOLO+传统量测）、WPF 技术栈、图片文件输入假设。

### 1.3 与参考软件的核心差异

| 维度 | GlueInspect | 本项目上位机 |
| --- | --- | --- |
| 输入 | 图片文件 | **三路相机实时流 + 文件回放** |
| 触发 | 无 | 机器人 24 V 硬触发，三路同步 |
| 算法 | U-Net 分割 + PatchCore | 传统量测 + RPN 定位 + YOLO 反光 |
| 定位 | 模板匹配 | RPN（论文 3.1） |
| 输出 | 界面 + 记录 | 界面 + 记录 + **报警 IO / 现场总线** |
| 部署 | Windows x64 | Windows x64（首版）+ **aarch64 / RK3588**（目标） |
| 语言 | C# | **C++（训练，libtorch）**、Rust（部署与量测）、Python（可选对照/工具） |

---

## 2. 目标与非目标

### 2.1 目标

1. 三路相机 30 fps 稳定采集，单帧端到端 ≤ 33 ms（NPU 推理 + Rust 量测）；
2. 完整业务闭环：配方示教 → 检测 → 判定 → 显示 → 记录 → 统计 → 导出；
3. 与 Python 算法**双实现对照**，同一张图结果一致（宽度差 ≤ 1 px，判定一致）；
4. 单文件发布（Windows），交叉编译到 aarch64；
5. 具备批量评测能力，指标口径与 `p0-spec.md` 一致。

### 2.2 非目标（首版不做）

- 云端平台、多机协同、远程运维；
- 自研 PLC 协议栈（走网关模块）；
- 云端训练平台（训练在本机 C++ 工程内完成）；
- 3D 胶条、多胶型（先胶条直线/曲线，后扩展）。

---

### 2.3 首版范围（MVP，已确认）

**做**：

| 项 | 内容 |
| --- | --- |
| 数据源 | **文件回放**：单图/目录/拖拽；不接相机 |
| 检测页 | 单帧与批量检测、叠加显示（RoI/掩膜/骨架/宽度线）、判定与耗时 |
| 配方页 | 新建/编辑/保存配方：检测位置（ROI）、标准宽度、预警/报警线、标定系数、模型版本 |
| 历史页 | 检测记录列表、筛选、详情（叠加图、测量值、耗时）、导出 CSV |
| 统计页 | 检出率、误报率、宽度分布、耗时 p50/p95 |
| 设置页 | 算法参数、存储路径、日志级别、语言 |
| 模拟 IO | 判定为缺陷时界面报警条 + 事件日志；不接真实 IO/PLC |
| 评测 | `glue-cli` 批量评测，输出与 `p0-spec.md` 同口径报告 |

**不做（本版）**：真实相机采集与触发、PLC/机器人通信、软件触发录像、多工位切换、远程运维、模型训练（训练在 `train-cpp/` 独立工程）。

**验收**：
1. 双击可运行，无外部运行时依赖；
2. 加载 `configs/default.yaml` + 一批合成图，能出检测结果、历史、统计；
3. 与 Python 端同一批图对照：判定一致率 100%，宽度差 ≤ 1 px；
4. 单帧（1280×960 合成图）≤ 100 ms（CPU 基线）。

## 3. 需求

### 3.1 功能需求

| 编号 | 需求 | 说明 |
| --- | --- | --- |
| FR-1 | 相机采集 | 三路 USB3，硬触发，支持连续采集与软触发；掉线自动重连 |
| FR-2 | 实时检测 | 定位 → 分割 → 量测 → 判定，叠加显示（RoI、掩膜、骨架、宽度线、缺陷框、反光框） |
| FR-3 | 配方管理 | 新建/复制/编辑/导入导出；含检测位置、标准宽度、阈值、标定系数、模型版本 |
| FR-4 | 标定 | 棋盘格标定参数导入；像素→毫米映射按位置保存 |
| FR-5 | 记录与追溯 | 每帧结果、耗时、图片存档路径、操作员、工件号 |
| FR-6 | 统计 | 班次/批次检出率、误报率、宽度分布、耗时分布 |
| FR-7 | 报警 | 缺陷时界面提示 + 可选 IO 输出；反光可疑时提示调参而非判废 |
| FR-8 | 批量评测 | 选目录跑评测，输出检出率/误报率/MAE/耗时报告（M1–M6 口径） |
| FR-9 | 设置 | 相机参数、算法参数、存储路径、日志级别、语言 |
| FR-10 | 回放 | 支持图片目录与录像回放，用于调试与复现 |

### 3.2 非功能需求

| 编号 | 指标 | 目标 |
| --- | --- | --- |
| NFR-1 | 单帧耗时 | 量测链路 ≤ 15 ms；含推理 ≤ 33 ms（CPU 基线可放宽到 100 ms） |
| NFR-2 | 吞吐 | 三路 × 30 fps 不丢帧；批量模式多核并行 |
| NFR-3 | 启动时间 | ≤ 2 s（冷启动） |
| NFR-4 | 内存 | 常驻 ≤ 500 MB（含模型） |
| NFR-5 | 稳定性 | 连续 8 h 无崩溃、无内存增长；相机掉线 3 s 内恢复 |
| NFR-6 | 部署 | 单 exe（Windows），无外部运行时依赖 |
| NFR-7 | 可移植 | 同一源码交叉编译 aarch64 运行于 RK3588 |
| NFR-8 | 可观测 | 结构化日志（帧号、耗时、判定、错误码） |

### 3.3 约束

- 相机：海康 MVS SDK（Windows/Linux 双版本），C API；
- 无商业视觉库（不用 Halcon/VisionPro/Cognex）；
- 配置与算法参数必须外置，不硬编码；
- 所有指标定义以 `docs/algo/p0-spec.md` 为准，不得为"好看"改口径。

---

## 4. 总体架构

### 4.1 分层

```
┌────────────────────────────────────────────────────────────────┐
│  glue-app（egui 桌面）             glue-cli（命令行）           │
│  检测 / 配方 / 历史 / 统计 / 设置    inspect / gen / bench       │
└──────────────┬──────────────────────────────────┬──────────────┘
               │                                  │
      ┌────────▼─────────┐              ┌─────────▼──────────┐
      │  glue-pipeline   │              │    glue-store      │
      │ 编排/并行/计时    │              │ SQLite 记录与统计   │
      └───┬──────┬───────┘              └─────────┬──────────┘
          │      │                                │
  ┌───────▼──┐ ┌─▼─────────┐            ┌─────────▼──────────┐
  │glue-vision│ │glue-infer │            │   glue-camera      │
  │纯 Rust 视觉│ │ONNX Runtime│           │ MVS FFI / 文件回放  │
  └───────┬──┘ └─┬─────────┘            └─────────┬──────────┘
          └──────┴──────────────┬─────────────────┘
                                │
                       ┌────────▼────────┐
                       │   glue-core     │
                       │ 类型/配置/判定/配方│
                       └─────────────────┘
```

依赖规则：**只能自上向下引用**；`glue-core` 不依赖图像库；`glue-vision` 不依赖推理；`glue-app` 不直接调相机（通过 `glue-camera` 的 trait）。

### 4.2 数据流（一帧）

```
硬触发 → 相机回调(3 线程) → 帧队列(按时间戳配对 3 路)
   → 检测线程池(rayon) → 流水线：
        图像检查 → 定位(RPN/阈值) → 裁剪 → 分割(Otsu) → 形态学
        → 连通域/骨架 → 宽度 → 标定换算 → 判定(含反光复判)
   → 结果队列 → UI 线程(显示) + 存储线程(SQLite/图片存档) + 报警输出(IO)
```

### 4.3 线程模型

| 线程 | 数量 | 职责 | 备注 |
| --- | --- | --- | --- |
| 采集 | 3（每相机 1） | MVS 取流、时间戳、丢帧统计 | 阻塞式 SDK 调用，独立栈 |
| 检测 | N = CPU 核数（rayon 池） | 流水线计算 | 单帧无共享可变状态 |
| 推理 | 1–2 | ONNX Runtime session | CPU 推理串行化，NPU 后可并行 |
| UI | 1 | egui 渲染与交互 | 只读最新结果，不阻塞 |
| 存储 | 1 | 异步写 SQLite、存图 | 批量提交，WAL 模式 |
| 看门狗 | 1 | 超时、断线重连、内存监控 | 定时器 |

队列策略：采集→检测用**有界队列**（丢最旧帧，避免延迟累积）；检测→UI 只保留最新；检测→存储用无界但批量。

### 4.4 错误处理与降级

| 故障 | 行为 |
| --- | --- |
| 相机掉线 | 标记 `CAMERA_LOST`，3 s 内自动重连，期间不产出结果 |
| 模型缺失/加载失败 | 启动时告警；定位回退启发式、反光回退空实现；结果 details 记录降级原因 |
| 标定缺失 | 只输出像素宽度，判定标注"未标定"，不伪造毫米值 |
| 单帧超时（> 200 ms） | 丢弃该帧并记录，不阻塞后续帧 |
| 磁盘满 | 停止存图，仅保留数据库记录并告警 |

---

## 5. crate 设计

| crate | 职责 | 对外接口（要点） | 依赖 | 测试 |
| --- | --- | --- | --- | --- |
| `glue-core` | 类型、配置、配方、判定规则 | `Config::load`、`decide()`、`FrameResult` | serde | 判定分支、配置解析 |
| `glue-vision` | 图像算法（Otsu/形态学/连通域/骨架/宽度/标定映射） | `GrayImage`、`measure_roi()`、`locate()` | core | 合成图已知宽度 |
| `glue-infer` | ONNX 推理（RPN 特征头 / YOLO 反光，tract 纯 Rust） | `run`（原始输出）、`rpn`/`yolo`（解码+NMS）；已与 onnxruntime 对拍 <1e-6 | core, tract-onnx | 与 Python 对拍（已过）；pipeline 集成待做 |
| `glue-camera` | 相机抽象：MVS 实现 + 文件/视频回放 | `FrameSource` trait、`CameraConfig` | core | 回放源可测；FFI 单独测 |
| `glue-pipeline` | 编排、并行、计时、降级 | `Inspector::inspect()`、`inspect_batch()` | vision, infer, core | 端到端合成图 |
| `glue-store` | SQLite 记录/统计（已实现） | `Store::insert/recent/stats` | core | 内存库测试 |
| `glue-app` | egui 界面（已实现：检测/配方/历史/统计/设置） | 无 | pipeline, store | 状态单测 + 无头模式 |
| `glue-cli` | 批量评测、样本生成、基准 | `inspect`/`gen`/`bench` | pipeline | CLI 冒烟 |

**关键 trait**（放在最低层，便于替换与测试）：

```rust
pub trait FrameSource {            // 相机/文件/视频统一抽象
    fn next_frame(&mut self) -> Result<Option<Frame>, CameraError>;
}
pub trait Detector {               // YOLO 反光 / RPN
    fn detect(&self, image: &GrayImage) -> Vec<Detection>;
}
pub trait Segmenter {              // 分割（可选）
    fn segment(&self, image: &GrayImage) -> BinaryMask;
}
pub trait Judge {                  // 规则判定（可切换 rule / fused）
    fn decide(&self, findings: &Findings) -> Decision;
}
```

---

## 6. 数据模型

### 6.1 配置（复用仓库现有文件）

`configs/default.yaml` 继续作为算法参数单一来源（与 Python 共用），Rust 侧只读：

```yaml
preprocess: {segment: {method}, morph: {ksize, order}}
measure:    {bfs_connectivity, min_area_px, width: {standard_mm, warn_delta_mm, alarm_delta_mm}}
calib:      {enabled, mm_per_px, pattern, square_mm}
quality:    {enabled, min_mean, max_mean, max_overexposed_ratio, min_contrast, min_sharpness}
decide:     {broken_min_rois, reflection_override}
locate:     {type, weights, input_size, nms: {type, thr}}
reflect:    {type, weights, input_size, score_thr}
```

**质量门控**（工程增强，论文未设此环节）：流水线在定位前计算灰度均值、p95−p05 对比度、饱和像素比例与 Laplacian 方差（清晰度），任一越界即返回 `NO_QUALITY`（"无法判定"），不进入定位与量测，避免把坏图误判为缺陷。阈值全部来自 `quality:` 配置段。

### 6.2 配方（Recipe，JSON）

```json
{
  "id": "rear-door-01",
  "name": "后门涂胶-左侧",
  "version": 3,
  "calib": {"mm_per_px": 0.0833, "mapping_table": "mapping/rear-door.json"},
  "positions": [
    {"id": "P1", "roi": [120, 80, 1180, 260], "standard_width_mm": 10.0,
     "warn_delta_mm": 0.5, "alarm_delta_mm": 1.0,
     "model": {"locate": "rpn-v1", "reflect": "yolo5fl-v1"}}
  ],
  "created_at": "2026-11-15T10:00:00+08:00",
  "operator": "admin"
}
```

### 6.3 单帧结果（与 Python `FrameResult` 字段一致）

```json
{
  "frame_id": 1024,
  "timestamp": "2026-11-15T10:00:01.234+08:00",
  "roi": {"boxes": [[...]], "count": 1},
  "measure": {"area_px": 12800, "length_px": 1200, "width_px": 10.6, "width_mm": 10.02},
  "decision": {"code": "OK", "message": "合格：10.02 mm"},
  "reflection": {"has_reflection": false, "boxes": [], "max_score": 0.02},
  "timings_ms": {"locate": 6.2, "measure": 8.1, "total": 18.4},
  "degraded": []
}
```

### 6.4 SQLite 表（首版）

| 表 | 字段 | 说明 |
| --- | --- | --- |
| `inspection` | id, ts, recipe_id, frame_id, verdict, width_mm, total_ms, image_path | 每帧一条 |
| `position_result` | inspection_id, position_id, verdict, width_mm, defects_json | 多位置 |
| `recipe` | id, name, version, json, updated_at | 配方版本 |
| `event` | id, ts, level, code, message | 告警/故障 |
| `model` | name, version, path, sha256, created_at | 模型登记 |

WAL 模式；批量事务（每 50 条或 200 ms 提交）；统计用索引 `(ts)`、`(recipe_id, verdict)`。

---

## 7. 检测流水线（含耗时预算）

| 步骤 | 模块 | 预算（ms） | 与 Python 对应 |
| --- | --- | --- | --- |
| ① 取帧/配对 | camera | 3 | 无（新增） |
| ② 图像检查（亮度/过曝/清晰度） | vision | 1 | 参考软件有，我们需要补 |
| ③ 定位（RPN 或阈值） | infer/vision | 6 | `locate`（假设 A5） |
| ④ 裁剪与预处理（Otsu、开闭） | vision | 2 | `preprocess` |
| ⑤ 量测（连通域、掩膜主方向跨度、宽度） | vision | 8 | `measure`（跨度代替骨架长度，避免端部缩短导致宽度高估） |
| ⑥ 反光复判（仅缺陷帧） | infer | 6 | `reflect` |
| ⑦ 判定与组装 | core | 1 | `decide` |
| ⑧ 显示/存储（异步） | app/store | 2 | — |
| **合计** | | **≈ 29** | Python 版 ~340 ms（待优化） |

二阶段目标：先做到"单帧 < 100 ms"（CPU 无 NPU），再冲 33 ms。

---

## 8. 训练架构（C++ / libtorch）

> 决策：训练侧从 Python 迁移到 **C++（LibTorch）**，与上位机/部署侧统一为 C++/Rust 技术栈。
> 说明：**libtorch 是 PyTorch 的 C++ 发行版**（autograd、优化器、序列化、TorchScript 齐全），
> 不是"从零手写训练框架"——自研 autograd 不做，风险与工作量都不可接受。

### 8.1 工程结构

```
train-cpp/
├─ CMakeLists.txt              # find_package(Torch REQUIRED)，C++17/20
├─ cmake/                      # Torch_DIR、OpenCV、可选 CUDA 的查找脚本
├─ src/
│  ├─ data/                    # manifest 读取、标注解析、letterbox、增广
│  ├─ model/                   # RPN、YOLO5FL + SPP（torch::nn）
│  ├─ loss/                    # DIoU/CIoU、RPN 采样损失、YOLO obj/cls/box
│  ├─ train/                   # 训练循环、验证、学习率调度、checkpoint
│  ├─ eval/                    # M1 平均 IoU、M2 召回、M3 框 IoU、耗时统计
│  └─ export/                  # TorchScript 导出；ONNX 转换（见 8.4）
├─ tools/
│  ├─ train_rpn.cpp            # CLI：训练 RPN
│  ├─ train_yolo.cpp           # CLI：训练 YOLO5FL+SPP（含消融开关）
│  ├─ eval.cpp                 # CLI：评测
│  ├─ gen_dataset.cpp          # 合成样本（与 Python synth 对齐）
│  └─ export_model.cpp         # CLI：导出
└─ tests/                      # GoogleTest / Catch2
```

### 8.2 训练与数据

| 项 | 方案 |
| --- | --- |
| 数据格式 | **复用现有 `data/manifest.csv` + `labels/*.json`**，与 Python 端同一份数据，便于对照 |
| 数据增强 | 水平/垂直翻转、旋转 ±5°、亮度/对比度、噪声、模糊（与论文场景一致）；增强参数进配置 |
| 采样 | RPN：每图 256 anchor、正负 1:1；YOLO：按正样本数归一化 obj 损失（沿用已验证做法） |
| 优化器 | AdamW + 余弦退火；梯度裁剪 10.0 |
| 精度 | FP32 默认；可选 AMP（CUDA） |
| 设备 | CPU 可跑通；CUDA 训练建议（见 8.5） |
| 断点 | 每轮存 `last.pt`/`best.pt`（TorchScript 序列化 + 训练状态） |
| 日志 | 每轮 loss/验证指标写 JSON + 终端输出，字段与 Python `history.json` 一致 |

### 8.3 与 Python 实现的关系

- **C++ 成为训练主实现**；Python（`algo/net/`）保留为**参考实现与对照基准**，不再新增功能；
- 对照测试：同一份固定 batch 输入，比较 C++/libtorch 与 Python/torch 的 forward 输出、loss、梯度范数（容差 1e-5）；
- 一旦对照通过，后续模型改动只在 C++ 侧进行，Python 侧冻结。

### 8.4 模型导出与部署衔接

```
C++ 训练 → TorchScript（.pt）→ ① Rust 侧用 libtorch/tch 直接加载
                              └→ ② 转 ONNX（一次性脚本）→ Rust ort / RKNN
```

- **ONNX 导出**：libtorch 不提供完整 ONNX 导出 API，采用"TorchScript → Python 一次性转换"（`torch.jit.load` + `torch.onnx.export`）；这一步只做格式转换，不含训练；
- 若要求全流程无 Python：Rust 侧用 `tch`（libtorch 绑定）直接加载 TorchScript，RK3588 侧再评估 TorchScript→RKNN 的官方链路（需验证）；
- 模型契约（`model.json`：输入尺寸、归一化、类别、anchor、阈值）与 Rust 侧共用。

### 8.5 环境与构建

| 项 | 要求 |
| --- | --- |
| 编译器 | MSVC 2022（Windows）/ GCC 11+（Linux） |
| 依赖 | LibTorch（CPU 或 CUDA 版，约 2–3 GB）、OpenCV C++（读图/增广）、CMake ≥ 3.24 |
| GPU | 有 NVIDIA GPU 建议 CUDA 版 libtorch；无 GPU 用 CPU 版（速度慢 10–30 倍） |
| CI | 至少保证 CPU 版可编译 + 单元测试通过 |

### 8.6 风险

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| libtorch 体积大、部署包臃肿 | 安装包变大 | 训练与部署分离：训练机装 libtorch，部署用 Rust/ONNX |
| C++ 调试效率低于 Python | 迭代慢 | 先用小数据/单 batch 对照，再扩量 |
| ONNX 导出缺 Python 步骤 | "全 C++"不彻底 | 明确"训练 C++、导出可借 Python 一次性脚本"；或走 TorchScript 直载 |
| GPU 环境缺失 | 训练慢 | 先 CPU 跑通流程，拿到 GPU 再扩量 |

## 9. 推理与模型

| 项 | 方案 |
| --- | --- |
| 模型导出 | Python（torch）→ ONNX；RPN 拆分：backbone+head 导出，proposal/NMS 在 Rust 实现 |
| 推理后端 | `ort`（ONNX Runtime Rust 绑定）；CPU 默认，可选 CUDA/TensorRT |
| 量化 | 先 FP32/FP16 对齐精度，再评估 INT8（反光小目标敏感） |
| 模型契约 | `model.json` 记录输入尺寸、归一化、类别、anchor、阈值；与参考软件思路一致 |
| 版本管理 | `models/<name>/<version>/model.onnx + model.json + sha256` |
| 热切换 | 配方引用模型版本；加载失败保留旧模型并告警 |

RPN 接入点：`glue-infer` 输出特征图（cls/reg），Rust 侧做解码 + Top-K + CIoU-NMS（与论文 3.1.4 一致），避免 ONNX 自定义算子。

---

## 10. 相机与触发

| 项 | 方案 |
| --- | --- |
| SDK | 海康 MVS C API；`bindgen` 生成 `mv_camera_control` 绑定；错误码映射为 Rust enum |
| 三路 | 每相机独立线程；统一时间戳（SDK 时间戳 + 本地单调时钟校正） |
| 触发 | 外部 24 V 硬触发（相机 IO）；软件记录触发序号，检测丢帧 |
| 同步 | 三路按触发序号配对，允许 ±2 ms 偏差；不齐则标记 `SYNC_MISS` |
| 回放 | `FileSource`（目录/视频）实现同一 `FrameSource` trait，便于无相机调试与 CI |
| 重连 | 看门狗检测心跳，断线后指数退避重连 |

> 风险：MVS Linux/aarch64 内核模块能否在 RK3588 BSP 上编译（见 RK3588 调研第 9 节）。首版先保证 Windows 可用。

---

## 11. 界面设计（egui）

交互框图：[host-ui-flow.png](host-ui-flow.png)（由 `scripts/make_host_ui_diagram.py` 生成）


### 10.1 页面

| 页面 | 内容 |
| --- | --- |
| 检测 | 三路画面缩略 + 主视图叠加、结果显示（判定/宽度/耗时）、开始/停止/单帧、报警条 |
| 配方 | 工位列表、位置编辑（画框选 ROI）、标准宽度/报警线、标定导入、模型选择 |
| 历史 | 时间/配方/判定筛选、缩略图、详情（叠加图、测量值、耗时）、导出 CSV |
| 统计 | 今日/班次检出率、误报率、宽度分布直方图、耗时 p50/p95 |
| 设置 | 相机（曝光/增益/触发）、算法参数、存储路径、日志、语言 |

### 10.2 交互状态机

```
Idle →(开始)→ Running →(停止)→ Idle
Running →(缺陷)→ Alarm（界面闪烁 + IO 输出，保持 500 ms）
Running →(相机断线)→ Degraded →(重连成功)→ Running
```

### 10.3 视觉风格（v0.2）

- 浅色工业风：内容区 `#F4F6F9`、卡片纯白、深色侧栏 `#1E293B`；
- 判定色：合格绿 `#16A34A`、预警琥珀 `#D97706`、报警红 `#DC2626`、未找到灰；
- 自动加载系统中文字体（DengXian / SimHei / Microsoft YaHei），避免中文显示为方块；
- 统计页用大数字卡片 + 判定分布条形图；检测页图像置于深色画布上，绿框标注 RoI。

### 10.4 渲染

- 主视图：图像纹理（`egui::TextureHandle`）+ 叠加层（框、骨架折线、宽度标注）；
- 大图缩放/平移；鼠标悬停显示坐标与像素值；
- 刷新率 ≤ 30 fps，UI 不阻塞检测（只取最新结果）。

---

## 12. 性能设计

| 措施 | 说明 |
| --- | --- |
| 纯 Rust 视觉 | 无 FFI/托管边界；Otsu/形态学/骨架只在 ROI 上做 |
| 零拷贝 | 相机缓冲直接映射为 `&[u8]`（MVS 回调缓冲），避免逐帧分配 |
| 内存池 | 帧缓冲与掩膜复用，减少 allocator 压力 |
| rayon | 三路并行 + 批量评测并行 |
| SIMD | Otsu 直方图、形态学可向量化（后续用 `std::simd` 或 `wide`） |
| 推理 | ONNX Runtime 图优化；NPU（RK3588）走 RKNN，Rust 侧 FFI |
| 基准 | `glue-cli bench` 固定输出 p50/p95/吞吐，进 CI 回归 |

**实测（M1，2026-11-15，1280×960 合成图，Windows x64 release）**：

| 指标 | Rust 版 | Python 版（对照） |
| --- | --- | --- |
| p50 | **8.1 ms** | 约 340 ms |
| p95 | **18.9 ms** | 约 500 ms |
| 吞吐 | **104 fps** | 约 3 fps |

说明：Python 版瓶颈在 BFS 连通域与骨架细化的 Python 循环；Rust 版把定位、形态学、量测都做成纯 Rust，且量测只在 ROI 掩膜上做。

---

## 13. 测试策略

| 层 | 方法 | 通过标准 |
| --- | --- | --- |
| 单元 | 合成图已知宽度/骨架/连通域 | 误差 ≤ 1 px |
| 训练对照 | C++/libtorch 与 Python/torch 同一 batch 的 forward/loss/梯度对比 | 误差 ≤ 1e-5 |
| 双实现对照 | Python `algo` 与 Rust 跑同一批图，比对 JSON | 宽度差 ≤ 1 px、判定一致率 100% |
| 流水线 | 五类合成用例（正常/窄/宽/断胶/反光） | 判定与预期一致 |
| 性能 | `bench` 固定机器基线 | 退化 > 20% 报警 |
| 故障注入 | 缺模型、缺标定、相机断线、磁盘满 | 按 4.4 降级，不崩溃 |
| 长稳 | 8 h 连续运行 | 无崩溃、内存增长 < 10% |

---

## 14. 部署

| 平台 | 方式 |
| --- | --- |
| Windows x64 | `cargo build --release` 单 exe；配置/模型/配方放 `%PROGRAMDATA%\GlueInspect\`；日志 `%LOCALAPPDATA%\GlueInspect\logs\` |
| aarch64/RK3588 | `scripts/build-aarch64.ps1` 交叉编译 `glue-agent`（aarch64-musl 静态单文件）+ systemd；详见 [host/DEPLOY-RK3588.md](host/DEPLOY-RK3588.md) |
| 安装 | 首版绿色包（zip）；后续可做 MSI |
| 升级 | 替换二进制 + 配置迁移脚本；模型按版本目录并存 |

---

## 15. 里程碑

| 阶段 | 内容 | 退出条件 |
| --- | --- | --- |
| M0 设计（本文件） | 架构、数据模型、接口、里程碑 | 评审通过 |
| M1 内核 | core/vision/pipeline/cli + 单测 | `cargo test` 全绿；CLI 出 JSON |
| M2 双实现对照 | Python/Rust 同图比对 | 宽度差 ≤ 1 px、判定一致 |
| M2.5 训练迁移 | C++/libtorch 训练工程（RPN + YOLO）、与 Python 单 batch 数值对照 | 误差 ≤ 1e-5；同数据同超参可收敛 |
| M3 推理 | `glue-infer`（tract 加载 ONNX；RPN/YOLO 解码 + NMS 已实现） | 数值对拍已过（<1e-6）；与 pipeline 集成、板端性能待做 |
| M4 相机 | `glue-camera`（MVS FFI、三路、触发） | 连续 8 h 无掉线 |
| M5 界面 | `glue-app`（5 页面 + 叠加渲染 + SQLite 历史/统计） | ✅ 文件回放闭环可演示（相机版待接） |
| M6 集成 | 存储/统计/报警、打包、ARM 交叉编译 | Windows 绿色包；RK3588 跑通 |

---

## 16. 风险与待决

| # | 风险/待决 | 影响 | 处理 |
| --- | --- | --- | --- |
| 1 | Rust GUI 生态弱于 WPF，界面开发慢 | 进度 | egui 先做功能；不满意再评估 Tauri/Slint |
| 2 | MVS FFI 与厂商库绑定、aarch64 内核模块 | 相机接入 | 先用文件/视频源跑通；FFI 单独小工程验证 |
| 3 | RPN 自定义算子不能直接进 ONNX | M3 延期 | 特征头导出 + Rust 后处理（与 RK3588 方案一致） |
| 4 | 双实现漂移 | 结果不一致 | M2 对照测试进 CI |
| 5 | 团队 Rust 熟练度 | 开发效率 | 先做内核与 CLI，UI 后置 |
| 6 | 是否必须与参考软件界面一致 | 验收 | **待确认**（见下） |

### 待确认问题（状态）

| # | 问题 | 状态 | 结论 |
| --- | --- | --- | --- |
| 1 | 技术栈确认 Rust？ | ✅ 已确认 | 用 Rust |
| 2 | 界面框架 egui / Tauri？ | ✅ 已确认 | 用 egui |
| 3 | 首版范围 | ✅ 已确认 | 文件回放 + 检测/配方/历史/统计（见 §2.3） |
| 4 | 是否与 GlueInspect 界面/操作习惯一致 | ⏳ 待定 | 默认参考其功能划分，不逐像素复刻 |
| 5 | 现场总线是否接真实 PLC/机器人 | ✅ 已确认 | 不接，用模拟信号 |
| 6 | 训练迁 C++ 的边界（ONNX 导出能否用 Python） | ⏳ 待定 | 建议允许一次性 Python 转换 |
| 7 | 训练机是否有 NVIDIA GPU | ⏳ 待定 | 决定 CUDA/CPU 版 libtorch |
| 8 | Python `algo/net/` 去留 | ⏳ 待定 | 建议保留为对照基准 |

## 17. 附录

### 16.1 目录结构规划

```
host/
├─ Cargo.toml                 # workspace
├─ crates/
│  ├─ core/                   # 类型、配置、配方、判定
│  ├─ vision/                 # 图像算法（纯 Rust）
│  ├─ infer/                  # ONNX Runtime
│  ├─ camera/                 # MVS FFI + 回放源
│  ├─ pipeline/               # 编排、并行、计时
│  ├─ store/                  # SQLite
│  ├─ app/                    # egui 界面
│  └─ cli/                    # bench / gen / inspect
├─ models/                    # 模型（gitignore，按版本）
└─ recipes/                   # 配方样例

train-cpp/                    # C++ 训练工程（libtorch）
├─ CMakeLists.txt
├─ src/{data,model,loss,train,eval,export}
├─ tools/{train_rpn,train_yolo,eval,export_model,gen_dataset}.cpp
└─ tests/
```

### 16.2 与 GlueInspect 的模块对照

| GlueInspect | 本设计 | 备注 |
| --- | --- | --- |
| `GlueInspect.Core` | `glue-core` | 领域模型与判定 |
| `GlueInspect.Vision` | `glue-vision` | OpenCV → 纯 Rust |
| `GlueInspect.Inference` | `glue-infer` | ONNX Runtime |
| `GlueInspect.Pipeline` | `glue-pipeline` | 编排 |
| `GlueInspect.Storage` | `glue-store` | SQLite |
| `GlueInspect.App` | `glue-app` | WPF → egui |
| `GlueInspect.Bench` | `glue-cli` | 批量评测 |
| `tools/GlueInspect.SampleGen` | `glue-cli gen` | 合成样本 |

### 16.3 术语

| 术语 | 含义 |
| --- | --- |
| RoI | 胶条所在的感兴趣区域 |
| 配方 | 某工位的检测配置（位置、宽度标准、阈值、模型） |
| 双实现 | Python（训练/评测）与 Rust（部署）两套实现 |
| 降级 | 模型/标定缺失时仍能运行，但结果标注原因 |
