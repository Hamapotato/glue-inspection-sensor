# 任务清单（P0 → P5）

> 用法：每完成一项勾选，并在 PR/commit 信息里引用编号（如 `A-03`）。
> 估时按"单人、有现场数据可用"估算；数据未到位时 B 组顺延，C 组不受影响。

---

## P0 口径与基线（3 天）

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [x] A-01 | 指标口径定义 | `docs/algo/p0-spec.md` §1 | 评审通过 |
| [x] A-02 | 假设登记表 | 同上 §5 | A1–A15 有初值 |
| [x] A-03 | 接口契约 | `docs/algo/interfaces.md` | 三个 Protocol 签名冻结 |
| [x] A-04 | 配置基线 | `configs/default.yaml` | 所有阈值可配置，无硬编码 |
| [x] A-05 | 数据目录与 manifest 模板 | `data/README.md` + 空 `manifest.csv` | 字段确认 |
| [ ] A-06 | 试标 20 张图 | `data/labels/*` | 双人一致，标注规范无歧义 |

## P1 传统量测链路（2 周）★ 优先

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [x] C-01 | 小波去噪 + 中值对比 | `algo/preprocess.py`（已初版） | 视觉对比图 + 参数记录 |
| [x] C-02 | 三种阈值分割对比 | 同上 | 复现图 4-9 的对比实验 |
| [x] C-03 | 开闭运算修孔洞 | 同上 | 图 4-10 效果 |
| [x] C-04 | 相机标定 + 重投影误差 | `algo/calib.py` | 误差表（对齐表 4-2 格式） |
| [x] C-05 | BFS 面积 + 外轮廓法对比 | `algo/measure.py` | 复现表 4-3/4-4 结构 |
| [x] C-06 | Zhang-Suen 骨架 + 长度 | 同上 | 骨架图 + 长度 vs 真实长度 |
| [x] C-07 | 宽度 = 面积/长度 → mm | 同上 | **宽度误差表**（MAE/std/max） |
| [x] C-08 | 合成样本生成器 | `algo/synth.py` | 能生成正常/断胶/过窄/反光四类 |
| [x] C-09 | 单元测试（几何/量测/标定） | `tests/` | `py -3 -m unittest` 全绿 |
| [x] C-10 | P1 报告 | `docs/algo/reports/p1-measure.md` | 与 0.23/0.25/0.39 mm 对照 |

## P2 判定与集成（1 周）

> 注：C-01~C-10 与 D-01~D-04 已完成（C 组含报告生成脚本与数据规范）；
> C-10 的报告骨架已就绪（`docs/algo/reports/p1-measure.md`），当前为合成自检，
> **真实指标待现场图像与卡尺真值**；D-05（p50/p95 统计）待补。
>
> 当前第一台相机和 MVS 已就绪，单相机标定及尺寸验证按
> [calibration-plan.md](calibration-plan.md) 执行。C-04 的代码已完成，
> 但真实相机的标定图和重投影误差验收仍属于当前阶段工作。

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [x] D-01 | 判定规则（断胶/宽窄/双报警线） | `algo/decide.py` | 单测覆盖所有分支 |
| [x] D-02 | 反光复判分支（命中则提示调参） | 同上 | 不把反光误判成缺陷 |
| [x] D-03 | 单帧流水线 | `algo/pipeline.py` | 合成图端到端跑通，输出 JSON |
| [x] D-04 | 叠加图输出（调试用） | `scripts/demo_synthetic.py` | 保存 ROI/分割/骨架叠加图 |
| [ ] D-05 | 计时统计（p50/p95） | `algo/metrics.py` | 报告出延迟表 |

## P3 RoI 定位（2 周）

> 注：B-01~B-04、B-07 的验证均为**合成数据冒烟**（见 [P3 报告](reports/p3-roi.md)），不是论文指标。

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [x] B-01 | 数据集与 DataLoader（按工件划分） | `algo/data.py` | 划分不泄漏 |
| [x] B-02 | K-means 先验框 | `algo/geometry.py`（已有） | 9 个 anchor，可视化 |
| [x] B-03 | RPN 网络 + DIoU 损失 | `feat/b1-b2-models`：`algo/net/rpn.py` | 合成数据可收敛（P3 报告） |
| [x] B-04 | CIoU-NMS 推理 | 同上 | 与标准 NMS 对比表（合成，P3 §5） |
| [ ] B-05 | 消融：IoU/GIoU/DIoU/CIoU（表 3-1） | `eval` 脚本 | 4 组 IoU + 耗时 |
| [ ] B-06 | 消融：NMS vs CIoU-NMS × 阈值（表 3-2） | 同上 | 5 阈值 × 2 方法 |
| [x] B-07 | 平均 IoU 评测与报告 | `docs/algo/reports/p3-roi.md` | 对齐 99% 口径 |

## P4 反光检测（2 周）

> 注：结构已完成并可训练，但合成反光小目标尚未收敛（[P4 报告](reports/p4-reflect.md)），B-12 待模型收敛后复测。

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [x] B-08 | YOLOv3 基线 | `feat/b1-b2-models`：`algo/net/yolo.py` | 能训练/推理 |
| [x] B-09 | 5 层特征层（YOLOv3-5FL） | 同上 | 5 个输出尺度 |
| [x] B-10 | SPP 模块（4/2/1） | 同上 | 21 bin |
| [ ] B-11 | 四组消融（表 3-4） | `eval` 脚本 | 逐图 IoU + 耗时 |
| [ ] B-12 | 反光召回评测 | `docs/algo/reports/p4-reflect.md`（骨架） | 模型收敛后复测 97% 口径 |

## P5 对标与收尾（1 周）

| ID | 任务 | 产出 | 验收 |
|---|---|---|---|
| [ ] E-01 | 端到端提速（30 fps 目标） | 优化记录 | p95 ≤ 33 ms 或给出结论 |
| [ ] E-02 | 总对照表 | `docs/algo/reports/summary.md` | 论文值/复现值/偏差原因 三列 |
| [ ] E-03 | 版本与复现说明 | `docs/algo/reproduce.md` | 他人可照做 |
| [ ] E-04 | 结论与后续建议 | 同上 | 明确哪些能复现、哪些不能 |
| [x] E-05 | ONNX 导出与数值对拍（B1/B2） | `scripts/export_onnx.py`、`docs/algo/reports/p5-onnx.md` | 两模型可导出，onnxruntime 与 PyTorch 最大误差 ≤1e-4 |
| [x] E-06 | Rust 端 ONNX 推理与对拍（tract） | `host/crates/infer`、`scripts/check_infer_parity.py` | 与 onnxruntime 对拍 ≤1e-5（实测 <1e-6） |
| [x] E-07 | RPN/YOLO 接入 pipeline（配置开关 + 回退 + 反光复判） | `glue-core::ObjectDetector`、`glue-pipeline::Detectors` | 随机权重冒烟通过；判定级对拍见 E-08 |
| [ ] E-08 | 端到端判定对拍（训练权重后） | 对拍脚本 | 同一批图判定码 100% 一致 |

---

## 依赖关系

```
A-04 ──> C-01..C-09 ──> C-10
              │
              └──> D-01..D-05
A-05/A-06 ──> B-01 ──> B-03 ──> B-05/B-06 ──> B-07
                        └──> B-09/B-10 ──> B-11 ──> B-12
C-07 ─────────────────────────────────────────────> E-02
```

## 当前阻塞项

1. **数据**：B-01 之前需要至少 200 张真实图 + RoI 标注；C-04 需要棋盘格标定图。
2. **相机/光学**：标定与工作距离确认依赖硬件到位（见复现 BOM 的首轮采购顺序）。
3. **决策**：B1/B2 自写 vs 改造现成框架，尚未拍板。
4. **真实数据**：B-05（表 3-1 四组损失）、B-06（表 3-2 NMS 阈值）、B-11（表 3-4 四组消融）
   均需真实数据才有意义；当前只在合成数据上做了单组验证，结果见 reports/p3-roi.md、p4-reflect.md。
