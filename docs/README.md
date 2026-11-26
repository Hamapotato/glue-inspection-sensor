# 项目文档索引

本目录保留当前“胶条分割模型 + 旋转扫描矩形”技术路线及其部署、结果查看
链路的项目文档。

## 算法

| 文档 | 内容 |
|---|---|
| [算法与标定报告](algo/algorithm-and-calibration-report.md) | 当前分割主线、扫描量测和标定边界 |
| [单相机标定计划](algo/calibration-plan.md) | 标定靶、3D 打印支架、采图、标定和尺寸验证流程 |
| [算法主线方案](algo/plan.md) | 分割模型、旋转扫描矩形、边缘中点曲线和质量判定 |
| [阶段进度](algo/progress.md) | 当前完成度和阶段性结论 |
| [P0 指标与数据规范](algo/p0-spec.md) | 分割、扫描量测、判定指标与假设登记 |
| [接口契约](algo/interfaces.md) | 算法接口、配置键、输出结构和降级策略 |
| [任务清单](algo/tasks.md) | 主线清理、模型、量测、标定和部署任务 |
| [VisionTrain 数据准备](algo/visiontrain-data-preparation.md) | 胶条/背景二分类数据生成、缺陷场景元数据、输出格式和非验收边界 |
| [VisionTrain 人工校验](algo/visiontrain-manual-review.md) | 新版二分类导入、缺陷场景语义检查和失败记录 |
| [VisionTrain 训练与部署 SOP](algo/visiontrain-training-sop.md) | 胶条/背景二分类训练、评估和接入主线 |
| [VisionMaster 部署流程](algo/visionmaster-deployment-sop.md) | VM 4.4 加载分割模型、输出掩膜、单图验证和接入主线 |
| [VisionMaster 纯 VM 主线](algo/visionmaster-pure-vm-mainline.md) | VM 内完成分割、掩膜、边缘对轨迹、缺陷和 OK/NG 判断 |
| [VisionMaster 拖拽端到端 SOP](algo/visionmaster-dragdrop-end-to-end.md) | 从空方案开始，按顺序配置模块、订阅输入、连接流程并输出 OK/NG |
| [VisionMaster 内存图像帧传输](algo/visionmaster-memory-frame.md) | C# 脚本将双图、判定和缺陷信息通过 TCP 二进制帧发送，避免逐帧落盘 |
| [M06 同轨迹验证数据集](algo/m06-matched-validation-dataset.md) | 固定标准胶路的正常、断胶、宽窄和弯曲配对数据 |
| [旋转扫描矩形量测](algo/scanline-measurement.md) | 沿胶条方向生成扫描窗、左右边缘点、中心曲线、断胶/宽度/弯曲判定 |

## 部署

| 文档 | 内容 |
|---|---|
| [M06 首轮部署包](../deployment/visionmaster/glue_m06_inspection_v1/README.md) | 正式命名且含内存帧接线的 VM 方案、VM 440 CPU 模型、M06 配套验证集和当前事实边界 |
| [VM 结果查看器](../deployment/vm-viewer/README.md) | 接收 VM 结果、按时间绑定双图、历史保存/回看、清理功能、构建和冒烟测试 |

## 项目与流程

| 文档 | 内容 |
|---|---|
| [Git 操作规范](project/git-workflow.md) | 分支模型、提交规范、同步/变基/推送与清理流程 |
| [论文算法 vs 产品算法对照](project/thesis-vs-product-algorithm.md) | 两份材料的算法拆解、逐项对比、选型建议与证据引用 |

## 归档

| 文档 | 内容 |
|---|---|
| [旧论文算法归档](../archive/legacy-paper-algorithm/README.md) | RPN、YOLO 反光、面积/骨架平均宽度、旧合成演示和历史文档 |
| [非主线文档归档](../archive/non-mainline-docs/README.md) | 硬件、采购、BOM、产品路线和汇报材料 |

## 数据

| 文档 | 内容 |
|---|---|
| [数据目录规范](../data/README.md) | 原始图像、标注、manifest、宽度真值和数据集划分规范 |

## 维护规则

1. 当前技术路线文档放在 `docs/algo/`；
2. 数据规范放在 `data/README.md`；
3. 非当前技术路线材料移入 `archive/`，不留在主文档入口；
4. 移动文档后必须检查相对链接；
5. 合成数据结果必须标明“非真实指标”，不得与正式验收结果混写。
