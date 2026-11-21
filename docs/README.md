# 项目文档索引

本目录按文档职责分为算法、硬件和项目三类。新增文档时应放入对应目录，并同步更新本索引。

## 算法

| 文档 | 内容 |
|---|---|
| [算法与标定阶段汇报](algo/algorithm-and-calibration-report.md) | 当前算法成果、验证结果、模型边界和近期标定安排 |
| [单相机标定计划](algo/calibration-plan.md) | 标定靶、3D 打印支架、采图、标定和尺寸验证流程 |
| [算法复现方案](algo/plan.md) | 论文算法拆解、分层架构、数据与评估方案 |
| [阶段进度](algo/progress.md) | 当前完成度、流程图和阶段性结论 |
| [P0 指标与数据规范](algo/p0-spec.md) | M1–M10 指标口径、数据规范与假设登记 |
| [接口契约](algo/interfaces.md) | 算法接口、配置键、输出结构和降级策略 |
| [任务清单](algo/tasks.md) | P0–P5 任务、依赖关系与阻塞项 |
| [P1 宽度量测报告](algo/reports/p1-measure.md) | 当前合成自检报告；真实指标待现场验证 |

## 硬件

| 文档 | 内容 |
|---|---|
| [复现 BOM](hardware/reproduction-bom.md) | 相机、镜头、LED、电源、接口和 PCB 初步方案 |
| [相机与镜头选型](hardware/hikrobot-usb-camera-comparison.md) | HIKROBOT USB 3.0 相机横向对比和最终选型 |
| [相机与镜头选型 PDF](hardware/hikrobot-usb-camera-comparison.pdf) | 便于汇报和分享的 PDF 版本 |
| [第一批采购清单](hardware/first-procurement-list.md) | 首台相机、镜头、线缆、安装件和到货验收要求 |

## 项目

| 文档 | 内容 |
|---|---|
| [产品开发路线](project/product-development-roadmap.md) | 从论文装置到 RK3588 工程样机和产品化的开发路线 |
| [导师汇报摘要](project/mentor-report-summary.md) | 项目总体方案、阶段安排和汇报重点 |
| [相机接入与 ROI 推理验证报告](project/camera-infer-verification-2026-11-22.md) | 板端相机链路、RPN 推理接入与算法耗时实测（2026-11-22） |
| [Git 操作规范](project/git-workflow.md) | 分支模型、提交规范、同步/变基/推送与清理流程 |

## 数据

| 文档 | 内容 |
|---|---|
| [数据目录规范](../data/README.md) | 原始图像、标注、manifest、宽度真值和数据集划分规范 |

## 维护规则

1. 算法方法、指标和实验文档放在 `docs/algo/`；
2. 器件、采购、相机和光学选型放在 `docs/hardware/`；
3. 产品路线、导师汇报和跨专业项目材料放在 `docs/project/`；
4. 新增顶层类别时同步修改本索引和根目录 `README.md`；
5. 移动文档后必须检查相对链接；
6. 合成数据结果必须标明“非真实指标”，不得与正式验收结果混写。
