# glue-inspection-sensor

基于机器视觉的车身涂胶检测传感器开发仓库。

本项目以学位论文《基于机器视觉的车身涂胶在线检测系统研究》为技术起点，从三相机圆形视觉传感器复现逐步演进到基于 RK3588 的边缘计算产品。

GitHub：[Hamapotato/glue-inspection-sensor](https://github.com/Hamapotato/glue-inspection-sensor)

## 项目状态

- [x] 阅读并整理相关论文与产品资料
- [x] 输出功能复现 BOM
- [x] 给出接口、供电、USB 和 LED 驱动初版方案
- [x] 制定从论文复现到 RK3588 边缘计算产品的开发路线
- [x] 完成传统视觉量测链路的可运行原型和合成数据验证
- [x] 完成 VisionTrain 合成涂胶分割训练集生成器（仅流程验证/预训练）
- [x] 完成首台 `MV-CB016-10UM-S` 相机与 MVS 环境准备
- [ ] 完成机械结构设计
- [ ] 完成主控板和 LED 灯板原理图、PCB
- [ ] 完成单相机标定、固定工位和真实尺寸验证
- [ ] 完成相机、镜头和照明联调
- [ ] 完成深度学习模型验证与训练闭环
- [ ] 完成检测算法产品化与上位机软件

> 论文未公开完整原理图、PCB、网表、机械尺寸和灯板接线，因此本项目是可工作的功能复现，不是原设备的严格 1:1 复刻。

## 方案概要

| 项目 | 初版方案 |
|---|---|
| 相机 | 3 × HIKROBOT `MV-CB016-10UM-S` 板级相机；首台已选用 |
| 镜头 | 3 × 6 mm M12 镜头；首台按 1/2.9"、500 万像素级选型 |
| 照明 | 26 × 约 655 nm 红色 LED，按 8+9+9 分三串驱动 |
| 触发 | 24 V IO 经光耦隔离，硬件触发相机和 LED 频闪 |
| 数据接口 | 3 台相机经 4 口 USB 3.0 Hub 连接工控机 |
| 计算平台 | 外置 Windows 工业计算机 |

## 仓库结构

```text
.
├── README.md
├── AGENTS.md                   # AI agent 协作规范（红线、DoD、提问边界）
├── requirements.txt            # 核心依赖（torch 为可选项）
├── .gitattributes              # 统一文本换行和二进制文件处理
├── algo/                       # 算法实现（标定、预处理、量测、模型、流水线）
├── configs/                    # 配置基线（阈值集中管理）
├── scripts/                    # 可执行入口
├── tests/                      # 单元测试
├── docs/
│   ├── README.md               # 文档总索引
│   ├── algo/                   # 算法方案、规范、任务、标定和阶段报告
│   ├── hardware/               # BOM、相机选型和采购记录
│   └── project/                # 产品路线、导师汇报和项目管理材料
├── references/
│   ├── README.md               # 参考资料说明
│   └── source/                 # 原始 PDF，仅本地保留，不提交到 Git
└── .cache/                     # PDF 预览等临时文件，不提交到 Git
```

## 快速入口

- [文档总索引](docs/README.md)
- [算法与标定阶段汇报](docs/algo/algorithm-and-calibration-report.md)
- [单相机标定与尺寸验证计划](docs/algo/calibration-plan.md)
- [算法复现阶段性汇报](docs/algo/progress.md)
- [算法复现方案](docs/algo/plan.md)｜[P0 指标与数据规范](docs/algo/p0-spec.md)｜[接口契约](docs/algo/interfaces.md)｜[任务清单](docs/algo/tasks.md)
- [VisionTrain 合成数据准备](docs/algo/visiontrain-data-preparation.md)
- [复现 BOM 与硬件设计说明](docs/hardware/reproduction-bom.md)
- [首台相机采购清单](docs/hardware/first-procurement-list.md)
- [海康 USB 相机与镜头选型对比](docs/hardware/hikrobot-usb-camera-comparison.md)
- [RK3588 产品开发路线](docs/project/product-development-roadmap.md)
- [导师汇报摘要](docs/project/mentor-report-summary.md)
- [AI agent 协作规范](AGENTS.md)
- [参考资料与版权说明](references/README.md)

## 安全提示

LED 灯板采用大电流恒流驱动，26 颗灯珠满功率时热负荷较高。调试时应从较低电流开始，使用铝基板、导热界面和金属外壳，并测量实际温升。24 V 输入、机器人 IO 和 USB 地之间应按设计进行隔离和单点接地。

## 原始资料

原始 PDF 可能受第三方版权或授权范围限制，仓库默认不提交这些文件。需要查阅时请将合法获得的 PDF 放在 `references/source/`，Git 会自动忽略该目录。
