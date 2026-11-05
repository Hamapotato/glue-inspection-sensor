# glue-inspection-sensor

基于机器视觉的车身涂胶检测传感器开发仓库。

本项目以学位论文《基于机器视觉的车身涂胶在线检测系统研究》为技术起点，从三相机圆形视觉传感器复现逐步演进到基于 RK3588 的边缘计算产品。

GitHub：[Hamapotato/glue-inspection-sensor](https://github.com/Hamapotato/glue-inspection-sensor)

## 项目状态

- [x] 阅读并整理相关论文与产品资料
- [x] 输出功能复现 BOM
- [x] 给出接口、供电、USB 和 LED 驱动初版方案
- [x] 制定从论文复现到 RK3588 边缘计算产品的开发路线
- [ ] 完成机械结构设计
- [ ] 完成主控板和 LED 灯板原理图、PCB
- [ ] 完成相机、镜头和照明联调
- [ ] 完成检测算法与上位机软件

> 论文未公开完整原理图、PCB、网表、机械尺寸和灯板接线，因此本项目是可工作的功能复现，不是原设备的严格 1:1 复刻。

## 方案概要

| 项目 | 初版方案 |
|---|---|
| 相机 | 3 × HIKROBOT `MV-CB013-20UM-S` 板级相机 |
| 镜头 | 3 × Evetar `M13B0618IR`，6 mm、F1.8、S-mount |
| 照明 | 26 × 约 655 nm 红色 LED，按 8+9+9 分三串驱动 |
| 触发 | 24 V IO 经光耦隔离，硬件触发相机和 LED 频闪 |
| 数据接口 | 3 台相机经 4 口 USB 3.0 Hub 连接工控机 |
| 计算平台 | 外置 Windows 工业计算机 |

## 仓库结构

```text
.
├── README.md
├── .gitattributes              # 统一文本换行和二进制文件处理
├── docs/
│   ├── reproduction-bom.md     # 器件清单、接线、PCB 要求和采购顺序
│   └── product-development-roadmap.md  # RK3588 产品开发路线
├── references/
│   ├── README.md               # 参考资料说明
│   └── source/                 # 原始 PDF，仅本地保留，不提交到 Git
└── .cache/                     # PDF 预览等临时文件，不提交到 Git
```

## 快速入口

- [复现 BOM 与硬件设计说明](docs/reproduction-bom.md)
- [RK3588 产品开发路线](docs/product-development-roadmap.md)
- [参考资料与版权说明](references/README.md)

## 安全提示

LED 灯板采用大电流恒流驱动，26 颗灯珠满功率时热负荷较高。调试时应从较低电流开始，使用铝基板、导热界面和金属外壳，并测量实际温升。24 V 输入、机器人 IO 和 USB 地之间应按设计进行隔离和单点接地。

## 原始资料

原始 PDF 可能受第三方版权或授权范围限制，仓库默认不提交这些文件。需要查阅时请将合法获得的 PDF 放在 `references/source/`，Git 会自动忽略该目录。
