# 部署资产

本目录集中保存可运行的部署包及其追溯清单。二进制方案、模型权重、验证图像
与标注属于本地资产，默认不提交到 Git；对应的 `README.md` 和
`bundle.json` 可以入库。

## 当前部署包

| 部署包 | 软件 | 用途 |
|---|---|---|
| [glue_m06_inspection_v1](visionmaster/glue_m06_inspection_v1/README.md) | VisionMaster 4.4 | 首轮胶条缺陷检测方案、VM 440 CPU 分割模型和 M06 配套验证集 |

## 目录规则

```text
deployment/
└── visionmaster/
    └── <方案名>/
        ├── README.md       # 使用说明和当前事实边界
        ├── bundle.json     # 来源、正式命名、哈希和验证状态
        ├── solution/       # 本地 .sol 方案
        ├── models/         # 本地模型及配套文件
        └── validation/     # 本地验证图像、掩膜和标签
```

方案或模型被移动、重新导出后，必须同步更新 `bundle.json` 中的路径和
SHA-256，不能只修改文件名。
