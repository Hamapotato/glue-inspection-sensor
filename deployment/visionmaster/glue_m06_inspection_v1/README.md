# M06 胶条检测首轮部署包

> 方案名：`glue_m06_inspection_v1`
> 软件：VisionMaster 4.4
> 模型：`glue_bg_seg_syn500_v5_vm440_cpu_v001`
> 数据：`glue-m06-matched-v1-100`
> 状态：首轮方案已运行；仅有人工观察反馈，尚无冻结指标

## 1. 当前事实

用户已完成首轮 VisionMaster 方案运行，并人工观察到方案可以识别：

- 断胶；
- 胶条过宽；
- 胶条过窄；
- 气泡。

这四项目前都只是首轮运行观察，不是已经统计的召回率、准确率、误报率或
节拍指标。M06 配套数据是合成数据，`acceptance_eligible=false`，不能作为
真实产线验收结果。

## 2. 正式命名

| 原名称 | 本项目正式名称 |
|---|---|
| `final.sol` | `solution/glue_m06_inspection_v1.sol` |
| `final.bin` / `Flaw_1_100_202611211543594_V1_Board.bin` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001_Board.bin` |
| `Flaw_1_100_202611211543594_V1_Nvidia.bin` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001_Nvidia.bin` |
| `Flaw_1_100_202611211543594_V1.mvbonnx` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001.mvbonnx` |
| `glue-m06-matched-v1-100/` | `validation/glue-m06-matched-v1-100/` |

`final.bin` 与后缀为 `_Board.bin` 的文件 SHA-256 完全一致。当前方案内部加载
的是 `_Board.bin`，因此正式模型标识按 VM 440 CPU 命名；同时保留
`_Nvidia.bin` 和 `.mvbonnx`，避免后续切换执行设备时缺失配套文件。

## 3. 目录

```text
glue_m06_inspection_v1/
├── README.md
├── bundle.json
├── solution/
│   └── glue_m06_inspection_v1.sol
├── models/
│   ├── glue_bg_seg_syn500_v5_vm440_cpu_v001_Board.bin
│   ├── glue_bg_seg_syn500_v5_vm440_cpu_v001_Nvidia.bin
│   └── glue_bg_seg_syn500_v5_vm440_cpu_v001.mvbonnx
└── validation/
    └── glue-m06-matched-v1-100/
```

验证集包含 100 张图像、100 张掩膜、100 份标签和 5 个说明/清单文件，
共 305 个文件。场景构成为 1 张标准模板、19 张正常图、20 张断胶图、
20 张过窄图、20 张过宽图和 20 张弯曲图。

## 4. 打开方案

1. 用 VisionMaster 4.4 打开：

   ```text
   D:\WORK\glue-inspection-sensor\deployment\visionmaster\glue_m06_inspection_v1\solution\glue_m06_inspection_v1.sol
   ```

2. 检查 DL 图像分割模块加载的模型是否为：

   ```text
   D:\WORK\glue-inspection-sensor\deployment\visionmaster\glue_m06_inspection_v1\models\glue_bg_seg_syn500_v5_vm440_cpu_v001_Board.bin
   ```

3. 检查模型类别仍为 `background/glue`，执行模块为 CPU。
4. 将图像源切换到 `validation/glue-m06-matched-v1-100/images`，先运行标准模板。
5. 再按正常、断胶、过窄、过宽、弯曲顺序检查。

方案内部保存的是本机绝对模型路径。移动整个部署目录后，必须在 VM 中重新
选择模型；只移动目录而不更新路径会导致模型加载失败。

## 5. 尚未完成

- 尚未记录 M06 100 张图的逐类命中数、漏报数和误报数；
- 尚未在独立气泡数据上固化气泡判定口径和指标；
- 尚未完成真实相机标定和毫米尺寸验证；
- 尚未记录 VM 推理与端到端 p50/p95 耗时；
- 尚不能声明真实产线检出率、误报率或节拍合格。

## 6. 结果输出与前端

VM 端通过“输出图像 + TCP 结果”两条通道接入独立查看器：

```text
原图模块：original_YYYYMMddHHmmssfff
卡尺图模块：caliper_YYYYMMddHHmmssfff
TCP 客户端：127.0.0.1:5000，发送 YYYYMMddHHmmssfff|OK/NG
```

前端按时间戳绑定两张图片和 OK/NG，并保存 `record.json` 与历史副本。前端的
构建、历史目录、清理范围和冒烟测试见
[VM 结果查看器](../../vm-viewer/README.md)。

该链路已验证通信和数据绑定，不等于已经完成 M06 逐类统计或真实产线验收。

来源、SHA-256 和验证边界见 [bundle.json](bundle.json)。
