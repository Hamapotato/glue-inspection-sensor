# M06 胶条检测首轮部署包

> 方案名：`glue_m06_inspection_v1`
> 软件：VisionMaster 4.4
> 模型：`glue_bg_seg_syn500_v5_vm440_cpu_v001`
> 数据：`glue-m06-matched-v1-100`
> 状态：内存帧发送已接入正式 `.sol`；真实连续运行和指标验收仍未完成

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
| `beifen-memory.sol` | `solution/glue_m06_inspection_v1.sol` |
| 内部方案名 `beifen-memory` | 内部正式名 `glue_m06_inspection_v1` |
| `final.bin` / `Flaw_1_100_202611211543594_V1_Board.bin` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001_Board.bin` |
| `Flaw_1_100_202611211543594_V1_Nvidia.bin` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001_Nvidia.bin` |
| `Flaw_1_100_202611211543594_V1.mvbonnx` | `models/glue_bg_seg_syn500_v5_vm440_cpu_v001.mvbonnx` |
| `glue-m06-matched-v1-100/` | `validation/glue-m06-matched-v1-100/` |

`final.bin` 与后缀为 `_Board.bin` 的文件 SHA-256 完全一致。当前方案内部加载
的是 `_Board.bin`，因此正式模型标识按 VM 440 CPU 命名；同时保留
`_Nvidia.bin` 和 `.mvbonnx`，避免后续切换执行设备时缺失配套文件。

本轮正式方案的来源是：

```text
D:\VMProjects\beifen-memory.sol
```

归档时只改写方案内部 `VmServer.xml` 的 `SolutionName`，其余模块、脚本、
图形收集、通信配置和附件保持来源文件内容。

## 3. 目录

```text
glue_m06_inspection_v1/
├── README.md
├── bundle.json
├── config/
│   └── vm_frame_sender.ini
├── solution/
│   └── glue_m06_inspection_v1.sol
├── scripts/
│   ├── test-frame-sender.ps1
│   ├── VmFrameSender.cs
│   └── VmFrameSenderSmokeTest.cs
├── models/
│   ├── glue_bg_seg_syn500_v5_vm440_cpu_v001_Board.bin
│   ├── glue_bg_seg_syn500_v5_vm440_cpu_v001_Nvidia.bin
│   └── glue_bg_seg_syn500_v5_vm440_cpu_v001.mvbonnx
└── validation/
    └── glue-m06-matched-v1-100/
```

正式 `.sol` 明确纳入 Git，由 `.gitignore` 对该文件设置白名单；模型文件和
验证图片仍默认作为本地部署资产，不进入 Git。当前方案约 80.9 MB，包含 124 个
压缩条目，其中包含 100 张 M06 模块附件图。

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
6. 启动 VM 结果查看器：

   ```text
   D:\WORK\glue-inspection-sensor\deployment\vm-viewer\start-viewer.cmd
   ```

7. 先运行正常图确认前端收到 OK，再运行 NG 图确认双图、判定和中文缺陷类型
   一致。

方案内部保存的是本机绝对模型路径。移动整个部署目录后，必须在 VM 中重新
选择模型；只移动目录而不更新路径会导致模型加载失败。

## 5. 尚未完成

- 尚未记录 M06 100 张图的逐类命中数、漏报数和误报数；
- 尚未在独立气泡数据上固化气泡判定口径和指标；
- 尚未完成真实相机标定和毫米尺寸验证；
- 尚未记录 VM 推理与端到端 p50/p95 耗时；
- 尚不能声明真实产线检出率、误报率或节拍合格。

## 6. 结果输出与前端

当前正式 `.sol` 已包含内存帧发送接线：

```text
图像源原图 ------------------------------┐
                                        ├-> C# 脚本
边缘对结果 -> 图形收集 -> 输出图像 ------┤     |
最终判定 + 缺陷类型 + 缺陷个数 ----------┘     v
                                      GIFR v1 TCP 二进制帧
                                             |
                                             v
                              查看器 127.0.0.1:5001
```

方案内已包含 `脚本2`、`图形收集1`、`输出图像1` 和 TCP 通信设备。脚本模块
订阅原图、卡尺绘制图、判定、缺陷类型和缺陷个数；前端直接接收二进制帧，
两张图在内存中显示，只有 NG 写盘。

完整协议、变量名、配置和排障见
[VisionMaster 内存图像帧传输](../../../docs/algo/visionmaster-memory-frame.md)。

VM 发送端资产：

```text
scripts/VmFrameSender.cs
scripts/VmFrameSenderSmokeTest.cs
scripts/test-frame-sender.ps1
config/vm_frame_sender.ini
```

目标链路：

```text
原图 + 卡尺绘制图 + OK/NG + 缺陷信息
  -> C# 脚本组帧
  -> TCP 客户端 127.0.0.1:5001
  -> WPF 前端内存接收
  -> 仅 NG 写盘
```

查看器默认入口已切换到内存帧协议，旧目录监听和文本解析代码暂时保留作迁移
对照。真实 VM 连续发送、断线重连和实际节拍仍需联调验证。

VM 脚本静态编译和协议冒烟测试：

```powershell
.\scripts\test-frame-sender.ps1
```

来源、SHA-256 和验证边界见 [bundle.json](bundle.json)。

## 7. 当前验证状态

已检查：

- 正式 `.sol` 可由 ZIP 工具完整打开，共 124 个条目；
- 内部方案名为 `glue_m06_inspection_v1`，不残留 `beifen-memory`；
- 方案内包含 `SENT_WITHOUT_DEFECT_DETAILS` 版本的帧发送脚本；
- VM 4.4 脚本 API 静态编译和二进制帧字段冒烟测试通过；
- 查看器 GIFR 分包、Mono8/RGB24 内存解码、OK 零落盘和 NG 原子保存通过；
- 查看器支持中文缺陷名称和纯中文显示。

尚未验证：

- 从正式 `.sol` 实际连接查看器并连续发送 OK/NG；
- 断线重连、缺帧、顺序和长时间运行；
- 同帧原图、卡尺图、判定和缺陷类型的人工核对；
- 真实产线节拍、长期稳定性和真实缺陷指标。

这些未完成项不能用“方案文件已归档”代替。

## 8. 常见问题

| 现象 | 检查 |
|---|---|
| 模型加载失败 | 是否把模型放在方案保存的绝对路径；移动目录后需在 VM 中重新选择模型 |
| `sendStatus=ERROR` | 查看脚本打印的原始异常；确认脚本仍读取 `config/vm_frame_sender.ini` |
| `sendStatus=SEND_FAILED` | 查看器是否先启动并监听 `127.0.0.1:5001` |
| `sendStatus=SENT_WITHOUT_DEFECT_DETAILS` | OK 帧已发送，但该 OK 图没有缺陷明细，不是失败 |
| 前端不显示缺陷中文 | 确认查看器已重启到最新 Release，且收到的是 GIFR v1 内存帧 |
| 双击启动脚本提示已运行 | 先关闭现有查看器窗口，再重新双击 |
