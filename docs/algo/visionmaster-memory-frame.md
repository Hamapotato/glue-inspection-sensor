# VisionMaster 内存图像帧传输

> 状态：内存帧脚本、通信和图形输出接线已归档到正式
> `glue_m06_inspection_v1.sol`；TCP 分包、Mono8/RGB24 解码和 NG-only 保存
> 通过冒烟测试。真实 VM 连续运行尚未验收。
>
> 软件基线：VisionMaster 4.4，脚本模块使用 C# 和 .NET Framework 4.6.1。
>
> 事实来源：脚本 API 以本机
> `D:\VM\VisionMaster4.4.0\Applications\Module(sp)\x64\Logic\ShellModule\Script.Methods.dll`
> 和 VM 4.4 帮助中的“脚本”“脚本开放接口”为准。

## 1. 目标

正式运行时不再让 VM 逐帧输出图片文件，也不让前端监视目录和时间戳配对图片。
VM 在最终判定后，把一个 C# 脚本模块放在流程末尾，将以下内容合并为一个
TCP 二进制帧：

```text
原图 ImageData
卡尺绘制图 ImageData
最终 OK/NG
缺陷类型数组
缺陷个数
帧号和时间戳
```

前端后续应直接接收这一帧，在内存中解码和显示。只有 NG 帧由前端写入缺陷
历史目录；OK 帧不创建图片文件。

发送端和查看器源码现已使用同一 GIFR v1 协议。旧“图片目录 + 文本 TCP”
实现仍保留作迁移对照，但默认查看器入口已切换到内存帧模式。

正式方案文件：

```text
deployment/visionmaster/glue_m06_inspection_v1/solution/glue_m06_inspection_v1.sol
```

方案包内部同时包含 `脚本2`、`图形收集1`、`输出图像1` 和 TCP 通信接线。

## 2. VM 工作流改动

保留当前 `M01` 到 `M12` 的检测主线：

```text
图像源
  -> DL 图像分割
  -> 二值化和形态学
  -> 边缘对模型缺陷检测
  -> Blob 分析
  -> 变量计算
  -> 条件检测
  -> 最终逻辑
  -> 格式化
```

结果传输改为：

```text
M01 原图 ---------------------------┐
                                    │
M06 绘制结果 -> 图形收集 -> 输出图像 -> 卡尺图
                                      │
M11/M06 判定、缺陷类型和缺陷个数 ------┴-> M14 C# 脚本 -> TCP 二进制帧
```

具体改动：

1. 在最终判定完成后增加一个 `脚本` 模块，建议命名为 `VM图像帧发送`。
2. 增加 `图形收集` 模块，把边缘对、左右边缘点、中心点和缺陷框收集为叠加图形；
   再增加一个 `输出图像` 模块，把 `图形收集` 的叠加结果渲染成真正的 `ImageData`，
   将 `输出图像` 的图像输出接到脚本的 `caliperImage`。
3. 将图像源原图接到脚本的 `originalImage`。
4. 将最终逻辑的 `OK/NG`、M06 的缺陷类型数组和缺陷个数接到脚本。
5. 新增 TCP 客户端通信设备，连接前端监听的 `127.0.0.1:5001`。
6. 将脚本配置中的 `tcp_device_id` 改成通信管理里实际的设备 ID。
7. 从正式运行链中移除两个旧的“输出图像”存图模块和旧文本结果发送；新增的
   `输出图像` 仅用于在内存中生成卡尺图，必须保留。迁移验证前可以先禁用旧
   调试分支，不要直接删除方案中的调试分支。

执行顺序仍以检测依赖为准。脚本必须放在最终判定之后，且每帧只执行一次。

## 3. 通信设备配置

在 VM 的“通信管理 -> 设备管理”中新增：

| 参数 | 配置 |
|---|---|
| 设备名称 | `ViewerFrameTcp` |
| 协议类型 | TCP 客户端 |
| 目标 IP | `127.0.0.1` |
| 目标端口 | `5001` |
| 自动重连 | 开启 |
| 数据上传 | 关闭 |

记录设备列表中显示的 `DeviceID`，它与 IP、端口不是同一个概念。脚本使用
`GlobalCommunicateModule.GetDevice(DeviceID)` 取得该设备，因此必须与
`config/vm_frame_sender.ini` 的 `tcp_device_id` 一致。

不要使用“发送数据”模块向这个端口发送字符串；图像帧由 C# 脚本以
`byte[]` 形式发送。

## 4. 脚本模块变量

在 `脚本` 模块的输入/输出变量编辑区按下列名称和类型创建变量。名称是脚本
读取变量时使用的键，修改名称必须同步修改脚本。

| 方向 | 名称 | 类型 | 订阅或取值 |
|---|---|---|---|
| 输入 | `originalImage` | image | 图像源输出的原始图像 |
| 输入 | `caliperImage` | image | `输出图像` 模块输出图像中的输出图像数据 |
| 输入 | `verdict` | string | 最终逻辑输出的 `OK` 或 `NG` |
| 输入 | `defectTypes` | int[] | M06 缺陷类型数组 |
| 输入 | `flawCount` | int | M06 缺陷个数 |
| 输出 | `lastFrameId` | string | 最后一帧 ID，便于调试 |
| 输出 | `sendStatus` | string | `SENT`、`SENT_WITHOUT_DEFECT_DETAILS`、`SENT_UNKNOWN_VERDICT`、`SEND_FAILED` 或 `ERROR` |

VM 4.4 脚本输入变量的“初始值”是绑定前序模块结果或全局变量的入口，不能直接
填普通路径字符串；否则会提示“未找到绑定数据”。因此脚本不再读取
`configPath` 输入变量，而是在 `VmFrameSender.cs` 的 `DefaultConfigPath`
常量中内置配置绝对路径：

```text
D:\WORK\glue-inspection-sensor\deployment\visionmaster\glue_m06_inspection_v1\config\vm_frame_sender.ini
```

如果部署目录移动，必须修改源码中的 `DefaultConfigPath`，重新导入脚本并执行
“预编译”。不要再在脚本模块中创建 `configPath` 输入变量。

`图形收集` 本身只能收集并在界面叠加点、线、框和文本，模块结果只有耗时和
模块状态，不提供 `ImageData` 输出，因此不能直接把 `caliperImage` 绑定到
`图形收集`。VM 4.4 官方帮助中的推荐做法是在 `图形收集` 后面增加
`输出图像` 模块，并使用其模块结果里的“输出图像数据”。

`输出图像` 按以下方式配置：

| 参数 | 配置 |
|---|---|
| 输入源 | `图像源1` 的原始图像 |
| 输出使能 | 开启 |
| 存图使能 | 关闭 |
| 像素格式 | `RGB24` |
| 前项存储设置 | 添加并启用 `图形收集1` |

关闭“存图使能”后，该模块只生成内存中的渲染图，不写图片文件。`存图使能`
默认为关闭，不要误开。

VM 4.4 帮助中的“脚本”模块支持创建多个 `image` 类型输入变量，每个变量
可分别绑定前序模块结果。因此 `originalImage` 和 `caliperImage` 设计为两个
独立输入，而不是把两张图拼成一张。实际保存方案时仍需在 VM GUI 中确认两个
图像订阅能同时生效；预编译成功本身不证明订阅正确。

`defectTypes` 在脚本中最多读取 64 项。若 VM 返回的数量超过该容量，脚本会
明确失败并输出 `sendStatus=ERROR`，不会只发送前 64 项。若现场单帧缺陷可能
超过 64 项，应先扩大脚本中的数组容量并重新预编译。

OK 帧可能没有 `FlawNum` 或 `DefectType` 结果，VM 对这类结果返回非零读取码。
当前脚本只在最终判定为 `OK` 时把读取失败解释为 `flawCount=0`、
`defectTypes=[]`，继续发送内存帧，并输出
`sendStatus=SENT_WITHOUT_DEFECT_DETAILS`。最终判定为 `NG` 时，任一缺陷明细
读取失败仍输出 `sendStatus=ERROR` 并中止该帧，避免丢失 NG 缺陷信息。该状态
表示“帧已发送，但该 OK 帧没有缺陷明细”，不表示通信失败。

脚本文件：

```text
deployment/visionmaster/glue_m06_inspection_v1/scripts/VmFrameSender.cs
```

导入后先执行“预编译”。预编译通过只说明脚本符合 VM 脚本 API；实际网络发送
仍需用 VM 运行验证。

## 5. 配置文件

文件：

```text
deployment/visionmaster/glue_m06_inspection_v1/config/vm_frame_sender.ini
```

| 键 | 默认值 | 含义 |
|---|---:|---|
| `tcp_device_id` | `1` | VM 通信管理中的 TCP 客户端 DeviceID |
| `max_frame_bytes` | `67108864` | 单帧最大字节数，默认 64 MiB |

配置在脚本第一次执行 `Process` 时读取一次，后续帧复用。它不是业务阈值，
不控制检测判定。

## 6. 二进制协议 v1

所有整数使用小端序。固定帧头长度为 `72` 字节。

| 偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 4 | bytes | ASCII `GIFR` |
| 4 | 2 | uint16 | 协议版本，当前为 `1` |
| 6 | 2 | uint16 | 帧头长度，当前为 `72` |
| 8 | 4 | uint32 | 整帧总长度 |
| 12 | 8 | uint64 | `frame_id` |
| 20 | 8 | int64 | UTC Unix 毫秒时间戳 |
| 28 | 1 | uint8 | 判定：`0=OK`、`1=NG`、`2=UNKNOWN` |
| 29 | 1 | uint8 | 保留，当前为 `0` |
| 30 | 2 | uint16 | 缺陷个数 |
| 32 | 2 | uint16 | 缺陷类型数量 |
| 34 | 2 | uint16 | 原图像素格式：`1=Mono8`、`2=RGB24` |
| 36 | 2 | uint16 | 卡尺图像素格式 |
| 38 | 4 | uint32 | 原图宽度 |
| 42 | 4 | uint32 | 原图高度 |
| 46 | 4 | uint32 | 原图行跨度 |
| 50 | 4 | uint32 | 原图字节数 |
| 54 | 4 | uint32 | 卡尺图宽度 |
| 58 | 4 | uint32 | 卡尺图高度 |
| 62 | 4 | uint32 | 卡尺图行跨度 |
| 66 | 4 | uint32 | 卡尺图字节数 |
| 70 | 2 | uint16 | 保留，当前为 `0` |
| 72 | `2*N` | uint16[] | 非零缺陷类型 |
| 72+`2*N` | `L1` | bytes | 原图像素 |
| 72+`2*N+L1` | `L2` | bytes | 卡尺图像素 |

前端必须校验：

- `magic == GIFR`；
- `version == 1`；
- `header_size == 72`；
- `frame_length` 与实际读取和两个图像长度一致；
- `stride * height == image_length`；
- `stride >= width * bytes_per_pixel`；
- `frame_id` 严格大于上一帧；不要求差值为 1，重连后允许跳号，但必须显示
  缺帧或重连降级。

## 7. 像素数据边界

脚本直接发送 `ImageData.Buffer`，不编码 PNG/JPEG，不写临时文件。VM 4.4
脚本接口只暴露 `MONO8` 和 `RGB24` 两种像素格式，其他格式会明确失败。

[假设] `ImageData.Buffer` 的长度按图像高度整除，且行数据在缓冲中连续。
代码据此计算：

```text
stride = buffer_length / height
```

首次前端联调必须使用一张上下和左右都不对称的图，确认首行位置、RGB 通道
顺序和行跨度。不能仅用中心对称图像判断像素方向。

## 8. 失败与降级

脚本失败时不会回退到旧文件通道：

| 情况 | 行为 |
|---|---|
| OK 帧缺少 `FlawNum`/`DefectType` | 按 `0` 和空数组发送，`sendStatus=SENT_WITHOUT_DEFECT_DETAILS` |
| NG 帧缺少 `FlawNum`/`DefectType` | 不发送，`sendStatus=ERROR`，模块返回失败 |
| 图像为空或格式不支持 | 不发送，`sendStatus=ERROR`，模块返回失败 |
| 判定不是 `OK/NG` | 发送 `UNKNOWN`，`sendStatus=SENT_UNKNOWN_VERDICT`，模块返回失败 |
| TCP 设备发送返回非零 | `sendStatus=SEND_FAILED`，模块返回失败 |
| 配置缺失或路径不是绝对路径 | 不发送，`sendStatus=ERROR`，模块返回失败 |
| 帧超过配置上限 | 不发送，`sendStatus=ERROR`，模块返回失败 |

查看器源码已实现二进制接收和 NG-only 落盘。真实 VM 联调前仍可先执行
“预编译”；替换正式结果输出后必须按第 9 节逐项核对。

## 9. 验证顺序

1. 导入脚本并预编译，确认无编译错误。
2. 用正常样本运行，检查 `sendStatus=SENT` 或
   `SENT_WITHOUT_DEFECT_DETAILS`，且 `lastFrameId` 每帧变化。
3. 用断胶样本运行，确认 `verdict=NG`、缺陷类型和缺陷个数正确。
4. 启动查看器源码，核对同一帧的原图、卡尺图、判定和缺陷信息。
5. 连续运行至少覆盖断线重连、缺帧、OK 与 NG 混合场景。
6. 记录发送耗时、帧大小、CPU 和内存占用；未实测前不得填写性能结论。

本文件只证明发送端实现和协议定义，不证明检测精度、产线节拍或前端显示已经
验收。
