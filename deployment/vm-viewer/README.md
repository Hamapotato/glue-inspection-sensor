# VM 结果查看器

> 状态：0.2.0。内存帧 TCP 分包、Mono8/RGB24 像素转换、双图显示、
> OK 零落盘和 NG 原子保存均已通过源码冒烟测试；真实 VM 连续运行尚未验收。

Windows WPF 前端，用于接收 VisionMaster 发送的 GIFR v1 内存图像帧。程序不
引用检测算法，不执行分割、量测或缺陷判定。

## 1. 数据链路

```text
VisionMaster C# 脚本
└── TCP 客户端 127.0.0.1:5001
    └── 单帧同时包含原图、卡尺图、OK/NG、缺陷个数和缺陷类型
                         │
                         ▼
VM 结果查看器
├── 从 TCP 流连续拆出完整二进制帧
├── 在内存中把 Mono8/RGB24 原始像素转换为 WPF 位图
├── 实时显示两张图和判定
├── OK：只刷新界面，不创建图片或历史目录
└── NG：把双图和 record.json 原子写入历史目录
```

默认监听地址：

```text
127.0.0.1:5001
```

默认 NG 历史目录：

```text
D:\GlueVM\history
```

## 2. VM 侧要求

VM 使用 TCP 客户端连接查看器：

| 参数 | 配置 |
|---|---|
| 协议类型 | TCP 客户端 |
| 目标 IP | `127.0.0.1` |
| 目标端口 | `5001` |
| 自动重连 | 开启 |
| 数据上传 | 关闭 |

VM 的 `caliperImage` 必须订阅“图形收集后再由输出图像模块生成的
`OutputImage.OutImage`”，不能直接订阅图形收集模块。完整接线见
[VisionMaster 内存图像帧传输](../../docs/algo/visionmaster-memory-frame.md)。

## 3. 协议校验

查看器按 `GIFR v1` 校验：

- magic、协议版本和 72 字节帧头；
- 整帧长度、缺陷类型数组长度和两张图片长度是否闭合；
- `stride * height == image_length`；
- `stride >= width * bytes_per_pixel`；
- 帧号严格递增；
- 像素格式只接受 `1=Mono8`、`2=RGB24`。

TCP 是字节流，一次读取不等于一帧。查看器会持续读取直到帧头声明的完整长度，
因此 VM 分包或粘包不会破坏后面的帧。

### 3.1 缺陷类型显示

查看器同时保留 VM 原始数字代号并显示中文名称：

| 代号 | 中文显示 |
|---:|---|
| 1 | 边缘对宽度缺陷 |
| 2 | 边缘对位置偏移 |
| 3 | 边缘对阶梯缺陷 |
| 4 | 边缘对断裂缺陷 |
| 5 | 边缘对气泡缺陷 |

界面上只显示中文名称，不显示括号和数字代号；`record.json` 中仍保留原始
数字。同一种缺陷出现多次时显示次数，例如 `边缘对断裂缺陷 ×2`。未知代号显示
为 `未知缺陷`，不会直接隐藏。映射按仓库现有 VM 4.4 核对资料记录；若现场模块
枚举发生变化，以 VM 实际 `DefectType` 结果为准并同步修改
`src/GlueInspect.VmViewer/Models/DefectTypeCatalog.cs`。

## 4. NG 历史记录

只有 `verdict=NG` 才创建磁盘记录。目录结构：

```text
D:\GlueVM\history\YYYYMMDD\时间记录号_接收时间\
├── original.png
├── caliper.png
└── record.json
```

图片和 JSON 先写入临时目录，全部成功后再改名为正式记录目录。`record.json`
包含帧号、VM 时间戳、判定、缺陷个数、缺陷类型、接收时间和原始数据摘要。
OK 帧不会创建目录、图片或记录。

“清除历史”只删除 NG 历史目录，不清空当前内存画面。该操作不可撤销。

## 5. 构建与验证

要求 Windows 和 .NET 10 SDK。本轮只改源码，不执行封装或发布。

双击下面的脚本即可自动构建 Release 并启动查看器：

```text
deployment\vm-viewer\start-viewer.cmd
```

也可以在 PowerShell 中运行：

```powershell
.\scripts\run-viewer.ps1
```

启动脚本默认使用
`D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe`；该路径不存在时会使用 `PATH` 中的
`dotnet.exe`。如果查看器已经运行，脚本只显示现有 PID，不会重复启动或关闭窗口。

```powershell
$dotnet = "D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe"

& $dotnet build .\src\GlueInspect.VmViewer\GlueInspect.VmViewer.csproj -c Release
.\scripts\test.ps1 -Dotnet $dotnet
```

直接运行源码：

```powershell
& $dotnet run --project .\src\GlueInspect.VmViewer -c Release
```

冒烟测试覆盖：

- 二进制帧字段和图像元数据解析；
- TCP 分包、粘包和连续两帧接收；
- Mono8/RGB24 到 WPF 位图的内存像素转换；
- 缺陷类型中文映射、未知代号保留和重复类型计数；
- OK 内存帧拒绝落盘，磁盘零文件；
- NG 双图与 `record.json` 原子保存；
- 旧目录链路的兼容测试。

Release 构建为 0 警告、0 错误。这里的通过只能证明源码链路，不代表检测精度、
产线节拍或长时间稳定性已经验收。

## 6. 当前边界

- 默认入口只监听内存帧 `127.0.0.1:5001`；
- 旧目录监听和文本 TCP 实现保留作迁移对照，但不再由默认界面启用；
- 尚未完成真实 VM 连续运行、断线重连、缺帧和实际节拍记录；
- 尚未完成异常断电、磁盘占满和长时间运行验收；
- 本机回环 TCP 没有网络认证和加密。

## 7. 目录

```text
deployment/vm-viewer/
├── README.md
├── bundle.json
├── scripts/
│   ├── publish.ps1
│   └── test.ps1
├── src/GlueInspect.VmViewer/
└── tests/GlueInspect.VmViewer.SmokeTests/
```
