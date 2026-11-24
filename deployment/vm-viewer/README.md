# VM 结果查看器

> 状态：0.1.0，源码构建、TCP/时间绑定/历史清理冒烟验证通过；
> 尚未完成真实产线长时间运行和异常场景验收。

Windows WPF 前端，用于接收 VisionMaster 的 OK/NG 结果并展示、保存原图和卡尺
检测图。本程序不引用原交付软件或本仓库的检测算法，不执行分割、量测或缺陷判定。

## 1. 数据链路

```text
VisionMaster 工作流
├── 输出图像：original_yyyyMMddHHmmssfff.bmp
├── 输出图像：caliper_yyyyMMddHHmmssfff.bmp
└── TCP 客户端：127.0.0.1:5000，发送 yyyyMMddHHmmssfff|OK/NG
                              │
                              ▼
VM 结果查看器
├── 监听 TCP 5000 并解析 OK/NG
├── 监视图片目录，按时间选择最近的原图和卡尺图
├── 实时显示两张图片和判定
├── 保存 record.json、原图和卡尺图到本地历史目录
└── 提供历史回看和确认式清理
```

默认路径：

```text
VM 输出图片：D:\GlueVM\results
本地历史目录：D:\GlueVM\history
```

## 2. VM 配置

两个“输出图像”模块：

- 原图前缀：`original`
- 卡尺图前缀：`caliper`
- 时间格式：`YYYYMMddHHmmssfff`

通信模块：

- 类型：TCP 客户端
- 地址：`127.0.0.1:5000`
- 建议消息：`时间|OK` 或 `时间|NG`
- 示例：`20261124151438471|NG`

前端默认在 `127.0.0.1:5000` 监听。VM 也可以只发送 `OK/NG`，此时前端使用
结果到达时间作为匹配参考。

## 3. 时间绑定与历史记录

前端在默认 2000 ms 窗口内分别寻找最接近消息时间的原图和卡尺图，并要求文件
大小和修改时间稳定、图片能够被正常解码后才绑定。窗口内找不到完整图片时，
记录会写入缺图降级信息，不会静默绑定上一轮旧图。

记录号统一按时间生成：

```text
yyyyMMddHHmmssfff
```

优先使用 VM 消息时间；VM 未带时间时使用结果到达时间。历史目录结构：

```text
D:\GlueVM\history\YYYYMMDD\时间记录号_接收时间\
├── original.bmp/png
├── caliper.bmp/png
└── record.json
```

“清除历史”会二次确认，然后删除：

1. 历史根目录中的全部记录、绑定结果和复制图片；
2. 当前 VM 输出目录顶层的支持图片。

清理不会删除目标目录中的非图片文件或子目录内容。该操作不可撤销，被占用而无法
删除的文件会在状态栏中报告。

## 4. 本轮代码审查与修复

已确认并修复：

- 修正 17 位文件名时间戳解析，历史记录不再丢失图片时间；
- 时间窗口外不再回退绑定上一轮最新图片；
- 等待两张输出图写完并确认可解码后再显示和入库；
- 清理与结果接收串行执行，避免清理期间重新写入记录；
- 清理按钮增加防重复点击和二次确认；
- 清理完成状态不再被实时刷新立即覆盖；
- 深色界面的辅助文字和值文字统一为白色高对比显示。

## 5. 构建与验证

要求 Windows 和 .NET 10 SDK。只构建源码时需要 SDK；发布后的 self-contained
程序不要求目标机器安装 .NET。

```powershell
$dotnet = "D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe"

& $dotnet build .\src\GlueInspect.VmViewer\GlueInspect.VmViewer.csproj -c Release
& $dotnet run --project .\tests\GlueInspect.VmViewer.SmokeTests -c Release
```

或使用脚本：

```powershell
.\scripts\test.ps1 -Dotnet "D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe"
.\scripts\publish.ps1 -Dotnet "D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe"
```

`publish.ps1` 默认发布到 `artifacts\vm-viewer`，该目录不进入 Git。

冒烟测试覆盖：

- `时间|OK/NG` 的 TCP 端到端收发；
- 相差 1 ms 的原图/卡尺图时间匹配；
- 时间窗口外不回退旧图；
- 正在写入的图片不被提前入库；
- 时间记录号、图片时间和 `record.json` 保存；
- 历史目录清理及目标目录图片清理范围。

当前验证结果是“前端链路与数据管理冒烟通过”，不是检测准确率、生产节拍或
长期稳定性验收。

常规 Release 构建和 .NET `latest-recommended` 分析规则均为 0 警告、0 错误。

## 6. 当前边界

- 仅支持 Windows WPF，目标框架为 `net10.0-windows`；
- 只负责结果接收、展示、绑定、历史保存和清理，不包含检测算法；
- 默认 TCP 仅面向本机回环地址，没有网络认证和加密；
- 默认时间窗口为 2000 ms，可通过 `--match-window-ms` 调整；
- 尚未完成真实产线节拍、长时间运行、异常断电和磁盘占满测试；
- 清理功能会删除本地数据，使用前必须确认当前输出目录和历史目录。

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
