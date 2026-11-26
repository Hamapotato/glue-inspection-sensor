# GlueInspect VM Viewer

> 状态：0.2.0；内存帧接收和 UI 冒烟测试通过，不包含检测算法。

独立的 VisionMaster 内存帧查看器。只负责：

- 在 `127.0.0.1:5001` 监听 VM TCP 客户端；
- 从 GIFR v1 字节流拆帧并校验两张图片、判定和缺陷信息；
- 把 Mono8/RGB24 原始像素直接在内存中转换为 WPF 位图；
- 同时显示原图和卡尺检测图；
- 把 VM 的缺陷类型代号同时显示为中文名称；
- OK 只刷新界面，不创建图片或历史目录；
- NG 把两张图和 `record.json` 原子保存到 `D:\GlueVM\history`；
- 回看 NG 历史、缩放、平移和适应窗口。

程序不引用 Core、Vision、Inference、Pipeline 或 Storage，也不执行检测算法。

## 启动

双击 `deployment/vm-viewer/start-viewer.cmd` 可以自动构建 Release 并启动。
也可以在 `deployment/vm-viewer` 目录运行：

```powershell
.\scripts\run-viewer.ps1
```

源码调试方式：

```powershell
dotnet run --project .
```

默认参数：

```text
--history D:\GlueVM\history
--host 127.0.0.1
--port 5001
```

VM 必须配置为 TCP 客户端并主动连接查看器。旧 `--images`、`--match-window-ms`
和 `--tcp-client` 参数仍由兼容代码解析，但默认内存界面不使用目录图片。

## 通信格式

一帧同时包含：

```text
GIFR v1 帧头
缺陷类型 uint16[]
原图原始像素
卡尺图原始像素
```

帧头中的判定编码：

```text
0 = OK
1 = NG
2 = UNKNOWN
```

查看器按整帧长度读取，不依赖一次 TCP 读取对应一帧。帧号必须递增；长度、
行跨度和像素格式不合法时，当前连接会被关闭，VM 自动重连。

## 历史记录

只有 NG 会创建：

```text
D:\GlueVM\history\YYYYMMDD\记录号_接收时间\
├── original.png
├── caliper.png
└── record.json
```

临时目录写完后再改名为正式记录，避免只保存到一半的缺陷记录。OK 帧的图片只
存在于内存和当前界面。
