# GlueInspect VM Viewer

独立的 VisionMaster 结果查看器。只负责：

- 监视 `D:\GlueVM\results` 中最新的 `original*` 和 `caliper*` 图片；
- 展示原始图和带卡尺的检测图；
- 在 `127.0.0.1:5000` 接收 VM 发送的 `时间|OK/NG`；
- 每次结果自动保存原图、卡尺图和 `record.json` 到本地历史目录；
- 在界面右侧回看历史记录并切换到实时画面；
- 支持图片缩放、平移和适应窗口。

本程序不引用原 GlueInspect 的 Core、Vision、Inference、Pipeline 或 Storage
项目，也不执行任何检测算法。

## 启动

```powershell
dotnet run --project .
```

默认参数：

```text
--images D:\GlueVM\results
--history D:\GlueVM\history
--host 127.0.0.1
--port 5000
--match-window-ms 2000
```

默认由前端监听 5000 端口，VM 使用 TCP 客户端连接。若 VM 已配置为 TCP
服务端，可增加 `--tcp-client`，让前端改为主动连接。

## 通信格式

VM 的两张输出图建议使用同一时间格式：

```text
YYYYMMddHHmmssfff
```

例如：

```text
original_20261124151438471.bmp
caliper_20261124151438470.bmp
```

两张图的毫秒值允许有少量差异。TCP 结果带上本次时间时使用：

```text
时间|OK
时间|NG
```

例如：

```text
20261124151438471|NG
```

前端会在 `--match-window-ms` 指定的时间窗口内，分别选取最接近该时间的原图
和卡尺图。默认窗口为 2000 ms。

如果 VM 仍只发送 `OK/NG`，前端使用结果到达时间作为参考时间进行同样的匹配。
结果先到时，前端会在时间窗口内短暂等待图片写完；窗口内仍找不到完整图片时，
历史记录会明确标记缺图，不会回退绑定上一轮图片。

旧的 `数字序号|OK/NG` 消息仍可解析，但历史记录号统一按时间生成：

```text
18|NG
```

## 历史记录

默认保存到 `D:\GlueVM\history\YYYYMMDD\记录号_时间\`，每次检测包含：

```text
original.png
caliper.png
record.json
```

`record.json` 保存时间记录号、两张图片的文件时间、OK/NG、接收时间和原始
文件名。右侧历史列表按接收时间倒序排列，点击记录即可回看当时的图片和结论。

记录号统一按时间生成，格式为 `yyyyMMddHHmmssfff`。VM 消息未携带时间时，
使用结果到达前端的时间。界面中的“清除历史”会二次确认，然后删除历史目录中的
绑定记录和复制图片，并清空当前 VM 输出目录顶层的支持图片文件。
