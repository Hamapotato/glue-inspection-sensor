# Ubuntu 板端部署检查清单（当天照做）

> 目标：在一块能跑 Ubuntu 的板子上，30 分钟内验证"检测软件可运行、判定正确、性能可测"。
> 包：`dist/glue-board-<日期>.tar.gz`（含 aarch64 与 x86_64 两个静态二进制，自动选架构）。

---

## 0. 出发前（今天）

- [ ] 生成最新包：`powershell -ExecutionPolicy Bypass -File scripts\build-dist.ps1`
- [ ] 产物：`dist/glue-board-<日期>.tar.gz`（约 7 MB）、`dist/linux-board/`（未压缩目录）
- [ ] 确认包内 `bin/aarch64/glue-agent`、`bin/x86_64/glue-agent` 为 ELF（`file` 可查）
- [ ] 把 tar 包拷到 U 盘/共享目录，或准备 `scp`
- [ ] 知道板子的登录方式（IP、用户名、密码/密钥）

## 1. 上板后先看环境（2 分钟）

```bash
uname -m            # aarch64 或 x86_64（决定用哪个二进制）
cat /etc/os-release # Ubuntu 版本
df -h /opt          # 至少留 200 MB
```

- aarch64 → 包内 `bin/aarch64/glue-agent`
- x86_64 → 包内 `bin/x86_64/glue-agent`

两个二进制均为 **musl 静态链接**，不依赖 glibc，Ubuntu 18.04+ 通用。

## 2. 传输与安装（5 分钟）

```bash
# 本机（Windows）：
scp dist/glue-board-<日期>.tar.gz user@<board-ip>:/home/user/

# 板端：
cd /home/user
mkdir -p glue-pkg && tar -xzf glue-board-<日期>.tar.gz -C glue-pkg
cd glue-pkg
sudo ./install.sh
```

`install.sh` 会：建 `/opt/glue/{bin,configs,samples,logs}` → 按架构装二进制 → 建 `glue` 用户 → 装 systemd 单元（不自动启动）。

## 3. 自检（2 分钟，关键一步）

```bash
cd /opt/glue
sudo ./verify.sh
```

预期：
- 输出 8 张样例的判定分布：**1 OK / 1 TOO_NARROW / 1 TOO_WIDE / 1 BROKEN / 1 NO_ROI / 3 NO_QUALITY**；
- 打印 `[PASS]`；
- 结果文件 `/opt/glue/logs/verify.json` 含 `verdict / width_mm / timings_ms / rois`。

若 python3 未安装：`sudo apt install -y python3`（仅统计用，非必需）。

## 4. 性能实测（1 分钟）

```bash
cd /opt/glue
time ./bin/glue-agent --config configs/default.yaml --input samples --mm-per-px 0.0833 --out logs/bench.json
```

记录：总耗时 / 8 张 ≈ 单帧耗时。参考值：x86 开发机 p50 ≈ 8 ms；aarch64 板端预计 20–40 ms（需实测，不要照搬）。

## 5. 常驻服务（可选，2 分钟）

```bash
sudo systemctl enable --now glue-agent
systemctl status glue-agent
journalctl -u glue-agent -f
# 服务会每 5 秒扫描 /opt/glue/samples，结果追加 /opt/glue/logs/results.jsonl
```

停止/卸载：

```bash
sudo systemctl disable --now glue-agent
sudo rm -f /etc/systemd/system/glue-agent.service
sudo systemctl daemon-reload
sudo rm -rf /opt/glue
```

## 6. GUI（需要桌面环境；无桌面可跳过）

在**仓库根目录**（含 `host/` 的目录）执行：

```bash
sudo ./deploy/build-gui-on-board.sh
```

脚本会：apt 安装 X11/GL 依赖与中文字体 → 装 Rust（若没有）→ `cargo build --release -p glue-app`。

运行：

```bash
cd host
GLUE_CONFIG=/opt/glue/configs/default.yaml \
GLUE_SAMPLES=/opt/glue/samples \
GLUE_DB=/opt/glue/logs/host.db \
./target/release/glue-app
```

- 中文字体缺失：`export GLUE_FONT=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`；
- 无显示器但想远程看：`ssh -X` 转发，或板端装 VNC/x11vnc；
- 纯采集/无界面场景继续用 `glue-agent` 即可。

## 7. 常见问题

| 现象 | 原因/处理 |
| --- | --- |
| `cannot execute binary file: Exec format error` | 架构选错（aarch64 包在 x86 上跑）；`install.sh` 会自动选，手拷二进制时注意 |
| `Permission denied` | 未 `chmod +x` 或未用 sudo 安装 |
| verify 判定分布不符 | 样例被改动/缺少 PNG；重新解压包再跑 |
| GUI 启动报 GL/X11 错误 | 无桌面或缺 `libgl1-mesa-dev`、`libx11-dev`；用 `build-gui-on-board.sh` 补齐 |
| GUI 界面中文显示方块 | 未装中文字体，按第 6 节设置 `GLUE_FONT` |
| systemd 服务反复重启 | `journalctl -u glue-agent`；常见为路径/权限（/opt/glue/logs 属主） |
| 结果全是 NO_QUALITY | 样例文件损坏或配置里 `quality` 阈值被改过；核对 `configs/default.yaml` |

## 8. 现场记录表（建议填写）

| 项 | 值 |
| --- | --- |
| 板子型号 / SoC | |
| Ubuntu 版本 / 内核 | |
| 架构（uname -m） | |
| 单帧耗时（8 张平均） | |
| verify 结果 | PASS / FAIL |
| GUI 是否启动 | 是 / 否（原因） |
| 备注（相机、显示、权限） | |

## 9. 边界说明

- 包内是**合成样例**，只证明"软件链路可跑、判定逻辑正确"，不代表真实产线精度；
- 当前未接相机、未接 NPU 推理（`glue-infer` 已有原始推理但与 pipeline 未集成）；明天上板不涉及这两项；
- 包内二进制对应提交：`feat/host-app`（含质量门控 `NO_QUALITY`）；若后续代码更新，重新执行第 0 步打包。
