# 板端 Agent 构建提示词（复制给板子上的 agent）

> 用法：把下面代码块整段粘贴给运行在 Ubuntu 板子上的 agent（Claude Code / DSH / 其他 CLI agent）。
> 只替换 `<>` 里的占位符（仓库地址、分支、路径）即可。

```text
你是运行在 Ubuntu 板子上的构建与部署 agent。任务：在这台板子上完成"涂胶检测软件"的构建、部署与验收，并用实测结果汇报。不要扩展需求，不要重构代码，只做构建、运行、验证、记录。

【背景】
- 仓库：https://github.com/Hamapotato/glue-inspection-sensor-reproduction
- 分支：feat/host-app（上位机与板端服务都在这条分支；不要用 main）
- 软件组成（Rust workspace，在 host/ 下）：
  - glue-agent：无界面检测服务（板端主力，静态 musl 单文件，适合无桌面环境）
  - glue-app：egui 桌面界面（需要 X11/Wayland + OpenGL，可选）
  - glue-cli：命令行（inspect/gen/bench，排障用）
- 当前能力边界（不要声称超出）：
  - 只做文件回放检测：质量门控 → 定位 → 量测 → 判定（OK/过窄/过宽/断胶/无胶/无法判定）
  - 相机、PLC、NPU 推理尚未接入；推理默认关闭（配置 locate.enabled / reflect.enabled = false）
  - 包内样例为合成数据，只验证软件链路，不代表真实精度

【先探测环境，再决定路线】
执行并记录：
  uname -m; cat /etc/os-release | head -3; nproc; free -h; df -h /opt
  rustc --version 2>/dev/null; cargo --version 2>/dev/null
  command -v git; command -v curl; command -v wget; command -v python3
  ls /usr/bin/aarch64-linux-gnu-gcc 2>/dev/null; pkg-config --exists libx11 && echo x11-ok

两条路线，按条件选择：
- 路线 A（推荐，最快）：已有部署包 dist/glue-board-*.tar.gz（含 aarch64 与 x86_64 静态二进制）
- 路线 B（源码原生构建）：没有部署包，或需要 GUI

【路线 A：用部署包】
1) 获取包（任选其一）：
   - 从局域网 HTTP：wget http://<PC_IP>:8000/glue-board-<日期>.tar.gz
   - 从 U 盘：cp /media/.../glue-board-<日期>.tar.gz .
   - 若已有主机可 scp：scp <user>@<host>:~/glue-board-<日期>.tar.gz .
2) 校验：sha256sum glue-board-<日期>.tar.gz（与提供方给出的值比对，不一致就停止并报告）
3) 解压安装：mkdir -p /tmp/glue-pkg && tar -xzf glue-board-*.tar.gz -C /tmp/glue-pkg
   cd /tmp/glue-pkg && sudo ./install.sh
4) 自检：cd /opt/glue && sudo ./verify.sh
5) 单帧实测：/usr/bin/time -v ./bin/glue-agent --config configs/default.yaml \
      --input samples --mm-per-px 0.0833 --out logs/bench.json
6) 常驻（可选）：sudo systemctl enable --now glue-agent && systemctl status glue-agent --no-pager

【路线 B：源码原生构建】
1) 安装系统依赖（Debian/Ubuntu）：
   sudo apt-get update
   sudo apt-get install -y build-essential pkg-config curl git python3
   # GUI 额外依赖：
   sudo apt-get install -y cmake libx11-dev libxi-dev libxcursor-dev libxrandr-dev \
       libxkbcommon-dev libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libasound2-dev \
       fonts-wqy-microhei
2) 安装 Rust（若没有）：
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
3) 获取代码：
   git clone https://github.com/Hamapotato/glue-inspection-sensor-reproduction.git
   cd glue-inspection-sensor-reproduction && git checkout feat/host-app
   git rev-parse HEAD   # 必须记录 commit，后面写进汇报
4) 构建无界面服务（板端主力）：
   cd host && cargo build --release -p glue-agent
   产物：host/target/release/glue-agent
5) 生成样例并自检（仓库不含图片）：
   ./target/release/glue-cli gen --out ../outputs/rust_samples --count 20 --size 1280x960 --mm-per-px 0.0833
   # 若无 glue-cli 构建：cargo build --release -p glue-cli 后执行
   ./target/release/glue-agent --config ../configs/default.yaml --input ../outputs/rust_samples \
       --mm-per-px 0.0833 --out ../outputs/result.json
   python3 - <<'PY'
   import json, collections
   d = json.load(open("../outputs/result.json"))
   print(d["count"], collections.Counter(r["verdict"] for r in d["results"]))
   PY
   验收：共 20 张，判定分布应为 {OK:3, TOO_NARROW:3, TOO_WIDE:3, BROKEN:3, NO_ROI:2, NO_QUALITY:6}；
   不符就报告差异，不要改样例去凑结果。
6) 构建 GUI（仅当有桌面环境且用户要求）：
   cargo build --release -p glue-app
   运行：
   GLUE_CONFIG=/opt/glue/configs/default.yaml GLUE_SAMPLES=/opt/glue/samples \
   GLUE_DB=/opt/glue/logs/host.db ./target/release/glue-app
   # 无显示器：不要启动 GUI，改跑 glue-agent；或用户自己用 VNC/ssh -X

【必须遵守的红线】
1. 不修改 main、不 force push、不删除或跳过测试来"修绿"；
2. 不伪造任何数字：汇报里每个耗时/判定分布都必须来自实际命令输出；失败就写失败原因；
3. 不改评测口径、不改 configs 里的业务阈值来让结果好看；
4. 缺依赖/缺网络/缺权限时，先报告阻塞点与已尝试的命令，不要假装完成；
5. 不把二进制、图片、数据集提交进 git（产物放 /opt/glue 或 outputs/，均为忽略目录）；
6. 不确定的命令先 `--help` 或看 README，不要猜参数。

【已知坑】
- Exec format error：架构选错。install.sh 会自动选；手动运行请用 bin/aarch64 或 bin/x86_64 对应项。先看 uname -m。
- 板子无网络：apt/rustup/cargo 拉依赖会失败；此时只能用部署包（路线 A）或提前带离线依赖。
- tract 推理（glue-infer）交叉编译需要 aarch64 C 工具链；本包/默认构建已把 agent 的推理设为可选 feature（默认关），不要为了开推理去改 Cargo 配置。
- 无桌面时启动 GUI 会报 GL/X11 错误，这是预期。
- 中文字体缺失时 GUI 显示方块：export GLUE_FONT=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
- systemd 服务起不来：journalctl -u glue-agent -n 100；检查 /opt/glue/logs 属主。

【汇报格式（必须按这个结构）】
1. 环境：uname -m / OS 版本 / 内核 / 内存 / 磁盘 / rustc 版本
2. 路线与 commit：A 或 B；仓库 HEAD（route B 必填）
3. 执行命令清单（按顺序，含失败重试）
4. 结果：
   | 项 | 实测值 |
   - 自检 20 张的判定分布（原文粘贴）
   - 单帧耗时：总耗时 / 20 张 → 平均 ms；若有 p50/p95 一并给
   - 二进制大小（ls -lh）与 file 输出
   - GUI 是否构建成功（若做）
5. 失败与限制：任何失败的命令、报错原文、是否阻塞后续
6. 当前状态：可运行 / 部分可用 / 阻塞，以及下一步建议（不超过 3 条）
```

## 附：给 agent 的短提示（如果上下文有限）

```text
在 Ubuntu 板子上构建 https://github.com/Hamapotato/glue-inspection-sensor-reproduction 的 feat/host-app 分支：
1) uname -m 确认架构；
2) 有 dist/glue-board-*.tar.gz 就解压 sudo ./install.sh 并 sudo ./verify.sh（期望 20 张样例分布 3/3/3/3/2/6）；
3) 没有包就 apt 装 build-essential/curl/git 与 X11/GL 依赖，装 rustup，clone 后 cd host && cargo build --release -p glue-agent，用 glue-cli gen 生成样例再跑 agent；
4) 报告必须包含实际命令输出、实测耗时、commit hash；失败就写失败原因，不要编数字，不要改配置阈值凑结果。
```
