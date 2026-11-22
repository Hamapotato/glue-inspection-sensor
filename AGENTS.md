# AGENTS.md

本文件规范**所有在本仓库工作的 AI agent**（以及人类协作者）的行为。
目标只有一个：让当前涂胶检测技术路线的结果**可核查、可复现、不含编造**。

> 违反第 2 节红线的改动，无论多小，都视为无效改动。

---

## 0. 开工前必读（按顺序，约 5 分钟）

1. [README.md](README.md) —— 仓库定位与当前进度
2. [docs/algo/progress.md](docs/algo/progress.md) —— 当前完成度、主线状态和边界
3. [docs/algo/plan.md](docs/algo/plan.md) —— 分割模型与旋转扫描量测主线
4. [docs/algo/p0-spec.md](docs/algo/p0-spec.md) —— 分割、量测、判定指标与假设登记
5. [docs/algo/interfaces.md](docs/algo/interfaces.md) —— 接口契约与配置键
6. [docs/algo/tasks.md](docs/algo/tasks.md) —— 当前任务与阻塞项
7. [docs/algo/calibration-plan.md](docs/algo/calibration-plan.md) —— 单相机标定、采图和尺寸验证流程
8. [docs/algo/scanline-measurement.md](docs/algo/scanline-measurement.md) —— 扫描矩形、边缘点和中心曲线
9. [docs/algo/algorithm-and-calibration-report.md](docs/algo/algorithm-and-calibration-report.md) —— 当前主线与标定边界

---

## 1. 仓库定位

- **是什么**：以《基于机器视觉的车身涂胶在线检测系统研究》（天津大学硕士学位论文）为起点的涂胶检测传感器复现项目。
- **不是什么**：不是论文原作者的代码仓库；论文未公开源码、数据与训练参数。
- **当前阶段**：唯一主线是“胶条分割模型 + 旋转扫描矩形 + 边缘中点曲线”。分割接口、二值后处理、扫描量测、判定和合成链路已完成（44 项主线测试）；尚无已验证的真实分割模型和真实相机指标。旧论文并列算法在 `archive/legacy-paper-algorithm/`。
- **事实来源**：`references/source/` 下的 PDF（本地，不入库）。一切关于论文的陈述以原文为准。

---

## 2. 红线（Hard rules）

| # | 规则 | 说明 |
|---|---|---|
| R1 | **不得编造数据或指标** | 不虚构实验数字、不"估算"论文没给的指标、不把合成数据结果当成真实数据结果 |
| R2 | **事实与假设必须分开** | 论文没给的参数写入 `p0-spec.md` §4 假设登记表；代码里用 `[假设]` 注释 |
| R3 | **不得为了通过而改评测口径** | 指标定义以 `p0-spec.md` 的 S/M/Q 指标为准；要改定义必须先改文档并说明理由 |
| R4 | **不得提交第三方受版权限制的材料** | `references/source/` 已被忽略；不要把 PDF、数据集、模型权重提交进 Git |
| R5 | **不得擅自 push / force push / 改远端** | 本地提交可以；推送到 GitHub 需人类确认 |
| R6 | **不得删除或跳过测试来"修绿"** | 修实现，不修测试；`skip` 必须有原因和关联任务号 |
| R7 | **不得静默降级** | 缺标定/缺模型时必须写进结果的 `degradations` 或 `details`，让下游能看见 |
| R8 | **不得在代码中硬编码业务阈值** | 全部走 `configs/default.yaml` |

---

## 3. 目录与所有权

| 路径 | 内容 | 修改约束 |
|---|---|---|
| `algo/` | 算法实现 | 主战场；按接口契约改 |
| `configs/` | 配置基线 | 新增阈值必须同步更新 `interfaces.md` §5 |
| `scripts/` | 可执行入口 | 只做编排，不放算法逻辑 |
| `tests/` | 单元测试 | 新增模块必须带测试 |
| `docs/algo/` | 算法文档 | 与代码同步更新；`tasks.md` 勾选 |
| `archive/legacy-paper-algorithm/` | 旧论文并列算法 | 只追溯，不导入主包 |
| `archive/non-mainline-docs/` | BOM、采购、产品路线和汇报材料 | 不进入当前技术路线入口 |
| `docs/README.md` | 文档总索引 | 新增顶层文档时同步维护 |
| `references/` | 参考资料说明 | **不改**；`references/source/` 不提交 |
| `data/` | 数据与标注 | 只提交 manifest 与划分文件，不提交图像 |

---

## 4. 代码规范

### 4.1 环境

- Python **≥3.10**；推荐 3.12（torch 生态最稳）。3.14 亦可（cv2 5.0 / torch 2.15 可用）。
- 核心依赖保持最小：`numpy`、`opencv-python`、`pyyaml`。深度学习为**可选依赖**（torch），导入必须放在函数内或 `try/except` 中，保证无模型运行时仍能跑接口、扫描量测和测试。
- 每个仓库一份独立 venv，不装进系统 Python。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 风格

- 标识符英文，注释/文档字符串中文。
- 所有公开函数带类型标注；数据结构用 `dataclass`。
- 文档字符串必须注明对应接口或技术文档；若实现直接来自论文，再注明论文位置。例如：

```python
def scan_glue_strip(mask, cfg, mm_per_px=None):
    """沿胶条方向生成扫描矩形，见 docs/algo/scanline-measurement.md。"""
```

- 数学实现优先 numpy 向量化，避免逐像素 Python 循环（BFS 除外）。
- 不引入未使用的依赖；新增依赖必须写进 `requirements.txt` 并说明用途。

### 4.3 单位与坐标

- 内部一律用**像素**；只在 `measure` 输出时换算毫米。
- 框格式 `(x1, y1, x2, y2)`，像素坐标，原点左上。
- 图像通道：灰度 `(H,W)`，彩色 BGR `(H,W,3)`。
- 时间单位毫秒。

---

## 5. 工作流

### 5.1 分支

- 不直接改 `main`。分支命名：`feat/<主题>`、`fix/<主题>`、`docs/<主题>`、`chore/<主题>`。
- 当前集成分支：`main`。新的功能、修复和文档工作从 `main` 创建独立分支。
- 完整分支模型、同步/合并/清理规则见 [docs/project/git-workflow.md](docs/project/git-workflow.md)。

### 5.2 提交

- 消息用英文祈使句，首字母大写，≤72 字符；正文写"做了什么 + 为什么"。
- 一个提交只做一件事；文档改动可与对应代码同一提交。
- 提交信息与暂存文件由 `.githooks/` 检查（`scripts/install-git-hooks.ps1` 安装）。
- 提交前必须：`git status` 干净（无 `.pyc`、无临时脚本）、测试通过。

### 5.3 完成定义（DoD）

一次代码改动算完成，必须同时满足：

- [ ] 实现符合 `interfaces.md` 的签名与语义
- [ ] 新增/更新了单元测试，且全部通过
- [ ] 无硬编码阈值（新参数进了 `configs/`）
- [ ] 不改变已有指标口径（若改变，先改 `p0-spec.md`）
- [ ] `tasks.md` 对应项已勾选
- [ ] 若涉及论文对照，报告写清"论文值 / 复现值 / 偏差原因"

### 5.4 GitHub 推送

只有用户明确要求推送时才执行本流程。推送前先确认当前分支、待推送提交和工作区状态；推送失败时保留本地提交，不得使用 `--force`。

#### 推送前检查

```powershell
git status --short --branch
git log -1 --oneline
git remote -v
```

工作区存在未提交改动时，不得静默扩大提交范围。只提交用户要求的内容，并再次检查：

```powershell
git diff --check
git diff --cached --check
```

#### 自动处理 Windows 系统代理

Git for Windows 不保证继承 Windows 系统代理。遇到以下网络错误时，先检查系统代理，而不是直接要求用户重复确认 VPN 状态：

- `Connection was reset`；
- `Failed to connect to github.com port 443`；
- `Recv failure`；
- `Operation timed out`。

读取系统代理配置：

```powershell
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' `
  | Select-Object ProxyEnable, ProxyServer
```

若 `ProxyEnable=1`，从 `ProxyServer` 动态取得代理地址，并检查代理端口是否监听：

```powershell
Test-NetConnection 127.0.0.1 -Port <代理端口> -InformationLevel Detailed
```

代理端口可用时，仅对当前 Git 命令传入代理，不修改全局配置：

```powershell
git -c http.proxy=http://<代理地址>:<端口> `
    -c https.proxy=http://<代理地址>:<端口> `
    push origin <分支>
```

若 `ProxyEnable=0`，先尝试直连。若系统代理已开启但代理端口未监听，明确提示用户 VPN 或代理客户端尚未真正启动，不要反复尝试直连。

`Test-NetConnection github.com -Port 443` 测试的是直连，不能用于判断 HTTP 代理是否可用。需要验证代理链路时使用：

```powershell
curl.exe --proxy http://<代理地址>:<端口> -I --max-time 20 https://github.com
```

#### 重试与验证

- 对临时网络错误最多自动重试 3 次；
- 每次重试前确认本地提交和分支没有变化；
- 推送成功后必须核对远端提交哈希：

```powershell
git -c http.proxy=http://<代理地址>:<端口> `
    -c https.proxy=http://<代理地址>:<端口> `
    ls-remote origin refs/heads/<分支>
```

- 本地提交哈希与远端一致，且 `git status --short --branch` 显示已同步，才算推送完成；
- 认证失败、权限不足、非快进和网络超时必须在最终回复中明确区分；
- 仅在本机已配置 GitHub SSH 公钥时才尝试 SSH；`Permission denied (publickey)` 表示密钥未配置，不能视为 HTTPS 失败后的通用备用方案。

#### 代理使用原则

- 代理地址和端口从当前系统配置读取，不写死；
- 默认使用命令级 `git -c` 参数，避免 VPN 关闭后全局 Git 配置失效；
- 不在日志和文档中写入代理凭据；
- 不因推送失败执行 `git reset`、`git checkout --` 或强推；
- 推送失败时，向用户说明本地提交仍然安全，并给出准确阻塞原因。

---

## 6. 评测纪律

1. **测试集冻结**：一旦确定，不得因结果不好而更换；换集必须新开一版并注明版本。
2. **固定随机性**：所有训练/采样脚本必须支持 `--seed` 并记录实际值。
3. **报告三列**：任何对照论文的指标，必须写"论文值 / 复现值 / 偏差原因"，不允许只写复现值。
4. **区分口径**：平均 IoU ≠ mAP；样本级召回 ≠ 像素 IoU；纯推理耗时 ≠ 端到端耗时。写报告前先查 `p0-spec.md` §1。
5. **合成数据边界**：合成数据只用于冒烟与预训练，**验收必须用真实数据**，并在报告中声明。

---

## 7. 常见任务配方

### 7.1 新增一个算法模块

1. 在 `docs/algo/tasks.md` 找到/新增任务编号；
2. 先在 `interfaces.md` 定接口（若涉及新接口）；
3. 新建 `algo/<module>.py`，docstring 注明对应接口或技术文档；
4. 在 `configs/default.yaml` 加参数（如需要）；
5. 写 `tests/test_<module>.py`（unittest）；
6. 跑测试 + 更新文档 + 勾选任务 + 提交。

### 7.2 新增或更新一次模型/量测实验

1. 在 `docs/algo/tasks.md` 找到或新增 M 组任务；
2. 明确使用合成数据还是真实数据，并记录版本、seed 和 commit hash；
3. 分割模型实验与扫描量测实验分开报告，不混用指标；
4. 结果落 `outputs/<name>/`（已被忽略）；需要入库的结论写入 `docs/algo/`；
5. 合成结果必须标记非验收数据。

### 7.3 修改指标定义

必须同时修改 `p0-spec.md` 的定义表、对应评测实现和受影响报告，并在提交信息里说明原因。不得通过修改指标定义掩盖失败结果。

---

## 8. 常用命令

```powershell
# 单元测试（unittest，无需 pytest）
py -3 -m unittest discover -s tests -v

# 合成掩膜主线冒烟
py -3 scripts/demo_scanline.py --out outputs/demo

# 生成 VisionTrain v5 合成数据
py -3 scripts/generate_visiontrain_dataset.py `
  --out outputs/synthetic/visiontrain-glue-binary-v5 `
  --samples 60 `
  --seed 20261120

# 棋盘格标定（需自备标定图）
py -3 scripts/calibrate_board.py --images <dir> --cols 13 --rows 9 --square-mm 4

# 导出 VisionTrain 二分类标注
py -3 scripts/export_visiontrain_annotations.py `
  --source outputs/synthetic/visiontrain-glue-binary-v5 `
  --out outputs/visiontrain-export/visiontrain-glue-binary-v5-full
```

---

## 9. Agent 的提问边界

**可以自行决定**：实现细节、变量命名、内部重构、测试写法、文档措辞。

**必须先问人类**：

1. 改指标口径或测试集；
2. 改 `interfaces.md` 的接口签名；
3. 引入新的重型依赖（torch / TensorRT / onnxruntime 之外）；
4. 删除或重命名已有文件、目录；
5. push、开 PR、改远端配置；
6. 任何"为了让数字好看"而调整评测的做法（应直接拒绝）。

---

## 10. 目录速查

```
algo/              分割接口、扫描量测、判定、标定
configs/           主线配置
scripts/           可执行入口（数据集 / 导出 / 标定 / 主线演示）
tests/             单元测试
docs/algo/         当前技术路线文档
data/              数据规范与 manifest（图像不入库）
archive/           旧算法与非主线文档
references/        参考资料（source 不入库）
```
