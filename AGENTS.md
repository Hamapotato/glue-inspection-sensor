# AGENTS.md

本文件规范**所有在本仓库工作的 AI agent**（以及人类协作者）的行为。
目标只有一个：让"论文复现"这件事的结果**可核查、可复现、不含编造**。

> 违反第 2 节红线的改动，无论多小，都视为无效改动。

---

## 0. 开工前必读（按顺序，约 5 分钟）

1. [README.md](README.md) —— 仓库定位与当前进度
2. [docs/algo/progress.md](docs/algo/progress.md) —— **当前完成度、流程图与阻塞项**
3. [docs/algo/plan.md](docs/algo/plan.md) —— 复现范围与分层架构
4. [docs/algo/p0-spec.md](docs/algo/p0-spec.md) —— 指标口径（M1–M10）与假设登记（A1–A10）
5. [docs/algo/interfaces.md](docs/algo/interfaces.md) —— 接口契约与配置键
6. [docs/algo/tasks.md](docs/algo/tasks.md) —— 当前任务与阻塞项

---

## 1. 仓库定位

- **是什么**：以《基于机器视觉的车身涂胶在线检测系统研究》（天津大学硕士学位论文）为起点的涂胶检测传感器复现项目。
- **不是什么**：不是论文原作者的代码仓库；论文未公开源码、数据与训练参数。
- **当前阶段**：P0 规范与 P1 量测链路已完成（60 项测试），P2 大部分完成；B1/B2 模型代码在 `wip/b1-b2` 分支未验证；尚无训练好的模型与真实数据指标。
- **事实来源**：`references/source/` 下的 PDF（本地，不入库）。一切关于论文的陈述以原文为准。

---

## 2. 红线（Hard rules）

| # | 规则 | 说明 |
|---|---|---|
| R1 | **不得编造数据或指标** | 不虚构实验数字、不"估算"论文没给的指标、不把合成数据结果当成真实数据结果 |
| R2 | **论文事实与复现假设必须分开** | 论文没给的参数写入 `p0-spec.md` §5 假设登记表；代码里用 `[假设]` 注释 |
| R3 | **不得为了通过而改评测口径** | 指标定义以 `p0-spec.md` M1–M10 为准；要改定义必须先改文档并说明理由 |
| R4 | **不得提交第三方受版权限制的材料** | `references/source/` 已被忽略；不要把 PDF、数据集、模型权重提交进 Git |
| R5 | **不得擅自 push / force push / 改远端** | 本地提交可以；推送到 GitHub 需人类确认 |
| R6 | **不得删除或跳过测试来"修绿"** | 修实现，不修测试；`skip` 必须有原因和关联任务号 |
| R7 | **不得静默降级** | 缺标定/缺权重时必须写进结果的 `details`，让下游能看见 |
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
| `docs/*.md`（其余） | BOM / 路线图 / 相机对比 | 仅在明确任务下修改 |
| `references/` | 参考资料说明 | **不改**；`references/source/` 不提交 |
| `data/`（待建） | 数据与标注 | 只提交 manifest 与划分文件，不提交图像 |

---

## 4. 代码规范

### 4.1 环境

- Python **≥3.10**；推荐 3.12（torch 生态最稳）。3.14 亦可（cv2 5.0 / torch 2.15 可用）。
- 核心依赖保持最小：`numpy`、`opencv-python`、`pyyaml`。深度学习为**可选依赖**（torch），导入必须放在函数内或 `try/except` 中，保证无 torch 环境仍能跑传统链路。
- 每个仓库一份独立 venv，不装进系统 Python。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 风格

- 标识符英文，注释/文档字符串中文。
- 所有公开函数带类型标注；数据结构用 `dataclass`。
- 文档字符串**必须注明对应论文位置**，例如：

```python
def bbox_diou(box, boxes):
    """DIoU 相似度（式 3-10），对应论文 3.1.3。"""
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
- 当前工作分支：`feat/algorithm-reproduction`。

### 5.2 提交

- 消息用英文祈使句，首字母大写，≤72 字符；正文写"做了什么 + 为什么"。
- 一个提交只做一件事；文档改动可与对应代码同一提交。
- 提交前必须：`git status` 干净（无 `.pyc`、无临时脚本）、测试通过。

### 5.3 完成定义（DoD）

一次代码改动算完成，必须同时满足：

- [ ] 实现符合 `interfaces.md` 的签名与语义
- [ ] 新增/更新了单元测试，且全部通过
- [ ] 无硬编码阈值（新参数进了 `configs/`）
- [ ] 不改变已有指标口径（若改变，先改 `p0-spec.md`）
- [ ] `tasks.md` 对应项已勾选
- [ ] 若涉及论文对照，报告写清"论文值 / 复现值 / 偏差原因"

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
3. 新建 `algo/<module>.py`，docstring 注明论文章节；
4. 在 `configs/default.yaml` 加参数（如需要）；
5. 写 `tests/test_<module>.py`（unittest）；
6. 跑测试 + 更新文档 + 勾选任务 + 提交。

### 7.2 新增一次消融实验

1. 明确对应论文哪张表（表 3-1/3-2/3-4/4-3/4-4/4-5）；
2. 配置写到 `configs/experiments/<name>.yaml`，不改默认配置；
3. 结果落 `outputs/<name>/`（已被忽略）或 `docs/algo/reports/`；
4. 报告注明 seed、数据版本、commit hash。

### 7.3 修改指标定义

必须同时改三处：`p0-spec.md` 的定义表、`algo/metrics.py` 的实现、受影响的历史报告（加注说明），并在提交信息里说明原因。

---

## 8. 常用命令

```powershell
# 单元测试（unittest，无需 pytest）
py -3 -m unittest discover -s tests -v

# 合成样本冒烟（无数据、无 torch 也能跑）
py -3 scripts/demo_synthetic.py

# 棋盘格标定（需自备标定图）
py -3 scripts/calibrate_board.py --images <dir> --cols 9 --rows 6 --square-mm 5

# 批量评测
py -3 scripts/eval_measure.py --images <dir> --labels <dir> --config configs/default.yaml
```

（脚本随 P1 逐步补齐；命令以 `scripts/` 实际入口为准。）

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
algo/              算法实现（types/config/geometry/preprocess/...）
configs/           默认配置与实验配置
scripts/           可执行入口（demo / 标定 / 报告 / 流程图）
tests/             单元测试
docs/algo/         进度、方案、P0 规范、接口、任务、报告、图表
docs/              BOM、路线图、相机对比
data/              数据规范与 manifest（图像不入库）
references/        参考资料（source 不入库）
```
