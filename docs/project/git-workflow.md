# Git 操作规范

本文件规定本仓库 **所有分支** 的统一 Git 操作方式。目标：历史可核查、分支可比较、推送可回滚，避免"每个分支各写各的"。
与 `AGENTS.md` 冲突时以 `AGENTS.md` 的红线与推送章节为准；本文件补充分支、提交、同步与清理细节。

---

## 1. 分支模型

| 分支 | 角色 | 规则 |
| --- | --- | --- |
| `main` | 集成分支 | 只接受合并；不直接提交；**永不 force push** |
| `feat/<主题>` | 新功能 | 从最新 `main` 创建；完成后合并回 `main` |
| `fix/<主题>` | 缺陷修复 | 同上 |
| `docs/<主题>` | 文档 | 同上 |
| `chore/<主题>` | 构建、工具、依赖 | 同上 |
| `wip/<主题>` | 未验证的实验 | 只读存档，不合并；需要继续时另开 `feat/` 分支 |
| `backup/<分支>-<时间>` | 变基/改写前的本地备份 | 不推送；确认无误后删除 |

命名只允许小写字母、数字与 `-`，主题用英文（例：`feat/host-app`、`fix/otsu-threshold`）。

---

## 2. 分支生命周期

```text
main ──► 创建 feat/xxx ──► 开发（小步提交）──► 同步 main ──► 合并回 main ──► 删除分支
```

1. **创建前先同步**：`git checkout main && git pull --ff-only`（HTTPS 不通时按 `AGENTS.md` §5.4 用 SSH）；
2. **开发中每天同步一次**：`git fetch origin && git rebase origin/main`；
3. **合并前**：分支必须 `git rebase origin/main`（或 merge），本地测试全绿；
4. **合并方式**：
   - 只有自己的分支、且远端未被他人引用 → `git merge --ff-only` 前先 `rebase`，保持线性；
   - 多人协作或已开 PR → 用 `git merge --no-ff` 保留合并记录，**不要 force push**。
5. **合并后删除**：`git branch -d feat/xxx`、`git push origin --delete feat/xxx`（远端删除需人类确认）。

### 已推送分支改历史

改写历史（rebase / amend）后必须：

```powershell
git push origin feat/xxx --force-with-lease=feat/xxx:<远端旧哈希>
```

- **只能用 `--force-with-lease`**，禁止裸 `--force`；
- 改写前先建本地备份：`git branch backup/feat-xxx-$(Get-Date -f yyyyMMdd)`；
- 改写后立刻验证：`git ls-remote origin refs/heads/feat/xxx` 与本地 HEAD 一致。

---

## 3. 提交规范

- **首行**：英文祈使句，首字母大写，≤72 字符，不用句号；
- **正文**（可选）：写"做了什么 + 为什么"，每行 ≤72 字符；
- **一个提交只做一件事**；文档可与对应代码同一提交；
- **不提交**：构建产物（`target/`、`dist/`、`train-cpp/build/`、`__pycache__/`）、虚拟环境（`.venv/`）、模型权重（`*.pt/*.pth/*.onnx/*.rknn/*.safetensors`）、数据与 PDF（`references/source/`）、数据库（`*.db`）；
- 大文件（>5 MB）需要人工确认后走例外流程。

示例：

```
Add C++ libtorch training for RPN and YOLO5FL

train-cpp/: CMake project reusing the venv torch wheel as libtorch.
```

提交信息与文件规则由 `.githooks/` 强制检查，安装方式见第 7 节。

---

## 4. 同步与冲突

- 同步只做 `fetch` + `rebase`（或 `merge`），**不要** `pull --rebase` 之外的花式操作；
- 冲突解决原则：
  1. **文档冲突**：两边事实都保留（例如测试数、阶段结论），不为了"干净"删掉一方；
  2. **代码冲突**：以 `docs/algo/interfaces.md` 的接口契约为准；
  3. **评测口径冲突**：先改 `p0-spec.md` 再改代码（`AGENTS.md` R3）；
- 冲突解决后必须重跑对应测试再 `rebase --continue`；
- 半途放弃用 `git rebase --abort`，不要手工删 `.git/rebase-merge`。

---

## 5. 推送纪律

1. 推送前检查：

```powershell
git status --short --branch
git log --oneline origin/<分支>..HEAD
git diff --check
```

2. 只推当前分支：`git push origin <分支>`；
3. 推送后核对：

```powershell
git ls-remote origin refs/heads/<分支>
```

4. `main` 的推送只允许快进（`git pull --ff-only` 后 push），不允许 force；
5. 网络失败按 `AGENTS.md` §5.4：先查系统代理 → 命令级 `-c http.proxy` → 重试 3 次 → 明确报错，不擅自 `reset`/强推。

---

## 6. 当前分支清单（2026-11）

| 分支 | 定位 | 基线 |
| --- | --- | --- |
| `main` | 集成 | — |
| `feat/algorithm-reproduction` | 算法文档与 RK3588 调研 | main |
| `feat/b1-b2-models` | RPN / YOLO5FL Python 实现与报告 | main |
| `feat/host-app` | Rust 上位机 + RK3588 部署 | main |
| `feat/train-cpp` | C++ libtorch 训练 | main |
| `feat/vm-results-viewer` | VM 结果查看前端（WPF、TCP、时间绑定、历史和清理） | main |
| `wip/b1-b2` | 早期未验证脚手架（只读） | 旧基线 |

新增分支时同步更新本表。

---

## 7. 钩子（提交前自动检查）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-git-hooks.ps1
```

安装后 `core.hooksPath = .githooks`，每次提交自动执行：

| 钩子 | 检查 |
| --- | --- |
| `pre-commit` | 拒绝构建产物、模型权重、PDF/数据、>5 MB 文件 |
| `commit-msg` | 首行必须英文大写开头、ASCII、≤72 字符；`Merge`/`Revert`/`fixup!` 例外 |

钩子只拦截明显的坏提交，不替代测试与人工检查；需要临时跳过时用 `git commit --no-verify`，并在提交信息里说明原因。

---

## 8. 常用命令速查

```powershell
# 从最新 main 开分支
git checkout main; git pull --ff-only; git checkout -b feat/xxx

# 同步 main 到当前分支
git fetch origin; git rebase origin/main

# 推送并核对
git push origin feat/xxx
git ls-remote origin refs/heads/feat/xxx

# 变基后安全强推
git push origin feat/xxx --force-with-lease=feat/xxx:<旧哈希>

# 合并回 main（线性）
git checkout main; git pull --ff-only; git merge --ff-only feat/xxx; git push origin main

# 清理已合并分支
git branch -d feat/xxx; git push origin --delete feat/xxx
```

---

## 9. 发布标记（可选）

里程碑完成时打 annotated tag：

```powershell
git tag -a v0.1-p1-pipeline -m "P1 traditional measurement pipeline"
git push origin v0.1-p1-pipeline
```

标签一经推送不删除、不移动；如需修正，新建版本号。
