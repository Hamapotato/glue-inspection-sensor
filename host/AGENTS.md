# 上位机开发规范（Rust / egui）

本文件规范 `host/` 目录下上位机软件的开发：AI agent 与人类协作者都必须遵守。
技术栈、目录、UI 规范、验收标准以本文件为准；与根目录 `AGENTS.md` 冲突时，根目录的红线优先。

---

## 0. 开工前必读

1. [docs/host-software-architecture.md](../docs/host-software-architecture.md) —— 架构、里程碑、数据模型
2. 本文件 —— 开发规范与 UI 规范
3. [../AGENTS.md](../AGENTS.md) —— 仓库级红线（不得编造指标、论文事实分离等）

---

## 1. 项目定位

- **是什么**：涂胶检测传感器的上位机（HMI）：文件回放 → 检测 → 判定 → 显示 → 记录 → 统计 → 配方管理。
- **不是什么**：不是训练工具（训练在 `train-cpp/`）；不是视觉算法库（算法在 `algo/` 与 `glue-vision`）。
- **当前范围**：文件回放 + 检测/配方/历史/统计/设置；相机与 PLC 未接入（模拟信号）。

---

## 2. 技术栈（锁定）

| 项 | 选型 | 约束 |
| --- | --- | --- |
| 语言 | Rust 2021，stable（1.95+） | 不用 nightly 特性 |
| GUI | `eframe`/`egui` **0.31.x** | 升级需评审并同步本文档 |
| 存储 | `rusqlite`（bundled SQLite） | 表结构变更必须写迁移或重建说明 |
| 图像 | `image` crate（png/jpeg/bmp） | 不在 GUI 线程做重计算 |
| 构建 | MSVC 2022 Build Tools + Windows SDK | 构建前加载 `vcvars64.bat`（`build.bat` 已封装） |
| 平台 | Windows x64 优先；crates/agent 交叉编译到 aarch64-musl（RK3588） | 不引入仅 Windows 可用的 crate（除非隔离） |

---

## 3. 目录与所有权

```
host/
├─ Cargo.toml               # workspace
├─ Cargo.lock               # 必须提交，锁定依赖
├─ build.sh / build.bat     # 构建入口（加载 MSVC 环境）
├─ run-app.bat              # 启动 GUI
├─ run-headless.bat         # 无头检测（回归/CI 用）
└─ crates/
   ├─ core/                 # 类型、配置、判定（不依赖 GUI/图像库）
   ├─ vision/               # 纯 Rust 视觉算法
   ├─ pipeline/             # 编排、计时
   ├─ store/                # SQLite 记录与统计
   ├─ app/                  # egui 界面与状态
   ├─ agent/                # 无界面检测服务（RK3588 / Linux 板端）
   ├─ camera/               # 相机采集抽象（GLUE 帧协议；MVS 由脚本接入）
   ├─ infer/                # ONNX 推理（tract，纯 Rust）
   └─ cli/                  # 命令行（inspect/gen/bench）
```

- 依赖方向：`app/cli → pipeline/store → vision → core`，**不得反向引用**；
- `core` 不依赖图像库；`store` 不依赖 GUI；`vision` 不依赖推理。

---

## 4. 红线（本目录）

| # | 规则 |
| --- | --- |
| H1 | **不得硬编码业务阈值**：标准宽度、报警线、标定比例全部来自 `configs/default.yaml` 或配方 JSON |
| H2 | **不得伪造性能或指标**：界面只显示实测耗时；不得写死"合格率 100%"等展示值 |
| H3 | **判定逻辑只能来自 `glue-core::decide`**：界面不做二次判定，不得在 UI 层改变结论 |
| H4 | **UI 线程不做重计算**：检测、读图、批量操作不得阻塞画布刷新（后续接相机/推理时更严格） |
| H5 | **不得删除/跳过测试修绿**：修实现，不修测试 |
| H6 | **配置与数据路径外置**：不写死用户目录路径；默认值只出现在设置页初始值 |
| H7 | **中文界面**：所有用户可见文案用中文；代码标识符用英文；提交信息用英文 |

---

## 5. 工作流

- 分支：`feat/host-app`（本目录的所有改动在此分支）；不直接改 `main`。
- 提交前：
  1. `cargo fmt --all`
  2. `cargo clippy --all-targets -- -D warnings`（允许少量既有 warning，但新增必须清零）
  3. `cargo test --release`
  4. 有界面的改动手动跑一次 `run-app.bat`（无交互桌面时跑 `run-headless.bat`）
- 提交信息：英文祈使句，≤72 字符，正文写"做了什么 + 为什么"。

### 完成定义（DoD）

- [ ] 改动有对应测试（状态逻辑/存储/算法），或说明为何不适用
- [ ] `cargo test --release` 全绿
- [ ] 无新的 clippy warning
- [ ] 新增配置项同步更新架构文档与本文档
- [ ] 涉及 UI 的改动：中文文案、颜色符合第 6 节规范、空状态/错误态有提示

---

## 6. UI 规范

### 6.1 视觉

| 项 | 值 |
| --- | --- |
| 背景 | `#F4F6F9`；卡片纯白；侧栏 `#1E293B` |
| 主色 | `#2563EB`（按钮/选中） |
| 判定色 | 合格 `#16A34A`｜预警 `#D97706`｜报警 `#DC2626`｜未找到 `#6B7280`｜反光 `#7C3AED` |
| 字体 | 系统中文（DengXian → SimHei → 微软雅黑），字号：标题 22、页头 18、正文 14、辅助 12 |
| 间距 | 全局 8 px；卡片内边距 12 px；卡片圆角 6–8 px |

### 6.2 布局

- 左侧固定导航（168 px），顶部页头（当前页 + 快捷信息），中间内容区，底部状态栏；
- 内容区按"卡片"分组：工具栏卡片 → 主内容卡片 → 辅助卡片；
- 检测页：左图像（深色画布，绿框标注），右侧结果卡（大号判定徽章 + 宽度 + 耗时）；
- 统计页：指标卡（总检测/合格率/p50/p95）+ 判定分布条形图 + 宽度直方图。

### 6.3 交互

| 操作 | 约定 |
| --- | --- |
| 快捷键 | `←/→` 上/下一张；`空格` 检测当前；`Ctrl+B` 批量检测；`F5` 刷新历史统计 |
| 空状态 | 必须给出下一步指引（如"请先扫描目录"） |
| 错误 | 状态栏红色显示原因，不弹模态框打断 |
| 长操作 | 批量检测显示进度/计数，界面保持可响应 |
| 危险操作 | 清空历史需二次确认（或提供撤销提示） |

---

## 7. 性能预算

| 场景 | 目标 |
| --- | --- |
| 单帧检测（1280×960） | ≤ 33 ms（当前 p50 约 8 ms） |
| 批量检测 | 不阻塞 UI；进度可见 |
| 启动 | ≤ 2 s |
| 内存 | 常驻 ≤ 500 MB |

界面只显示实测值；性能回归超过 20% 时在报告/提交中说明原因。

---

## 8. 常用命令

```powershell
# 构建（会自动加载 MSVC 环境）
.\build.bat

# 启动界面
.\run-app.bat

# 启动"推理演示"版界面（RPN 定位 + 演示权重）
.\run-demo.bat

# 无头检测（回归）
.\run-headless.bat

# 相机连续检测（需配置 camera.command，见 deploy/CAMERA-MVS.md）
.\target\release\glue-agent.exe --config ..\configs\experiments\camera-synthetic.yaml --camera --frames 5

# 交叉编译 RK3588 版（产物 dist/rk3588）
powershell -ExecutionPolicy Bypass -File scripts\build-aarch64.ps1

# 打包 Ubuntu 板端部署包（aarch64 + x86_64，产物 dist/glue-board-*.tar.gz）
powershell -ExecutionPolicy Bypass -File ..\scripts\build-dist.ps1

# 测试 / 格式 / 检查
cargo test --release
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

---

## 8.1 部署（RK3588）

- 无界面服务：`scripts/build-aarch64.ps1` 交叉编译 `glue-agent`（aarch64-musl 静态单文件）；
- **部署包**：`../scripts/build-dist.ps1` 同时构建 aarch64/x86_64 并组装 `install.sh` / `verify.sh` / 样例；
- 上板清单：[deploy/UBUNTU-CHECKLIST.md](deploy/UBUNTU-CHECKLIST.md)；拷板、systemd、字体、GUI 板端编译见 [DEPLOY-RK3588.md](DEPLOY-RK3588.md)；
- 部署相关改动必须同时更新上述文档与清单。

---

## 9. 常见任务配方

### 9.1 新增一个页面

1. `state.rs` 加 `Page` 枚举值与状态字段；
2. `ui.rs` 加 `xxx_page()`，在 `update()` 的 match 中挂载；
3. 左侧导航加图标与文案；
4. 补状态逻辑测试（`state.rs` 的 tests）；
5. 更新架构文档 §11 页面清单。

### 9.2 新增一个指标

1. 先确认口径（`docs/algo/p0-spec.md`）；
2. `glue-store` 增加查询与聚合，写单元测试（内存库）；
3. `state.rs` 暴露字段，`ui.rs` 只做展示；
4. 报告中注明"实测"与"样本量"。

### 9.3 接入相机（后续）

1. 新建 `glue-camera` crate，实现 `FrameSource` trait；
2. 采集线程独立，通过有界队列送帧；
3. 不允许在 UI 线程调用 SDK 阻塞接口；
4. 掉线/超时按架构文档 §4.4 降级。
