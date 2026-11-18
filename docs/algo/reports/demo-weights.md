# 演示权重：数据生成、训练与推理验证

> 日期：2026-11-19 ｜ 性质：**合成数据演示权重**，只用于展示"数据集 → 训练 → 导出 → 推理"的完整链路，不代表论文指标或真实精度。
> 代码：`scripts/make_demo_dataset.py`、`scripts/train_demo_weights.py`；推理：`host/crates/infer`（tract）。

## 1. 数据集

| 项 | 值 |
| --- | --- |
| 生成脚本 | `scripts/make_demo_dataset.py` |
| 训练 / 验证 | 200 张 / 40 张（PNG + `*_labels.json`） |
| 分辨率 | 1280×960 灰度，标定比例 0.0833 mm/px（与 `host/crates/cli gen` 一致） |
| 类别 | normal / narrow / wide / broken / no_roi / reflection |
| 胶条宽度 | 10 mm（120 px）、5 mm（60 px）、15 mm（180 px） |
| 断胶 | 中间留 30 px 缺口，两段各给一个框 |
| 无胶图 | 6×6 点阵背景（有对比度、无长条目标），标签为空 |
| 反光 | 正常胶条 + 椭圆亮斑，附反光框（供 YOLO 用） |
| 存放 | `outputs/demo_dataset/`（已被 `.gitignore` 忽略，不入库） |

## 2. 训练（RPN 定位）

- 模型：`RPN(width=32)`，约 0.99 M 参数，输入 320×320（letterbox）；
- 先验框：**K-means**（论文 3.1.2）由训练集框统计得到：
  `(142.5,30)×3, (300,15), (300,30)×2, (300,45)×3`；
- 优化：AdamW，lr 1e-3，40 epochs，batch=1，每 20 轮验证一次（20 张）；
- 结果（第 40 轮）：**验证集平均 IoU 0.7227，IoU≥0.5 命中率 0.765**（n=17，无胶图因无真值被跳过）；
  参照：不含 no_roi 的上一版训练为 IoU 0.8237 / 命中 1.000，加入 no_roi 负样本后指标下降，属于预期（任务更难）；
- 权重：`outputs/demo/rpn/best.pt`（`RPN.load` 可直接加载，配置含 anchors）。

## 3. 导出 ONNX

```powershell
.venv\Scripts\python.exe scripts\export_onnx.py --rpn-weights outputs\demo\rpn\best.pt --out outputs\demo\onnx
```

- 产物：`outputs/demo/onnx/rpn.onnx`（3.87 MB，单文件）；
- onnxruntime 与 PyTorch 最大误差约 3.9e-3（训练后权重；随机初始化时约 2.4e-6，误差放大量级来自长条框数值范围，仍属可接受范围，未做逐层排查）。

## 4. Rust 推理验证（`feat/host-app` 分支）

配置：`configs/experiments/demo-rpn.yaml`（`locate.enabled=true`，`anchors_wh` 与训练一致，`score_thr=0.95`）。
推理侧新增：`locate.anchors_wh` 支持；RPN 候选框按 IoM>0.3 合并（一条长胶条 → 一个 RoI，断胶两段不相交 → 两个 RoI）。

```powershell
host\target\release\glue-cli.exe inspect --config configs\experiments\demo-rpn.yaml \
  --input outputs\rust_samples --mm-per-px 0.0833 --out outputs\demo\inspect.json
```

20 张合成样例（`outputs/rust_samples`）结果：

| 判定 | 数量 | 与真值 |
| --- | --- | --- |
| OK | 3 | ✅ 正常 |
| TOO_NARROW | 3 | ✅ 过窄（5.00 mm） |
| TOO_WIDE | 3 | ✅ 过宽（15.00 mm） |
| BROKEN | 3 | ✅ 断胶（2 个 RoI） |
| NO_ROI | 2 | ✅ 无胶 |
| NO_QUALITY | 6 | ✅ 过暗/过曝/模糊被质量门控拦下 |

**20/20 与样例类别一致**（此前未加 RoI 合并时，正常图会输出多个 RoI 而误判 BROKEN）。
耗时（本机 CPU，含 RPN 推理）：平均 28 ms，最大 50.6 ms（质量门控拦下的 6 张更快）。

## 5. 如何运行演示

一键生成权重（约 5–10 分钟，Windows；需在本分支执行）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-demo-weights.ps1
```

启动演示 GUI（仓库根目录）：

```powershell
host\run-demo.bat      # Windows
host/run-demo.sh        # Linux/板端（需先构建 glue-app）
```

等价的手动命令：

```powershell
# CLI
host\target\release\glue-cli.exe inspect --config configs\experiments\demo-rpn.yaml \
  --input outputs\rust_samples --mm-per-px 0.0833 --out outputs\demo\inspect.json

# GUI（Windows）
$env:GLUE_CONFIG = "configs/experiments/demo-rpn.yaml"
$env:GLUE_SAMPLES = "outputs/rust_samples"
host\target\release\glue-app.exe
```

GUI 检测页会显示 RPN 输出的 RoI 绿框；配方页把 `mm/px` 设为 0.0833 后可见毫米宽度。

## 6. 限制（必须随演示说明）

1. **合成数据 + 演示权重**：不是论文指标，也不能外推到真实胶条；
2. **score_thr=0.95** 是该演示权重的经验值，换成真实数据需要重新标定；
3. 断胶图像虽然框合并为 2 个 RoI，但**宽度按整条掩膜计算**（9.75 mm），BROKEN 判定来自 RoI 数，与宽度无关；
4. 反光检测（YOLO）权重尚未训练，`reflect.enabled=false`；
5. 板端 `glue-agent` 默认未编入推理（tract 交叉编译需 aarch64 C 工具链），演示在 Windows GUI/CLI 上完成；
6. ONNX 误差 3.9e-3 仅在推理层面记录，尚未做逐层定位。
