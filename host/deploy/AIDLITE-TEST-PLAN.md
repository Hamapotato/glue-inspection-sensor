# 板端 AidLite / NPU 测试流程（给板子上的智能体执行）

> 目标：在本板（AidLux，Qualcomm QCS8550 / 广翼智联 FV04）上确认：
> 1）AidLite 能否加载并运行我们的 RPN（ONNX 后端）；
> 2）官方示例中 **CPU vs NPU** 的推理耗时，作为"NPU 是否可行"的直接证据。
>
> 本文档只做**测试与记录**，不修改仓库、不安装系统组件、不编造数据。

## 0. 前置条件

- 仓库已拉到提交 `269e66c` 或更新（`git log -1 --oneline` 确认）；
- 板端已确认：`python3 -c "import aidlite"` 正常；`/vendor/lib64/libQnnHtp.so` 存在；
- 所有测试命令都在**仓库根目录**执行（若路径不同，用 `REPO` 变量替换）：

```bash
export REPO=~/glue-inspection-sensor-reproduction   # 按实际路径修改
cd "$REPO" && git pull && git log -1 --oneline
```

- 测试仅做推理，不涉及相机/PLC，无运动部件风险；
- **不要**安装/升级 AidLite、QNN、MVS；**不要**改仓库文件；需要改示例时先拷贝到 `/tmp`。

## 1. 测试 A：AidLite 跑我们的 RPN（ONNX 后端）

```bash
cd "$REPO"
mkdir -p /tmp/aidlite_test
python3 scripts/aidlite_rpn_test.py \
    --image outputs/rust_samples/0000_normal.png \
    --iters 5 --accel cpu 2>&1 | tee /tmp/aidlite_test/A_rpn_onnx_cpu.log
```

**预期输出**（用于判断是否成功）：
- `[aidlite] lib=... py=... abi=...`（版本号）；
- `[aidlite] load_model ret=0`；
- `[in ] ... name=image shape=[1,1,320,320]`、`[out] ... name=cls / name=reg`；
- `[aidlite] invoke x5: p50≈xx.xx ms`；
- `[result] ...` 后列出 `top0..top4` 的 score 与**原图坐标框**；
- **判据**：`top0` 的框应覆盖胶条区域（正常样例约 `y≈420~540`，`x≈0~1280`）。

可选（GPU 后端，若失败只记录，不影响主结论）：

```bash
python3 scripts/aidlite_rpn_test.py --image outputs/rust_samples/0000_normal.png \
    --iters 5 --accel gpu 2>&1 | tee /tmp/aidlite_test/A_rpn_onnx_gpu.log
```

## 2. 测试 B：官方 YOLOv5 的 CPU 基线（AidLite ONNX 后端）

```bash
cp -r /usr/local/share/aidlite/examples/aidlite_onnx /tmp/aidlite_test/
cd /tmp/aidlite_test/aidlite_onnx/python
python3 onnx_yolov5_multi.py 2>&1 | tee /tmp/aidlite_test/B_yolo_onnx_cpu.log
```

**记录**：
- `invoke cost time` 的 10 次循环结果（脚本会打印每次耗时与汇总）；
- 是否生成结果图 `onnx_yolov5_multi.jpg`；
- 若脚本支持 `--gpu` 参数，再跑一次 GPU 并记录（不支持就跳过，写"该示例不支持 GPU 参数"）。

> 该示例默认输入 640×640，与测试 C 使用同一个 YOLOv5 模型，可直接对比。

## 3. 测试 C：官方 YOLOv5 的 NPU 推理（AidLite QNN248 后端）

```bash
cp -r /usr/local/share/aidlite/examples/aidlite_qnn248 /tmp/aidlite_test/
cd /tmp/aidlite_test/aidlite_qnn248/python
python3 qnn_yolov5_multi.py 2>&1 | tee /tmp/aidlite_test/C_yolo_qnn248_npu.log
```

**记录**：
- `Device : device[...]-type[...]`（确认使用 NPU/加速器）；
- `load model success`；
- `invoke cost time` 的 10 次结果；
- 是否生成结果图。

**失败回退**（按顺序尝试，记录每次错误原文）：

1. 若 qnn248 失败：`cp -r /usr/local/share/aidlite/examples/aidlite_qnn231 /tmp/aidlite_test/` 后运行同名脚本；
2. 再失败则 `aidlite_snpe229` 示例（若存在）尝试，并记录；
3. 全部失败：保留日志，测试 C 记为"未通过"，不要修改系统配置。

## 4. 测试 D（本板不可执行，只记录结论）

我们的 RPN 要上 NPU，需要 QNN `.ctx.bin`；板端**没有** ONNX→ctx.bin 转换工具（`oma`/`aidgen`/qairt 均不是转换器）。因此：

- 本板**不执行**转换；
- 该项记为"待 x86_64 Linux + QNN SDK 转换后验证"；
- 若后续拿到 `.ctx.bin`，按 `scripts/verify_rknn.py` 同样的思路另写 AidLite QNN 验证脚本。

## 5. 结果记录模板

在 `/tmp/aidlite_test/report.md` 写入以下内容（数值必须来自日志原文）：

```markdown
# AidLite / NPU 板端测试记录

- 日期：
- 板子：AidLux / aarch64（QCS8550）
- 仓库 commit：`...`
- AidLite 版本：`...`（从测试 A 日志复制）
- QNN 运行库：`/vendor/lib64/libQnnHtp.so` 等是否存在

## 测试 A：我们 RPN（AidLite ONNX CPU）
- load_model ret：
- 输入/输出张量信息（原文粘贴）：
- invoke p50（ms）：
- top0 框（原图坐标）：
- 判定：通过 / 不通过（框是否覆盖胶条）
- GPU 尝试：通过 / 不支持 / 失败（错误原文）

## 测试 B：官方 YOLOv5 CPU（AidLite ONNX）
- invoke 时间（10 次，ms）：
- 平均 invoke（ms）：
- 结果图是否生成：

## 测试 C：官方 YOLOv5 NPU（AidLite QNN248）
- Device 信息：
- load model：
- invoke 时间（10 次，ms）：
- 平均 invoke（ms）：
- 结果图是否生成：
- 回退记录（如使用 qnn231/snpe 示例）：

## 结论
- NPU 是否可用：是 / 否（依据：测试 C）
- NPU 相对 CPU 的加速比：C 平均 / B 平均 = x.x 倍（同一模型）
- 我们的 RPN 能否走 AidLite ONNX：是 / 否（依据：测试 A）
- 我们的 RPN 上 NPU 阻塞点：缺 QNN `.ctx.bin`（需 x86 Linux 转换）
```

## 6. 回传清单

完成后把以下内容发给主导人：

1. `/tmp/aidlite_test/report.md`；
2. 三个日志文件：`A_rpn_onnx_cpu.log`、`B_yolo_onnx_cpu.log`、`C_yolo_qnn248_npu.log`（以及 GPU/回退日志，如有）；
3. 异常项的原始报错（不要改写）。

## 7. 纪律

- 所有数字必须来自实际日志；失败就写失败，**不得估算**；
- 不改仓库、不装/升级软件、不动相机与 MVS 配置；
- 遇到权限/路径问题，先记录错误原文再停止，不要反复尝试修改系统。
