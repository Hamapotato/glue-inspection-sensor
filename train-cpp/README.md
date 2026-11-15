# C++ 训练工程（libtorch）

把 RPN 与 YOLOv3-5FL+SPP 的训练从 Python 迁移到 C++，批训练 + 多线程，目标是在同一台机器上把训练吞吐提上去。

## 1. 依赖

- **libtorch**：直接复用仓库 `.venv` 里 pip 安装的 PyTorch（wheel 自带 C++ 头文件与 `TorchConfig.cmake`），无需额外下载 2–3 GB 的独立 libtorch；
- **CMake ≥ 3.24**（VS 2022 Build Tools 自带）；
- **MSVC 2022**（VS Build Tools 的 C++ 工作负载）。

## 2. 构建

```powershell
# 1) 找到 venv 里的 torch
$torch = (Resolve-Path ".venv\Lib\site-packages\torch").Path

# 2) 配置 + 构建（VS 生成器）
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake -S train-cpp -B train-cpp/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$torch"
& $cmake --build train-cpp/build --config Release --target train_cpp

# 3) 运行时需要 torch 的 DLL
$env:PATH = "$torch\lib;$env:PATH"
```

## 3. 运行

```powershell
# RPN：合成胶条数据，160×160，批 16
.\train-cpp\build\Release\train_cpp.exe --task rpn --epochs 20 --batch 16 --img 160 --train-n 256 --threads 8

# YOLO：5 层 + SPP，128×128
.\train-cpp\build\Release\train_cpp.exe --task yolo --levels 5 --epochs 20 --batch 16 --img 128 --train-n 128 --threads 8

# 消融：3 层 / 去掉 SPP
.\train-cpp\build\Release\train_cpp.exe --task yolo --levels 3 --no-spp --epochs 20 ...
```

参数：`--task rpn|yolo`、`--epochs`、`--batch`、`--img`、`--width`（RPN 骨干宽度）、`--levels 3|5`、`--no-spp`、`--lr`、`--threads`、`--train-n`、`--seed`。

## 4. 实现对照

| 项 | Python（`algo/net/`） | C++（本工程） |
| --- | --- | --- |
| RPN 骨干 | 4×stride2 → stride16，256 通道 | 同构（`src/rpn.h`） |
| RPN 先验框 | `anchors_from_kmeans` | `wh_kmeans`（K-means++，25 轮） |
| RPN 损失 | DIoU + 正负采样（256/50%） | 同口径（`rpn_loss`） |
| YOLO 结构 | 5 层 FPN + SPP（4/2/1） | 同构（`src/yolo.h`） |
| YOLO 先验框 | 15 个（5 层×3） | 同表（`default_anchors`） |
| YOLO 损失 | obj（正样本归一化）+ cls + CIoU | 同口径（`yolo_loss`） |
| 批训练 | batch=1 | **batch 可配**（16/32/64） |
| 数据 | PNG + letterbox | 合成张量（内存内，无 IO） |

## 5. 实测性能（本机 CPU，2026-11-15）

| 任务 | 配置 | Python（batch=1） | C++（batch 16） | 提升 |
| --- | --- | --- | --- | --- |
| RPN | 160×160，128 样本 | ~34 samples/s（1.4 s/48） | **319 samples/s** | **≈9×** |
| YOLO 5L+SPP | 128×128，64 样本 | ~2.7 samples/s（≈9 s/24） | **47 samples/s** | **≈17×** |
| YOLO 3L+SPP | 128×128，32 样本 | — | 78 samples/s | — |

注：以上均为**合成数据**上的吞吐，不代表真实数据收敛效果；CPU 为 8 线程。

## 6. 已知限制

1. **数据源**：目前只吃内存内合成数据，未接 PNG/manifest（后续加 `stb_image` 或 OpenCV C++ 读取即可）；
2. **无验证/指标**：训练只打印 loss 与吞吐，M1/M2 指标仍在 Python 侧评测；
3. **导出**：尚未做 TorchScript/ONNX 导出，部署侧（Rust/ONNX）暂未接；
4. **box loss**：YOLO 的 CIoU 用逐正样本解码后计算，与 Python 版数值口径一致但向量化方式不同；
5. **PyTorch 版本**：与 venv 中的 torch 2.15 绑定，换版本需重编。

## 7. 下一步

- 接真实数据（manifest + PNG 解码）；
- 训练结束自动导出 TorchScript，供 Rust 端 `glue-infer` 使用；
- 与 Python 版做同 batch 数值对照（forward/loss 误差 ≤1e-5）；
- 在 GPU（如有）上验证吞吐。
