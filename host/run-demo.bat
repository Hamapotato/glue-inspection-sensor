@echo off
rem 启动"推理演示"版上位机：RPN 定位（训练好的演示权重）+ 传统量测/质量门控
rem 在仓库根目录运行：host\run-demo.bat
setlocal
cd /d "%~dp0.."

if not exist "host\target\release\glue-app.exe" (
  echo [ERROR] 缺少 GUI 二进制 host\target\release\glue-app.exe
  echo         请先在 host 目录执行 build.bat
  exit /b 1
)
if not exist "outputs\demo\onnx\rpn.onnx" (
  echo [ERROR] 缺少演示权重 outputs\demo\onnx\rpn.onnx
  echo         请先执行（脚本在 feat/b1-b2-models 分支）：powershell -ExecutionPolicy Bypass -File scripts\build-demo-weights.ps1
  echo         或手动：python scripts\make_demo_dataset.py ^&^& python scripts\train_demo_weights.py --task rpn --kmeans-anchors ^&^& python scripts\export_onnx.py --rpn-weights outputs\demo\rpn\best.pt --out outputs\demo\onnx
  exit /b 1
)

set "GLUE_CONFIG=configs\experiments\demo-rpn.yaml"
set "GLUE_SAMPLES=outputs\rust_samples"
set "GLUE_DB=outputs\host.db"

echo [demo] 配置: %GLUE_CONFIG%
echo [demo] 权重: outputs\demo\onnx\rpn.onnx（RPN 定位）
echo [demo] 样例: %GLUE_SAMPLES%
host\target\release\glue-app.exe
