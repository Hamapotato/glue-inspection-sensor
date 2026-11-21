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
  echo [WARN] 缺少演示权重 outputs\demo\onnx\rpn.onnx（权重不入库，git pull 不会带来）
  if defined GLUE_DEMO_WEIGHTS_URL (
    echo [demo] 从 %GLUE_DEMO_WEIGHTS_URL% 下载权重...
    python scripts\fetch_demo_weights.py --url %GLUE_DEMO_WEIGHTS_URL% || exit /b 1
  ) else (
    echo         方式 A：从构建机下载后启动：
    echo            set GLUE_DEMO_WEIGHTS_URL=http://^<构建机IP^>:8000
    echo            host\run-demo.bat
    echo         方式 B：在构建机生成：powershell -ExecutionPolicy Bypass -File scripts\build-demo-weights.ps1
    exit /b 1
  )
)

set "GLUE_CONFIG=configs\experiments\demo-rpn.yaml"
set "GLUE_SAMPLES=outputs\rust_samples"
set "GLUE_DB=outputs\host.db"

echo [demo] 配置: %GLUE_CONFIG%
echo [demo] 权重: outputs\demo\onnx\rpn.onnx（RPN 定位）
echo [demo] 样例: %GLUE_SAMPLES%
host\target\release\glue-app.exe
