# Build the demo weight used by host/run-demo.bat (Windows).
# Steps: synthetic dataset -> RPN training (K-means anchors) -> ONNX export.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build-demo-weights.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[1/4] creating .venv and installing requirements"
    py -3 -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}
& $py -c "import torch, onnx, onnxruntime" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/4] installing torch/onnx/onnxruntime"
    & $py -m pip install torch onnx onnxruntime onnxscript
}

Write-Host "[2/4] generating demo dataset (200 train / 40 val, 1280x960)"
& $py scripts\make_demo_dataset.py --train 200 --val 40 --with-reflection
if ($LASTEXITCODE -ne 0) { throw "dataset generation failed" }

Write-Host "[3/4] training RPN (40 epochs, K-means anchors) - about 5-10 minutes"
& $py -u scripts\train_demo_weights.py --task rpn --epochs 40 --val-every 20 --val-count 20 --kmeans-anchors --out outputs\demo
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host "[4/4] exporting ONNX"
& $py scripts\export_onnx.py --rpn-weights outputs\demo\rpn\best.pt --out outputs\demo\onnx
if ($LASTEXITCODE -ne 0) { throw "ONNX export failed" }

Write-Host ""
Write-Host "Done. Weight: outputs\demo\onnx\rpn.onnx"
Write-Host "Run the GUI demo: host\run-demo.bat"
