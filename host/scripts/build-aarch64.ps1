# 交叉编译 RK3588（aarch64）版本的无界面检测服务
# 用法：powershell -ExecutionPolicy Bypass -File scripts\build-aarch64.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$root = Split-Path (Get-Location) -Parent

Write-Host "[1/4] 安装 aarch64 target..."
rustup target add aarch64-unknown-linux-musl | Out-Null

Write-Host "[2/4] 编译 glue-agent（静态 musl，无系统依赖）..."
cargo build --release -p glue-agent --target aarch64-unknown-linux-musl
if ($LASTEXITCODE -ne 0) { throw "编译失败" }

Write-Host "[3/4] 收集部署文件到 dist\rk3588 ..."
$out = Join-Path $root "dist\rk3588"
New-Item -ItemType Directory -Force -Path "$out\bin", "$out\configs", "$out\samples", "$out\logs" | Out-Null
Copy-Item "target\aarch64-unknown-linux-musl\release\glue-agent" "$out\bin\glue-agent"
Copy-Item (Join-Path $root "configs\default.yaml") "$out\configs\default.yaml"
Copy-Item (Join-Path $PSScriptRoot "..\deploy\glue-agent.service") "$out\glue-agent.service"

Write-Host "[4/4] 完成：$out"
Get-ChildItem "$out\bin" | Select-Object Name, Length
