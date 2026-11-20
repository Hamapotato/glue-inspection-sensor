# Package Linux board deployment bundle:
#   glue-agent (aarch64 + x86_64 static musl) + config + samples + scripts
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build-dist.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$hostDir = Join-Path $root "host"
$out = Join-Path $root "dist\linux-board"

Write-Host "[1/5] Add cross-compile targets..."
rustup target add aarch64-unknown-linux-musl x86_64-unknown-linux-musl | Out-Null

Write-Host "[2/5] Build glue-agent for both architectures..."
Push-Location $hostDir
try {
    cargo build --release -p glue-agent --target aarch64-unknown-linux-musl
    if ($LASTEXITCODE -ne 0) { throw "aarch64 build failed" }
    cargo build --release -p glue-agent --target x86_64-unknown-linux-musl
    if ($LASTEXITCODE -ne 0) { throw "x86_64 build failed" }
} finally {
    Pop-Location
}

Write-Host "[3/5] Assemble bundle at $out ..."
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path "$out\bin\aarch64", "$out\bin\x86_64", "$out\configs", "$out\configs\experiments", "$out\samples", "$out\deploy", "$out\scripts" | Out-Null
Copy-Item (Join-Path $hostDir "target\aarch64-unknown-linux-musl\release\glue-agent") "$out\bin\aarch64\glue-agent"
Copy-Item (Join-Path $hostDir "target\x86_64-unknown-linux-musl\release\glue-agent") "$out\bin\x86_64\glue-agent"
Copy-Item (Join-Path $root "configs\default.yaml") "$out\configs\default.yaml"
Copy-Item (Join-Path $root "configs\experiments\camera-mvs.yaml"), (Join-Path $root "configs\experiments\camera-synthetic.yaml") "$out\configs\experiments\" 2>$null
Copy-Item (Join-Path $root "scripts\camera_mvs.py"), (Join-Path $root "scripts\camera_synthetic.py") "$out\scripts\"
Copy-Item (Join-Path $hostDir "deploy\*.sh"), (Join-Path $hostDir "deploy\glue-agent.service"), (Join-Path $hostDir "deploy\README.txt") "$out\deploy\"
Copy-Item (Join-Path $hostDir "deploy\README.txt") "$out\README.txt"

# Samples: one per kind (normal/narrow/wide/broken/no_roi/dark/overexposed/blurry)
$src = Join-Path $root "outputs\rust_samples"
if (Test-Path $src) {
    $picked = @{}
    Get-ChildItem $src -Filter *.png | Sort-Object Name | ForEach-Object {
        $kind = ($_.BaseName -split '_', 2)[1]
        if (-not $picked.ContainsKey($kind)) { $picked[$kind] = $_.FullName }
    }
    $picked.Values | ForEach-Object { Copy-Item $_ "$out\samples\" }
}

Write-Host "[4/5] Normalize shell scripts to LF ..."
Get-ChildItem "$out\deploy" -Filter *.sh | ForEach-Object {
    $text = Get-Content $_.FullName -Raw
    [System.IO.File]::WriteAllText($_.FullName, $text.Replace("`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))
}

Write-Host "[5/5] Create tar.gz ..."
$stamp = Get-Date -Format "yyyyMMdd"
$tarball = Join-Path $root "dist\glue-board-$stamp.tar.gz"
if (Test-Path $tarball) { Remove-Item $tarball -Force }
tar -czf $tarball -C $out .
Write-Host "[OK] $tarball"
Get-ChildItem "$out\bin\aarch64", "$out\bin\x86_64" | Select-Object Name, Length
Write-Host ("samples: " + (Get-ChildItem "$out\samples" -Filter *.png).Count)
