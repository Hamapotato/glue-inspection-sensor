param(
    [string]$Dotnet = "dotnet",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root "src\GlueInspect.VmViewer\GlueInspect.VmViewer.csproj"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $root "artifacts\vm-viewer"
}

& $Dotnet publish $project `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -o $OutDir
if ($LASTEXITCODE -ne 0) {
    throw "Viewer publish failed with exit code $LASTEXITCODE"
}

Write-Host "Published to $OutDir"
