param(
    [string]$Dotnet = "dotnet"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root "tests\GlueInspect.VmViewer.SmokeTests\GlueInspect.VmViewer.SmokeTests.csproj"

& $Dotnet build $project -c Release
if ($LASTEXITCODE -ne 0) {
    throw "Viewer build failed with exit code $LASTEXITCODE"
}

& $Dotnet run --project $project -c Release --no-build
if ($LASTEXITCODE -ne 0) {
    throw "Viewer smoke tests failed with exit code $LASTEXITCODE"
}
