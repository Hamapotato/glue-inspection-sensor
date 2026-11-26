param(
    [string]$Dotnet = "D:\WORK\.dotnet-sdk-10.0.402\dotnet.exe"
)

$ErrorActionPreference = "Stop"

$viewerRoot = Split-Path -Parent $PSScriptRoot
$project = Join-Path $viewerRoot "src\GlueInspect.VmViewer\GlueInspect.VmViewer.csproj"
$viewerDirectory = Join-Path $viewerRoot "src\GlueInspect.VmViewer"
$viewerExe = Join-Path $viewerDirectory "bin\Release\net10.0-windows\GlueInspect.VmViewer.exe"

if (-not (Test-Path -LiteralPath $Dotnet))
{
    $dotnetCommand = Get-Command "dotnet.exe" -ErrorAction SilentlyContinue
    if ($null -eq $dotnetCommand)
    {
        throw "找不到 .NET 10。请安装 .NET 10 SDK，或传入 -Dotnet 指向 dotnet.exe。"
    }
    $Dotnet = $dotnetCommand.Source
}

if (-not (Test-Path -LiteralPath $project))
{
    throw "找不到查看器项目：$project"
}

$existing = Get-Process "GlueInspect.VmViewer" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $existing)
{
    Write-Host "VM 结果查看器已经在运行，PID=$($existing.Id)。"
    exit 0
}

Write-Host "正在构建 VM 结果查看器..."
& $Dotnet build $project -c Release
if ($LASTEXITCODE -ne 0)
{
    throw "查看器 Release 构建失败，退出码 $LASTEXITCODE"
}

Write-Host "正在启动 VM 结果查看器..."
$launcher = Start-Process `
    -FilePath $Dotnet `
    -ArgumentList @(
        "run",
        "--project",
        $project,
        "-c",
        "Release",
        "--no-build"
    ) `
    -WorkingDirectory $viewerDirectory `
    -WindowStyle Hidden `
    -PassThru

for ($attempt = 0; $attempt -lt 20; $attempt++)
{
    Start-Sleep -Milliseconds 250
    $viewer = Get-Process "GlueInspect.VmViewer" -ErrorAction SilentlyContinue |
        Where-Object {
            try
            {
                $_.Path -eq $viewerExe
            }
            catch
            {
                $false
            }
        } |
        Select-Object -First 1
    if ($null -ne $viewer)
    {
        Write-Host "VM 结果查看器已启动，PID=$($viewer.Id)。"
        exit 0
    }
}

if ($launcher.HasExited)
{
    throw "查看器启动失败，启动器退出码 $($launcher.ExitCode)"
}

throw "查看器启动超时，请检查 $viewerExe"
