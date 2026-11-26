param(
    [string]$VisionMasterRoot = "D:\VM\VisionMaster4.4.0",
    [string]$Csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$shellModule = Join-Path $VisionMasterRoot "Applications\Module(sp)\x64\Logic\ShellModule"
$senderSource = Join-Path $scriptRoot "VmFrameSender.cs"
$testSource = Join-Path $scriptRoot "VmFrameSenderSmokeTest.cs"
$scriptMethods = Join-Path $shellModule "Script.Methods.dll"
$scriptAlgorithm = Join-Path $shellModule "Script.Algorithm.dll"
$scriptSupport = Join-Path $shellModule "Script.Support.dll"

foreach ($required in @($Csc, $senderSource, $testSource, $scriptMethods, $scriptAlgorithm, $scriptSupport))
{
    if (-not (Test-Path -LiteralPath $required))
    {
        throw "Required file not found: $required"
    }
}

$baseTemp = [System.IO.Path]::GetFullPath($env:TEMP)
$buildDirectory = Join-Path $baseTemp ("vm-frame-sender-test-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $buildDirectory | Out-Null

try
{
    $testExecutable = Join-Path $buildDirectory "VmFrameSenderSmokeTest.exe"
    $compilerArguments = @(
        "/nologo",
        "/target:exe",
        "/langversion:5",
        "/out:$testExecutable",
        "/r:$scriptMethods",
        "/r:$scriptAlgorithm",
        "/r:$scriptSupport",
        "/r:System.dll",
        "/r:System.Core.dll",
        "/r:System.Windows.Forms.dll",
        $senderSource,
        $testSource
    )

    & $Csc $compilerArguments
    if ($LASTEXITCODE -ne 0)
    {
        throw "VM frame sender compilation failed with exit code $LASTEXITCODE"
    }

    Copy-Item -LiteralPath $scriptMethods -Destination $buildDirectory
    Copy-Item -LiteralPath $scriptAlgorithm -Destination $buildDirectory
    Copy-Item -LiteralPath $scriptSupport -Destination $buildDirectory

    & $testExecutable
    if ($LASTEXITCODE -ne 0)
    {
        throw "VM frame sender smoke test failed with exit code $LASTEXITCODE"
    }
}
finally
{
    $resolvedBuild = [System.IO.Path]::GetFullPath($buildDirectory)
    if ($resolvedBuild.StartsWith($baseTemp + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedBuild).StartsWith("vm-frame-sender-test-", [StringComparison]::Ordinal))
    {
        Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
    }
}
