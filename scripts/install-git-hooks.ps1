# Install repository Git hooks from .githooks
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install-git-hooks.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git config --local core.hooksPath .githooks
if ($LASTEXITCODE -ne 0) { throw "failed to set core.hooksPath" }

# Ensure hooks are executable on Linux/macOS clones (skip if not yet tracked)
git ls-files --error-unmatch .githooks/pre-commit *> $null
if ($LASTEXITCODE -eq 0) {
    git update-index --chmod=+x .githooks/pre-commit .githooks/commit-msg
}

Write-Host "[OK] core.hooksPath = .githooks"
Write-Host "     pre-commit: blocks build artifacts, models, PDFs, files >5MB"
Write-Host "     commit-msg: English subject, capital first letter, ASCII, max 72 chars"
