@echo off
setlocal

set "VIEWER_SCRIPT=%~dp0scripts\run-viewer.ps1"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%VIEWER_SCRIPT%"
if errorlevel 1 (
    echo.
    echo Viewer startup failed. See the error above.
    pause
)

endlocal
