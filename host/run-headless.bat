@echo off
cd /d "%~dp0"
"target\release\glue-app.exe" --headless ..\outputs\rust_samples --mm-per-px 0.0833
pause