@echo off
cd /d "%~dp0"
if exist "target\release\glue-app.exe" goto launch
echo [INFO] binary not found, building first...
call "%~dp0build.bat"
if errorlevel 1 exit /b 1
:launch
"target\release\glue-app.exe" %*