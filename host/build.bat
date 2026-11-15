@echo off
cd /d "%~dp0"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" goto do_build
echo [ERROR] MSVC not found: %VCVARS%
echo Install Visual Studio 2022 Build Tools with C++ workload.
exit /b 1
:do_build
call "%VCVARS%" >nul
cargo build --release -p glue-app %*
if errorlevel 1 exit /b 1
echo [OK] built: target\release\glue-app.exe