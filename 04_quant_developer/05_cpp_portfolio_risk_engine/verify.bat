@echo off
setlocal enabledelayedexpansion

echo === Portfolio Risk Engine - Verification Script ===
echo.

echo Checking project structure...
set missing=0

set files=README.md CMakeLists.txt src\main.cpp src\core\rng.h src\core\path_generator.h src\engine\thread_pool.h src\pricing\option.h src\portfolio\position.h src\risk\var_calculator.h tests\test_rng.cpp benchmarks\benchmark.cpp docs\theory.md docs\architecture.md

for %%f in (%files%) do (
    if exist "%%f" (
        echo [32m✓[0m Found: %%f
    ) else (
        echo [31m✗[0m Missing: %%f
        set /a missing+=1
    )
)

echo.
if %missing%==0 (
    echo [32m✓ All required files present![0m
) else (
    echo [31m✗ Missing %missing% file(s)[0m
    exit /b 1
)

echo.
echo Checking for build tools...

where cmake >nul 2>&1
if %errorlevel%==0 (
    echo [32m✓[0m CMake found
    cmake --version | findstr /C:"version"
) else (
    echo [31m✗[0m CMake not found
)

where cl >nul 2>&1
if %errorlevel%==0 (
    echo [32m✓[0m MSVC compiler found
) else (
    where g++ >nul 2>&1
    if !errorlevel!==0 (
        echo [32m✓[0m GCC compiler found
    ) else (
        echo [31m✗[0m No C++ compiler found
    )
)

echo.
echo Code statistics:
for /f %%i in ('dir /s /b src\*.h ^| find /c /v ""') do set h_count=%%i
for /f %%i in ('dir /s /b src\*.cpp ^| find /c /v ""') do set cpp_count=%%i
for /f %%i in ('dir /s /b tests\*.cpp ^| find /c /v ""') do set test_count=%%i
for /f %%i in ('dir /s /b docs\*.md ^| find /c /v ""') do set doc_count=%%i

echo   Header files: %h_count%
echo   Source files: %cpp_count%
echo   Test files: %test_count%
echo   Documentation files: %doc_count%

echo.
echo === Verification Complete ===
echo.
echo Next steps:
echo   1. Run: build.bat Release
echo   2. Run: build\Release\risk_engine.exe
echo   3. Run: build\Release\benchmark.exe
echo   4. Run: cd build ^&^& ctest -C Release

endlocal
