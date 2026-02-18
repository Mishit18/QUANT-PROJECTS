@echo off
REM Build script for Event-Driven Market Simulator (Windows)

echo Event-Driven Market Simulator - Build Script
echo =============================================
echo.

REM Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

REM Build
echo Building...
cmake --build . --config Release

echo.
echo Build complete!
echo.
echo Executables:
echo   build\Release\market_simulator.exe     - Main simulator
echo   build\Release\latency_benchmark.exe    - Latency benchmarks
echo   build\Release\run_tests.exe            - Unit tests (if available)
echo.

cd ..
