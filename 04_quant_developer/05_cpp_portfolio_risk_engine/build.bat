@echo off
setlocal

echo === Portfolio Risk Engine Build Script ===

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo Build type: %BUILD_TYPE%

if exist build (
    echo Cleaning existing build directory...
    rmdir /s /q build
)

mkdir build
cd build

echo Running CMake...
cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..

echo Building...
cmake --build . --config %BUILD_TYPE% -j

echo.
echo === Build Complete ===
echo.
echo Executables:
echo   .\build\%BUILD_TYPE%\risk_engine.exe       - Main simulation
echo   .\build\%BUILD_TYPE%\benchmark.exe         - Performance benchmarks
echo   .\build\%BUILD_TYPE%\test_rng.exe          - RNG tests
echo   .\build\%BUILD_TYPE%\test_pricing.exe      - Pricing tests
echo   .\build\%BUILD_TYPE%\test_portfolio.exe    - Portfolio tests
echo   .\build\%BUILD_TYPE%\test_thread_pool.exe  - Thread pool tests
echo.
echo Run tests with: cd build ^&^& ctest -C %BUILD_TYPE%

endlocal
