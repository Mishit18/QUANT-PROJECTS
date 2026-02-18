@echo off
echo Building execution engine...

if not exist "build" mkdir build

g++ -std=c++17 -O3 -Wall -Wextra -I./include ^
    src/order_book/order_book.cpp ^
    src/venue/venue.cpp ^
    src/cost_model/cost_model.cpp ^
    src/queue_model/queue_model.cpp ^
    src/execution/twap_strategy.cpp ^
    src/execution/vwap_strategy.cpp ^
    src/execution/pov_strategy.cpp ^
    src/execution/aggressive_strategy.cpp ^
    src/sor/smart_order_router.cpp ^
    src/execution/execution_engine.cpp ^
    src/analytics/analytics.cpp ^
    main.cpp ^
    -o build/execution_engine.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Executable: build/execution_engine.exe
) else (
    echo Build failed!
    exit /b 1
)
