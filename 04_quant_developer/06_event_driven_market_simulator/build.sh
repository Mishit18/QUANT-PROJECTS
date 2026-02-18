#!/bin/bash

# Build script for Event-Driven Market Simulator

set -e

echo "Event-Driven Market Simulator - Build Script"
echo "============================================="
echo ""

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete!"
echo ""
echo "Executables:"
echo "  ./build/market_simulator     - Main simulator"
echo "  ./build/latency_benchmark    - Latency benchmarks"
echo "  ./build/run_tests            - Unit tests (if available)"
echo ""
