#!/bin/bash

set -e

echo "=== Portfolio Risk Engine Build Script ==="

BUILD_TYPE=${1:-Release}
NUM_JOBS=${2:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

echo "Build type: $BUILD_TYPE"
echo "Parallel jobs: $NUM_JOBS"

if [ -d "build" ]; then
    echo "Cleaning existing build directory..."
    rm -rf build
fi

mkdir -p build
cd build

echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

echo "Building..."
make -j$NUM_JOBS

echo ""
echo "=== Build Complete ==="
echo ""
echo "Executables:"
echo "  ./build/risk_engine       - Main simulation"
echo "  ./build/benchmark         - Performance benchmarks"
echo "  ./build/test_rng          - RNG tests"
echo "  ./build/test_pricing      - Pricing tests"
echo "  ./build/test_portfolio    - Portfolio tests"
echo "  ./build/test_thread_pool  - Thread pool tests"
echo ""
echo "Run tests with: cd build && ctest"
