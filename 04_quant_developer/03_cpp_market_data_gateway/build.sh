#!/bin/bash

set -e

echo "Building HFT Feed Handler..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)

echo ""
echo "Build complete!"
echo ""
echo "Executables:"
echo "  ./build/feed_handler - Main feed handler"
echo "  ./build/benchmark    - Benchmark suite"
echo "  ./build/tests        - Unit tests"
echo ""
echo "To run:"
echo "  ./build/feed_handler --udp-port 9000 --tcp-port 9001"
echo "  ./build/benchmark"
echo "  ./build/tests"
