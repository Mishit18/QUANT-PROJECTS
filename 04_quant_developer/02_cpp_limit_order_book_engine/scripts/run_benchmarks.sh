#!/bin/bash

set -e

BUILD_DIR="build"
BENCHMARK_BIN="$BUILD_DIR/lob_benchmark"
OUTPUT_DIR="benchmarks"

mkdir -p "$OUTPUT_DIR"

echo "Building..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..

echo ""
echo "Running benchmarks..."

echo ""
echo "100K orders"
"$BENCHMARK_BIN" 100000 "$OUTPUT_DIR/results_100k.csv"

echo ""
echo "1M orders"
"$BENCHMARK_BIN" 1000000 "$OUTPUT_DIR/results_1m.csv"

echo ""
echo "5M orders"
"$BENCHMARK_BIN" 5000000 "$OUTPUT_DIR/results_5m.csv"

echo ""
echo "Results written to $OUTPUT_DIR/"
