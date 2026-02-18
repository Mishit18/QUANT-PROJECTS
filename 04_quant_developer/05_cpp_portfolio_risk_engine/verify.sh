#!/bin/bash

echo "=== Portfolio Risk Engine - Verification Script ==="
echo ""

# Check for required files
echo "Checking project structure..."

required_files=(
    "README.md"
    "CMakeLists.txt"
    "src/main.cpp"
    "src/core/rng.h"
    "src/core/path_generator.h"
    "src/engine/thread_pool.h"
    "src/pricing/option.h"
    "src/portfolio/position.h"
    "src/risk/var_calculator.h"
    "tests/test_rng.cpp"
    "benchmarks/benchmark.cpp"
    "docs/theory.md"
    "docs/architecture.md"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo " Missing: $file"
        missing_files=$((missing_files + 1))
    else
        echo " Found: $file"
    fi
done

echo ""
if [ $missing_files -eq 0 ]; then
    echo " All required files present!"
else
    echo " Missing $missing_files file(s)"
    exit 1
fi

# Check for C++ compiler
echo ""
echo "Checking for C++ compiler..."
if command -v g++ &> /dev/null; then
    echo " g++ found: $(g++ --version | head -n1)"
elif command -v clang++ &> /dev/null; then
    echo " clang++ found: $(clang++ --version | head -n1)"
else
    echo " No C++ compiler found"
    exit 1
fi

# Check for CMake
echo ""
echo "Checking for CMake..."
if command -v cmake &> /dev/null; then
    echo " CMake found: $(cmake --version | head -n1)"
else
    echo " CMake not found"
    exit 1
fi

# Count source files
echo ""
echo "Code statistics:"
h_files=$(find src -name "*.h" | wc -l)
cpp_files=$(find src -name "*.cpp" | wc -l)
test_files=$(find tests -name "*.cpp" | wc -l)
doc_files=$(find docs -name "*.md" | wc -l)

echo "  Header files: $h_files"
echo "  Source files: $cpp_files"
echo "  Test files: $test_files"
echo "  Documentation files: $doc_files"

# Check documentation
echo ""
echo "Documentation completeness:"
docs=(
    "docs/PROJECT_OVERVIEW.md"
    "docs/QUICKSTART.md"
    "docs/theory.md"
    "docs/architecture.md"
    "docs/performance.md"
    "docs/api_reference.md"
    "docs/usage_examples.md"
    "docs/design_decisions.md"
    "docs/FAQ.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        lines=$(wc -l < "$doc")
        echo "   $doc ($lines lines)"
    else
        echo "   $doc (missing)"
    fi
done

echo ""
echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "  1. Run: ./build.sh Release"
echo "  2. Run: ./build/risk_engine"
echo "  3. Run: ./build/benchmark"
echo "  4. Run: cd build && ctest"
