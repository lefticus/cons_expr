#!/bin/bash

# Debug Coverage Build and Test Script
# Usage: ./build_and_test_coverage.sh [build_folder]
# Default build folder: build-coverage-debug

set -e  # Exit on any error

# Configuration
BUILD_DIR="${1:-build-coverage-debug}"
SOURCE_DIR="$(pwd)"

echo "=== Debug Coverage Build and Test Script ==="
echo "Source directory: $SOURCE_DIR"
echo "Build directory: $BUILD_DIR"
echo

# Step 1: Configure with CMake using Ninja and debug coverage (if needed)
if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/build.ninja" ]; then
    echo "Step 1: Build directory exists and is configured, skipping configuration..."
    echo "ℹ To force reconfiguration, delete $BUILD_DIR and run again"
else
    echo "Step 1: Configuring with CMake (Debug + Coverage + Ninja)..."
    cmake -S . -B "$BUILD_DIR" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -Dcons_expr_ENABLE_CLANG_TIDY:Bool=ON \
        -Dcons_expr_ENABLE_IPO:Bool=ON \
        -Dcons_expr_ENABLE_HARDENING:Bool=ON \
        -Dcons_expr_ENABLE_COVERAGE:Bool=ON
    echo "✓ Configuration complete"
fi
echo

# Step 3: Verify gcov is available for coverage
echo "Step 3: Verifying coverage toolchain..."
if ! command -v gcovr >/dev/null 2>&1; then
    echo "❌ ERROR: gcovr not found - coverage cannot be generated"
    echo "Please install gcovr (usually part of gcc/build-essential package)"
    exit 1
fi
echo "✓ gcovr found - coverage generation enabled"
echo

# Step 4: Build all targets except constexpr_tests in parallel
echo "Step 4: Building all targets except constexpr_tests in parallel..."
cmake --build "$BUILD_DIR" --target relaxed_constexpr_tests tests cons_expr ccons_expr speed_test

echo "✓ All targets built successfully (except constexpr_tests)"
echo

# Step 5: Execute relaxed_constexpr_tests
echo "Step 5: Running relaxed_constexpr_tests..."
./$BUILD_DIR/test/relaxed_constexpr_tests

echo "✓ relaxed_constexpr_tests passed"
echo

# Step 6: Execute tests
echo "Step 6: Running tests..."
./$BUILD_DIR/test/tests

echo "✓ tests passed"
echo


# Step 2: Clean up any existing coverage files to avoid stamp mismatches
echo "Step 2: Cleaning up existing coverage files (gcda and gcno)..."
cd "$BUILD_DIR"
find . -name "*.gcda" -delete
cd -
echo "✓ Coverage files cleaned (prevents gcov stamp mismatch)"
echo


# Step 7: Build constexpr_tests (compile-time tests)
echo "Step 7: Building constexpr_tests (compile-time validation)..."
cmake --build $BUILD_DIR --target constexpr_tests

echo "✓ constexpr_tests compiled successfully (all static assertions passed)"
echo



# Step 8: Final build to catch any remaining tools
echo "Step 8: Final build to catch any remaining tools..."
cmake --build $BUILD_DIR
echo "✓ Final build completed"
echo

# Step 10: Run all tests with CTest in parallel
echo "Step 10: Running all tests with CTest in parallel..."
cd $BUILD_DIR
ctest -C Debug -j
cd -

echo "✓ All CTest tests completed"
echo

# Step 11: Generate comprehensive coverage report with decision/call coverage and multiple output formats
echo "Step 11: Generating comprehensive coverage information..."
cd $BUILD_DIR
gcovr --filter ../include/cons_expr/cons_expr.hpp --exclude-directories _deps --gcov-ignore-errors=no_working_dir_found . --html --html-details --html-title "cons_expr Coverage Report" -o coverage_report.html -j 4 --decisions --calls --json=coverage_report.json --txt-summary
# Note: gcovr automatically handles .gcov file cleanup and generates multiple output formats simultaneously
cd -
echo "✓ Comprehensive coverage reports generated in build directory:"
echo "  - Text summary: displayed above"
echo "  - HTML detailed report: $BUILD_DIR/coverage_report.html"
echo "  - JSON data report: $BUILD_DIR/coverage_report.json"

echo
echo "=== All Steps Completed Successfully! ==="
echo "Build directory: $BUILD_DIR"
echo "Tests passed: relaxed_constexpr_tests, tests, constexpr_tests, CTest suite"
if [ -d "coverage_html" ]; then
    echo "Coverage report: $BUILD_DIR/coverage_html/index.html"
fi
echo
