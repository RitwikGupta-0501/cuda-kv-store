#!/bin/bash
# WarpKV v2.0.2 — GPU Test Harness for Google Colab
#
# Usage:
#   bash scripts/run_tests_colab.sh              # Run all tests
#   bash scripts/run_tests_colab.sh --phase 4   # Run Phase 4 tests only
#   bash scripts/run_tests_colab.sh --build-only # Only build, don't test
#
# This script:
#   1. Checks CUDA/cmake availability
#   2. Builds the project
#   3. Runs selected tests with summary report

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
PHASE="${1:-all}"
BUILD_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================================================"
echo "WarpKV v2.0.2 — GPU Test Harness"
echo "============================================================================"
echo "Repository: $REPO_ROOT"
echo "Build Directory: $BUILD_DIR"
echo "Phase: $PHASE"
echo ""

# Check dependencies
echo "[1/4] Checking dependencies..."
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Install with: apt-get install cmake"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA toolkit not found (nvcc not in PATH)"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}')
echo "  ✓ cmake $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ CUDA $CUDA_VERSION"
echo ""

# Build
echo "[2/4] Building WarpKV..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=Release "$REPO_ROOT" > /dev/null 2>&1 || {
    echo "ERROR: CMake configuration failed"
    cmake -DCMAKE_BUILD_TYPE=Release "$REPO_ROOT"
    exit 1
}

cmake --build . -j$(nproc) > /dev/null 2>&1 || {
    echo "ERROR: Build failed"
    cmake --build . -j$(nproc)
    exit 1
}

echo "  ✓ Build successful"
echo ""

if [ "$BUILD_ONLY" = true ]; then
    echo "Build-only mode. Exiting."
    exit 0
fi

# Test mapping
declare -A TESTS
TESTS[1]="test_xxhash3"
TESTS[2]="test_bucket_layout test_arena_allocator"
TESTS[3]="test_warp_lookup"
TESTS[4]="test_cuckoo_insert test_eviction_chains"
TESTS[5]="test_rehash_kernel"

# Run tests
echo "[3/4] Running tests..."
echo ""

TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=()

run_test() {
    local test_name=$1
    local test_path="$BUILD_DIR/$test_name"

    if [ ! -f "$test_path" ]; then
        echo "  ⚠ SKIP  $test_name (not built)"
        return
    fi

    TOTAL=$((TOTAL + 1))

    if output=$("$test_path" 2>&1); then
        PASSED=$((PASSED + 1))
        # Count actual test count from output
        test_count=$(echo "$output" | grep -o "\[==========\] [0-9]* test" | head -1 | awk '{print $2}')
        if [ -n "$test_count" ]; then
            echo "  ✓ PASS  $test_name ($test_count tests)"
        else
            echo "  ✓ PASS  $test_name"
        fi
    else
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
        echo "  ✗ FAIL  $test_name"
        # Print last 20 lines of output for debugging
        echo "    Output (last 20 lines):"
        echo "$output" | tail -20 | sed 's/^/      /'
    fi
}

case "$PHASE" in
    all)
        for phase in 1 2 3 4 5; do
            for test in ${TESTS[$phase]}; do
                run_test "$test"
            done
        done
        ;;
    1|2|3|4|5)
        for test in ${TESTS[$PHASE]}; do
            run_test "$test"
        done
        ;;
    *)
        echo "ERROR: Unknown phase '$PHASE'. Use: all, 1, 2, 3, 4, 5"
        exit 1
        ;;
esac

echo ""
echo "[4/4] Test Summary"
echo "============================================================================"
echo "Total Tests Run: $TOTAL"
echo "  ✓ Passed: $PASSED"
echo "  ✗ Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "Run individually to debug:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  $BUILD_DIR/$test"
    done
    exit 1
else
    echo "All tests passed! ✓"
    exit 0
fi
