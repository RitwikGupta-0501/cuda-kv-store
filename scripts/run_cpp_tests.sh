#!/usr/bin/env bash
# ==============================================================================
# run_cpp_tests.sh — Build & Test Runner for WarpKV C++ Engine
# ==============================================================================
#
# Tests C++ Engine Deliverables:
#   1. Step 1.1: Standalone warpkv_device.cuh (test_warpkv_device_header)
#   2. Step 1.2: PTX device module generation (warpkv_device_ptx)
#   3. Step 1.3: Asynchronous non-blocking pipeline (test_async_pipeline)
#   4. Regression: Full suite (test_hash_function, test_bucket_layout,
#                  test_warp_lookup, test_cuckoo_insert, test_lookup_correctness,
#                  test_engine_concurrent)
#
# Usage:
#   bash scripts/run_cpp_tests.sh [options]
#
# Options:
#   --build-dir <dir>   Path to build directory (default: ./build)
#   --arch <sm_arch>    CUDA compute architecture number, e.g. 75, 80, 89 (default: auto/50)
#   --clean             Perform a clean build
#   --phase1-only       Run only the new C++ Engine test targets
#   --help              Show this message
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
CUDA_ARCH=""
CLEAN_BUILD=false
PHASE1_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --phase1-only)
            PHASE1_ONLY=true
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo " WarpKV C++ Engine Verification Suite"
echo "=============================================================================="
echo " Repository root: $REPO_ROOT"
echo " Build directory: $BUILD_DIR"

# 1. Dependency checks
if ! command -v cmake &> /dev/null; then
    echo "[-] ERROR: cmake is not installed or not in PATH."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "[-] ERROR: nvcc (CUDA Toolkit) is not found in PATH."
    echo "    Please install CUDA or run this on a GPU instance (e.g. Google Colab / AWS)."
    exit 1
fi

echo "[+] cmake: $(cmake --version | head -n1)"
echo "[+] nvcc:  $(nvcc --version | grep release | sed 's/.*release //; s/,.*//')"

if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "[*] Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 2. Configure
CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=Release")
if [ -n "$CUDA_ARCH" ]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
fi

echo "[*] Configuring CMake..."
cmake "${CMAKE_ARGS[@]}" "$REPO_ROOT"

# 3. Build Targets
echo "[*] Building targets..."
TARGETS=(
    "test_warpkv_device_header"
    "warpkv_device_ptx"
    "test_async_pipeline"
)

if [ "$PHASE1_ONLY" = false ]; then
    TARGETS+=(
        "test_hash_function"
        "test_bucket_layout"
        "test_warp_lookup"
        "test_cuckoo_insert"
        "test_lookup_correctness"
        "test_engine_concurrent"
    )
fi

for tgt in "${TARGETS[@]}"; do
    echo "    -> Building target: $tgt"
    cmake --build . --target "$tgt" -j"$(nproc)"
done

echo ""
echo "=============================================================================="
echo " Running Tests"
echo "=============================================================================="

PASSED=0
FAILED=0
FAILED_NAMES=()

run_executable() {
    local name="$1"
    local bin="$BUILD_DIR/$name"
    if [ ! -f "$bin" ]; then
        echo "[-] FAIL: $name binary not found at $bin"
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$name")
        return
    fi

    echo ">>> Running $name..."
    if "$bin"; then
        echo "[+] PASS: $name"
        PASSED=$((PASSED + 1))
    else
        echo "[-] FAIL: $name exited with error code $?"
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$name")
    fi
    echo ""
}

# Step 1.1 test: Standalone header acceptance
run_executable "test_warpkv_device_header"

# Step 1.2 check: PTX file verification
echo ">>> Checking warpkv_kernels.ptx..."
if [ -f "$BUILD_DIR/warpkv_kernels.ptx" ]; then
    echo "[+] Found warpkv_kernels.ptx (Size: $(wc -c < "$BUILD_DIR/warpkv_kernels.ptx") bytes)"
    # Check that the three C-linkage entry kernels exist inside the PTX
    for kern in "warpkv_lookup_kernel_c" "warpkv_insert_kernel_c" "warpkv_delete_kernel_c"; do
        if grep -q "\.entry $kern" "$BUILD_DIR/warpkv_kernels.ptx"; then
            echo "    -> Verified PTX entry point: $kern"
        else
            echo "    [-] ERROR: Missing entry point $kern in PTX!"
            FAILED=$((FAILED + 1))
            FAILED_NAMES+=("ptx_entry_$kern")
        fi
    done
    PASSED=$((PASSED + 1))
else
    echo "[-] FAIL: $BUILD_DIR/warpkv_kernels.ptx was not generated"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("warpkv_device_ptx")
fi
echo ""

# Step 1.3 test: Async pipeline
run_executable "test_async_pipeline"

# Additional tests if full suite requested
if [ "$PHASE1_ONLY" = false ]; then
    run_executable "test_hash_function"
    run_executable "test_bucket_layout"
    run_executable "test_warp_lookup"
    run_executable "test_cuckoo_insert"
    run_executable "test_lookup_correctness"
    run_executable "test_engine_concurrent"
fi

echo "=============================================================================="
echo " Test Summary"
echo "=============================================================================="
echo " Passed: $PASSED"
echo " Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo " Failed targets:"
    for fn in "${FAILED_NAMES[@]}"; do
        echo "   - $fn"
    done
    exit 1
else
    echo " All C++ Engine deliverables and regression checks verified! ✓"
    exit 0
fi
