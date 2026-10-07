cmake_minimum_required(VERSION 3.18)
project(warpkv LANGUAGES CXX CUDA)

# ============================================================================
# WarpKV v2.0.2 — Production Build Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture: CC 7.5 (Turing, Tesla T4 target)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/warpkv)

# ============================================================================
# Google Test Framework (for unit tests)
# ============================================================================
enable_testing()
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Hash function unit tests (Phase 1)
add_executable(test_hash_function
    tests/unit/test_hash_function.cpp
    src/gpu/hash.cu
)
target_link_libraries(test_hash_function GTest::gtest_main CUDA::cudart)
add_test(NAME HashFunctionTests COMMAND test_hash_function)

# Bucket layout unit tests (Phase 2)
add_executable(test_bucket_layout
    tests/unit/test_bucket_layout.cpp
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_bucket_layout GTest::gtest_main CUDA::cudart)
add_test(NAME BucketLayoutTests COMMAND test_bucket_layout)

# Warp lookup kernel unit tests (Phase 3)
add_executable(test_warp_lookup
    tests/unit/test_warp_lookup.cu
    src/gpu/warp_lookup.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_warp_lookup GTest::gtest_main CUDA::cudart)
add_test(NAME WarpLookupTests COMMAND test_warp_lookup)

# ============================================================================
# Integration Tests (Phase 3 onwards)
# ============================================================================

# Cuckoo insertion kernel unit tests (Phase 4)
add_executable(test_cuckoo_insert
    tests/unit/test_cuckoo_insert.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_cuckoo_insert GTest::gtest_main CUDA::cudart)
add_test(NAME CuckooInsertTests COMMAND test_cuckoo_insert)

# Debug kernel tests
add_executable(test_debug_kernels
    tests/unit/test_debug_kernels.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_debug_kernels GTest::gtest_main CUDA::cudart)
add_test(NAME DebugKernels COMMAND test_debug_kernels)

# Eviction chain correctness tests (Phase 4 Extended)
add_executable(test_eviction_chains
    tests/unit/test_eviction_chains.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_eviction_chains GTest::gtest_main CUDA::cudart)
add_test(NAME EvictionChainTests COMMAND test_eviction_chains)

# Rehashing kernel unit tests (Phase 5)
add_executable(test_rehash_kernel
    tests/unit/test_rehash_kernel.cu
    src/gpu/rehash_kernel.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_rehash_kernel GTest::gtest_main CUDA::cudart)
add_test(NAME RehashKernelTests COMMAND test_rehash_kernel)

# Lookup and Delete correctness integration test (Phase 3 & 5)
add_executable(test_lookup_correctness
    tests/integration/test_lookup_correctness.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(test_lookup_correctness GTest::gtest_main CUDA::cudart)
add_test(NAME LookupCorrectnessTests COMMAND test_lookup_correctness)

# Engine Concurrent Pipeline Test (Phase 6)
add_executable(test_engine_concurrent
    tests/unit/test_engine_concurrent.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(test_engine_concurrent GTest::gtest_main CUDA::cudart)
add_test(NAME EngineConcurrentTests COMMAND test_engine_concurrent)

# ============================================================================
# Phase 1 Tests
# ============================================================================

# Step 1.1: warpkv_device.cuh self-contained header acceptance test
# This test includes ONLY warpkv_device.cuh — no other project headers.
# Verifies the header is truly standalone and all device APIs are accessible.
add_executable(test_warpkv_device_header
    tests/unit/test_warpkv_device_header.cu
)
target_link_libraries(test_warpkv_device_header CUDA::cudart)
# No GTest — this is a standalone self-validating binary.
add_test(NAME DeviceHeaderSelfContained COMMAND test_warpkv_device_header)

# Step 1.3: Async pipeline acceptance test
add_executable(test_async_pipeline
    tests/unit/test_async_pipeline.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(test_async_pipeline GTest::gtest_main CUDA::cudart)
add_test(NAME AsyncPipelineTests COMMAND test_async_pipeline)


# ============================================================================
# Benchmarks
# ============================================================================

# Hash throughput benchmark
add_executable(bench_hash_throughput
    benchmarks/hash_throughput.cu
    src/gpu/hash.cu
)
target_link_libraries(bench_hash_throughput CUDA::cudart)

# Phase 9: Engine Benchmark
add_executable(engine_benchmark
    benchmarks/engine_benchmark.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(engine_benchmark CUDA::cudart)

# Phase 9: YCSB Benchmark
add_executable(ycsb_benchmark
    benchmarks/ycsb_benchmark.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(ycsb_benchmark CUDA::cudart)

# Phase 9: Load Factor Benchmark
add_executable(load_factor_benchmark
    benchmarks/load_factor_benchmark.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(load_factor_benchmark CUDA::cudart)

# Synthetic load test — GPU version (real kernel execution)
add_executable(gpu_synthetic_load_test
    benchmarks/gpu_synthetic_load_test.cu
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(gpu_synthetic_load_test CUDA::cudart)

# ============================================================================
# Python Bindings (Phase 7)
# ============================================================================

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, downloading...")
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
    )
    FetchContent_MakeAvailable(pybind11)
endif()
pybind11_add_module(warpkv
    src/python/warpkv_binding.cpp
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
target_link_libraries(warpkv PRIVATE CUDA::cudart)

# ============================================================================
# Phase 1: PTX Device Module (extern "C" __global__ entry points)
# ============================================================================
# Builds a .ptx file containing the three C-linkage kernel entry points.
# Loadable at runtime by Numba (cuModuleLoad), CuPy (RawModule), or
# any CUDA Driver API consumer (Rust-CUDA, ctypes, etc.)
#
# Usage: cmake --build . --target warpkv_device_ptx
#        Output: warpkv_kernels.ptx in the build directory
# ============================================================================

# Use a custom command since CMake doesn't natively support --ptx output
find_program(NVCC_EXECUTABLE nvcc
    HINTS ${CUDAToolkit_BIN_DIR}
    REQUIRED
)

add_custom_target(warpkv_device_ptx
    COMMAND ${NVCC_EXECUTABLE}
            -arch=sm_${CMAKE_CUDA_ARCHITECTURES}
            --ptx
            -std=c++14
            --expt-extended-lambda
            -I ${CMAKE_SOURCE_DIR}/include
            -I ${CMAKE_SOURCE_DIR}/include/warpkv
            -o ${CMAKE_CURRENT_BINARY_DIR}/warpkv_kernels.ptx
            ${CMAKE_SOURCE_DIR}/src/gpu/warpkv_entry_kernels.cu
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/warpkv_kernels.ptx
    COMMENT "Building WarpKV PTX device module -> warpkv_kernels.ptx"
    VERBATIM
)

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "WarpKV v2.0.2 Build Configuration")
message(STATUS "  CUDA Architecture: CC 7.5 (Turing, Tesla T4)")
message(STATUS "  C++ Standard: C++17")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Unit Tests: ENABLED")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  cmake --build . --target test_hash_function     # Hash unit tests")
message(STATUS "  cmake --build . --target test_bucket_layout     # Bucket layout tests")
message(STATUS "  cmake --build . --target test_warp_lookup       # Warp lookup tests")
message(STATUS "  cmake --build . --target warpkv_device_ptx      # Phase 1: PTX module")
message(STATUS "  ctest                                            # Run all tests")
message(STATUS "")
message(STATUS "Phase 1 (Device API):")
message(STATUS "  Device header: include/warpkv/warpkv_device.cuh")
message(STATUS "  PTX entry:     src/gpu/warpkv_entry_kernels.cu")
message(STATUS "  PTX output:    <build>/warpkv_kernels.ptx")
message(STATUS "")


# ============================================================================
# Engine Library
# ============================================================================
add_library(warpkv_engine STATIC
    src/engine/warpkv_engine.cu
    src/gpu/cuckoo_insert.cu
    src/gpu/cuckoo_delete.cu
    src/gpu/warp_lookup.cu
    src/gpu/rehash_kernel.cu
    src/gpu/hash.cu
)
set_target_properties(warpkv_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(warpkv_engine PUBLIC CUDA::cudart)

# PyTorch Extension (Phase 4)
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found PyTorch: ${TORCH_VERSION}, building warpkv_torch")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    
    add_library(warpkv_torch MODULE src/python/warpkv_torch.cpp)
    target_link_libraries(warpkv_torch PRIVATE warpkv_engine ${TORCH_LIBRARIES} pybind11::module)
    
    # Configure extension name macro for Pybind11
    target_compile_definitions(warpkv_torch PRIVATE TORCH_EXTENSION_NAME=warpkv_torch)
    set_target_properties(warpkv_torch PROPERTIES PREFIX "" SUFFIX ".so")
else()
    message(WARNING "PyTorch not found, skipping warpkv_torch build.")
endif()
