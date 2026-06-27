cmake_minimum_required(VERSION 3.18)
project(warpkv LANGUAGES CXX CUDA)

# ============================================================================
# WarpKV v2.0.2 — Production Build Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture: CC 5.0 (Maxwell, MX130 target)
set(CMAKE_CUDA_ARCHITECTURES 50)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

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

# ============================================================================
# Phase 0: Foundation Targets
# ============================================================================

# Hardware validation utility (Phase 0)
add_executable(hardware_validation
    src/core/hardware_validation.cpp
)
target_link_libraries(hardware_validation CUDA::cudart)

# ============================================================================
# Unit Tests (Phase 0 onwards)
# ============================================================================

# Hash function unit tests (Phase 1)
add_executable(test_xxhash3
    tests/unit/test_xxhash3.cpp
    src/gpu/xxhash3.cu
)
target_link_libraries(test_xxhash3 GTest::gtest_main CUDA::cudart)
add_test(NAME XXHash3Tests COMMAND test_xxhash3)

# Bucket layout unit tests (Phase 2)
add_executable(test_bucket_layout
    tests/unit/test_bucket_layout.cpp
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_bucket_layout GTest::gtest_main CUDA::cudart)
add_test(NAME BucketLayoutTests COMMAND test_bucket_layout)

# Arena allocator unit tests (Phase 2)
add_executable(test_arena_allocator
    tests/unit/test_arena_allocator.cpp
    src/gpu/arena_allocator.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_arena_allocator GTest::gtest_main CUDA::cudart)
add_test(NAME ArenaAllocatorTests COMMAND test_arena_allocator)

# Warp lookup kernel unit tests (Phase 3)
add_executable(test_warp_lookup
    tests/unit/test_warp_lookup.cpp
    src/gpu/warp_lookup.cu
    src/gpu/xxhash3.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_warp_lookup GTest::gtest_main CUDA::cudart)
add_test(NAME WarpLookupTests COMMAND test_warp_lookup)

# ============================================================================
# Integration Tests (Phase 3 onwards)
# ============================================================================

# Cuckoo insertion kernel unit tests (Phase 4)
add_executable(test_cuckoo_insert
    tests/unit/test_cuckoo_insert.cpp
    src/gpu/cuckoo_insert.cu
    src/gpu/xxhash3.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_cuckoo_insert GTest::gtest_main CUDA::cudart)
add_test(NAME CuckooInsertTests COMMAND test_cuckoo_insert)

# Eviction chain correctness tests (Phase 4 Extended)
add_executable(test_eviction_chains
    tests/unit/test_eviction_chains.cpp
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_eviction_chains GTest::gtest_main CUDA::cudart)
add_test(NAME EvictionChainTests COMMAND test_eviction_chains)

# Rehashing kernel unit tests (Phase 5)
add_executable(test_rehash_kernel
    tests/unit/test_rehash_kernel.cpp
    src/gpu/rehash_kernel.cu
    src/gpu/xxhash3.cu
    src/gpu/bucket_cuckoo.h
)
target_link_libraries(test_rehash_kernel GTest::gtest_main CUDA::cudart)
add_test(NAME RehashKernelTests COMMAND test_rehash_kernel)

# Lookup correctness integration test (Phase 3)
add_executable(test_lookup_correctness
    tests/integration/test_lookup_correctness.cpp
)
target_link_libraries(test_lookup_correctness GTest::gtest_main CUDA::cudart)
add_test(NAME LookupCorrectnessTests COMMAND test_lookup_correctness)

# ============================================================================
# Benchmarks
# ============================================================================

# Hash throughput benchmark (Phase 1)
add_executable(bench_hash_throughput
    benchmarks/hash_throughput.cu
    src/gpu/xxhash3.cu
)
target_link_libraries(bench_hash_throughput CUDA::cudart)

# ============================================================================
# Main Application (Phase 7+)
# ============================================================================

# Example usage (built after Python binding is ready)
add_executable(warpkv_example
    src/main.cpp
    src/core/config.cpp
)
target_link_libraries(warpkv_example CUDA::cudart)

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "WarpKV v2.0.2 Build Configuration")
message(STATUS "  CUDA Architecture: CC 5.0 (Maxwell)")
message(STATUS "  C++ Standard: C++17")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Unit Tests: ENABLED")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  cmake --build . --target hardware_validation  # Phase 0")
message(STATUS "  cmake --build . --target test_xxhash3         # Phase 1")
message(STATUS "  cmake --build . --target test_bucket_layout   # Phase 2")
message(STATUS "  cmake --build . --target test_warp_lookup     # Phase 3")
message(STATUS "  ctest                                         # Run all tests")
message(STATUS "")
