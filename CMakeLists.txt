cmake_minimum_required(VERSION 3.18)
project(gpu-kvstore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/src)

# Hello CUDA test
add_executable(hello_cuda
    src/gpu/hello_cuda.cu
)
target_link_libraries(hello_cuda CUDA::cudart)

# Hash function tests
add_executable(test_hash
    tests/test_hash.cpp
)
target_link_libraries(test_hash CUDA::cudart)

# Main executable
add_executable(gpu-kvstore
    src/main.cpp
    src/core/config.cpp
    src/core/cuckoo.cpp
)
target_link_libraries(gpu-kvstore CUDA::cudart)
