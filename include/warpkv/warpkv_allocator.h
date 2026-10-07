#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace warpkv {

/// Abstract base class for GPU memory allocation.
/// Used to decouple the engine's internal table allocations from bare cudaMalloc,
/// preventing conflicts with PyTorch's caching allocator.
struct WarpKVAllocator {
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual ~WarpKVAllocator() = default;
};

/// Default implementation using bare cudaMalloc and cudaFree.
class DefaultCudaAllocator : public WarpKVAllocator {
public:
    void* allocate(size_t bytes) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

} // namespace warpkv
