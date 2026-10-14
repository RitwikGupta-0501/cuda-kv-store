#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace warpkv {

template <typename ScalarType>
struct SlabPool {
    /// Pointer to a massive contiguous chunk of device memory [capacity * vector_dim]
    ScalarType* data;

    /// The length of each vector (e.g., 64, 128)
    uint32_t vector_dim;

    /// Maximum number of rows in the pool
    uint32_t max_capacity;

    /// Global atomic counter for bump allocation
    uint32_t* atomic_counter;

    __device__ inline ScalarType* get_vector(uint32_t index) const {
        return data + (index * vector_dim);
    }
};

} // namespace warpkv
