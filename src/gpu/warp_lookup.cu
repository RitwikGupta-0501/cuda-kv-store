#include "warp_lookup.h"
#include <stdexcept>
#include <cstring>

namespace warpkv {

// Host-side wrapper: allocate temporary buffers and launch kernel
void warp_lookup_batch(
    const BucketTable* d_table,
    const LookupBatch& batch,
    cudaStream_t stream) {

    if (batch.num_keys == 0) {
        return;
    }

    // Allocate device-side buffers for this batch
    uint32_t* d_keys = nullptr;
    uint32_t* d_values = nullptr;
    uint32_t* d_found = nullptr;

    size_t keys_size = batch.num_keys * sizeof(uint32_t);
    size_t values_size = batch.num_keys * sizeof(uint32_t);
    size_t found_size = batch.num_keys * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&d_keys, keys_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc d_keys failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_values, values_size);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        throw std::runtime_error(std::string("cudaMalloc d_values failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_found, found_size);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        throw std::runtime_error(std::string("cudaMalloc d_found failed: ") +
                               cudaGetErrorString(err));
    }

    // Copy keys to device
    err = cudaMemcpyAsync(d_keys, batch.h_keys, keys_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_found);
        throw std::runtime_error(std::string("cudaMemcpyAsync keys failed: ") +
                               cudaGetErrorString(err));
    }

    // Launch kernel: one warp (32 threads) per key
    // Grid: (num_keys / (threads_per_block / 32)) blocks
    uint32_t threads_per_block = 256;  // 8 warps per block
    uint32_t keys_per_block = threads_per_block / 32;
    uint32_t num_blocks = (batch.num_keys + keys_per_block - 1) / keys_per_block;

    warp_lookup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_table,
        d_keys,
        d_values,
        d_found,
        batch.num_keys
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_found);
        throw std::runtime_error(std::string("Kernel launch failed: ") +
                               cudaGetErrorString(err));
    }

    // Copy results back to host
    err = cudaMemcpyAsync(batch.h_values, d_values, values_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_found);
        throw std::runtime_error(std::string("cudaMemcpyAsync values failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(batch.h_found, d_found, found_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_found);
        throw std::runtime_error(std::string("cudaMemcpyAsync found failed: ") +
                               cudaGetErrorString(err));
    }

    // Synchronize if stream is null
    if (stream == nullptr) {
        cudaDeviceSynchronize();
    }

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_found);
}

}  // namespace warpkv
