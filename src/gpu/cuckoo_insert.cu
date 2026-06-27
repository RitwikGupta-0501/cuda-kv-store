#include "cuckoo_insert.h"
#include <stdexcept>
#include <cstring>

namespace warpkv {

// Host-side wrapper: allocate temporary buffers and launch kernel
void warp_insert_batch(
    BucketTable* d_table,
    StashQueue* d_stash,
    const InsertBatch& batch,
    cudaStream_t stream) {

    if (batch.num_keys == 0) {
        return;
    }

    // Allocate device-side buffers for this batch
    uint32_t* d_keys = nullptr;
    uint32_t* d_values = nullptr;
    InsertStatus* d_statuses = nullptr;

    size_t keys_size = batch.num_keys * sizeof(uint32_t);
    size_t values_size = batch.num_keys * sizeof(uint32_t);
    size_t statuses_size = batch.num_keys * sizeof(InsertStatus);

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

    err = cudaMalloc(&d_statuses, statuses_size);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        throw std::runtime_error(std::string("cudaMalloc d_statuses failed: ") +
                               cudaGetErrorString(err));
    }

    // Copy keys and values to device
    err = cudaMemcpyAsync(d_keys, batch.h_keys, keys_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_statuses);
        throw std::runtime_error(std::string("cudaMemcpyAsync keys failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(d_values, batch.h_values, values_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_statuses);
        throw std::runtime_error(std::string("cudaMemcpyAsync values failed: ") +
                               cudaGetErrorString(err));
    }

    // Launch kernel: one warp (32 threads) per key
    uint32_t threads_per_block = 256;  // 8 warps per block
    uint32_t keys_per_block = threads_per_block / 32;
    uint32_t num_blocks = (batch.num_keys + keys_per_block - 1) / keys_per_block;

    warp_insert_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_table,
        d_stash,
        d_keys,
        d_values,
        d_statuses,
        batch.num_keys
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_statuses);
        throw std::runtime_error(std::string("Kernel launch failed: ") +
                               cudaGetErrorString(err));
    }

    // Copy results back to host
    err = cudaMemcpyAsync(batch.h_statuses, d_statuses, statuses_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_statuses);
        throw std::runtime_error(std::string("cudaMemcpyAsync statuses failed: ") +
                               cudaGetErrorString(err));
    }

    // Synchronize if stream is null
    if (stream == nullptr) {
        cudaDeviceSynchronize();
    }

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_statuses);
}

}  // namespace warpkv
