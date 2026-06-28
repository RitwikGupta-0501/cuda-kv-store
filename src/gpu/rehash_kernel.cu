#include "rehash_kernel.h"
#include <stdexcept>
#include <cstring>

namespace warpkv {

// Host-side wrapper: orchestrate the full rehash pipeline
void execute_rehash(
    const RehashContext& ctx,
    RehashStats* out_stats,
    cudaStream_t stream) {

    if (!ctx.d_stash) {
        throw std::runtime_error("Invalid rehash context stash pointer");
    }

    // Allocate device counters
    uint32_t* d_entries_rehashed = nullptr;
    uint32_t* d_entries_drained = nullptr;

    cudaError_t err = cudaMalloc(&d_entries_rehashed, sizeof(uint32_t));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc d_entries_rehashed failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_entries_drained, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        throw std::runtime_error(std::string("cudaMalloc d_entries_drained failed: ") +
                               cudaGetErrorString(err));
    }

    // Initialize counters
    uint32_t zero = 0;
    err = cudaMemcpyAsync(d_entries_rehashed, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("cudaMemcpyAsync init failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(d_entries_drained, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("cudaMemcpyAsync init failed: ") +
                               cudaGetErrorString(err));
    }

    // ========== Stage 1: Rehash all entries from old table ==========
    // Each warp processes one bucket
    uint32_t threads_per_block = 256;
    uint32_t warps_per_block = threads_per_block / 32;
    uint32_t num_blocks = (ctx.old_table.num_buckets + warps_per_block - 1) / warps_per_block;

    rehash_table_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        ctx.old_table,
        ctx.new_table,
        d_entries_rehashed
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("rehash_table_kernel launch failed: ") +
                               cudaGetErrorString(err));
    }

    // ========== Stage 2: Drain stash queue ==========
    // Each warp processes one stash entry
    uint32_t stash_threads = 256;
    uint32_t warps_per_stash_block = stash_threads / 32;
    uint32_t num_stash_warps = STASH_CAPACITY;
    uint32_t stash_blocks = (num_stash_warps + warps_per_stash_block - 1) / warps_per_stash_block;

    drain_stash_kernel<<<stash_blocks, stash_threads, 0, stream>>>(
        ctx.new_table,
        ctx.d_stash,
        d_entries_drained
    );
    
    // Reset the stash head to 0 so the stash is empty and ready for new evictions
    cudaMemsetAsync(&ctx.d_stash->head, 0, sizeof(uint32_t), stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("drain_stash_kernel launch failed: ") +
                               cudaGetErrorString(err));
    }

    // ========== Copy results back ==========
    uint32_t h_entries_rehashed = 0;
    uint32_t h_entries_drained = 0;

    err = cudaMemcpyAsync(&h_entries_rehashed, d_entries_rehashed, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("cudaMemcpyAsync rehashed count failed: ") +
                               cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(&h_entries_drained, d_entries_drained, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_entries_rehashed);
        cudaFree(d_entries_drained);
        throw std::runtime_error(std::string("cudaMemcpyAsync drained count failed: ") +
                               cudaGetErrorString(err));
    }

    // Synchronize to ensure async copies complete before we read the stack variables
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // Fill output stats
    if (out_stats) {
        out_stats->entries_copied = h_entries_rehashed;
        out_stats->entries_stashed = h_entries_drained;
        out_stats->new_table_capacity = ctx.new_table.num_buckets;
        out_stats->status = REHASH_COMPLETE;
    }

    // Cleanup
    cudaFree(d_entries_rehashed);
    cudaFree(d_entries_drained);
}

}  // namespace warpkv
