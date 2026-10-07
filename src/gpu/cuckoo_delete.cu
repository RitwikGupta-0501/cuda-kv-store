#include "cuckoo_delete.h"

namespace warpkv {

void warp_delete_batch_sync(
    BucketTable table,
    StashQueue* d_stash,
    const DeleteBatch& batch,
    cudaStream_t stream) {

    if (batch.num_keys == 0) return;

    KeyT* d_keys = nullptr;
    uint32_t* d_deleted = nullptr;

    size_t keys_size = batch.num_keys * sizeof(KeyT);
    size_t deleted_size = batch.num_keys * sizeof(uint32_t);

    cudaMalloc(&d_keys, keys_size);
    cudaMalloc(&d_deleted, deleted_size);

    cudaMemcpyAsync(d_keys, batch.h_keys, keys_size, cudaMemcpyHostToDevice, stream);

    uint32_t threads_per_block = 256;
    uint32_t keys_per_block = threads_per_block / 32;
    uint32_t num_blocks = (batch.num_keys + keys_per_block - 1) / keys_per_block;

    warp_delete_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        table, d_stash, d_keys, d_deleted, batch.num_keys
    );

    cudaMemcpyAsync(batch.h_deleted, d_deleted, deleted_size, cudaMemcpyDeviceToHost, stream);

    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    cudaFree(d_keys);
    cudaFree(d_deleted);
}

}  // namespace warpkv
