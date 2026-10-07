#include "hash.h"
#include <cstring>

namespace warpkv {

// GPU kernel: batch hash computation
// Input: array of keys
// Output: array of hash values
__global__ void warpkv_hash64_batch_kernel(
    const KeyT* __restrict__ d_keys,
    uint64_t* __restrict__ d_hashes,
    uint32_t num_keys) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_keys) {
        d_hashes[idx] = warpkv_hash64(d_keys[idx]);
    }
}

// Host function: batch hash computation (GPU)
void warpkv_hash64_batch_gpu(
    const KeyT* d_keys,
    uint64_t* d_hashes,
    uint32_t num_keys) {

    int block_size = 256;
    int grid_size = (num_keys + block_size - 1) / block_size;

    warpkv_hash64_batch_kernel<<<grid_size, block_size>>>(
        d_keys, d_hashes, num_keys);
}

// Host function: batch hash computation (CPU)
void warpkv_hash64_batch_cpu(
    const KeyT* h_keys,
    uint64_t* h_hashes,
    uint32_t num_keys) {

    for (uint32_t i = 0; i < num_keys; ++i) {
        h_hashes[i] = warpkv_hash64_host(h_keys[i]);
    }
}

// Host function: compute hash pair (for testing)
HashPair compute_hash_pair_host(KeyT key, uint32_t bucket_mask) {
    return compute_hash_pair(key, bucket_mask);
}

}  // namespace warpkv
