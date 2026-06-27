#include "xxhash3.h"
#include <cstring>

namespace warpkv {

// GPU kernel: batch hash computation
// Input: array of keys
// Output: array of hash values
__global__ void xxhash3_batch_kernel(
    const uint32_t* __restrict__ d_keys,
    uint32_t* __restrict__ d_hashes,
    uint32_t num_keys) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_keys) {
        d_hashes[idx] = xxhash3_32(d_keys[idx]);
    }
}

// Host function: batch hash computation (GPU)
void xxhash3_batch_gpu(
    const uint32_t* d_keys,
    uint32_t* d_hashes,
    uint32_t num_keys) {

    int block_size = 256;
    int grid_size = (num_keys + block_size - 1) / block_size;

    xxhash3_batch_kernel<<<grid_size, block_size>>>(
        d_keys, d_hashes, num_keys);
}

// Host function: batch hash computation (CPU)
void xxhash3_batch_cpu(
    const uint32_t* h_keys,
    uint32_t* h_hashes,
    uint32_t num_keys) {

    for (uint32_t i = 0; i < num_keys; ++i) {
        h_hashes[i] = xxhash3_32_host(h_keys[i]);
    }
}

// Host function: compute hash pair (for testing)
HashPair compute_hash_pair_host(uint32_t key, uint32_t bucket_mask) {
    const uint32_t h = xxhash3_32_host(key);

    HashPair result;
    result.b1 = h & bucket_mask;
    result.b2 = ((h >> 16) ^ 0xDEADBEEFu) & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 24);

    return result;
}

}  // namespace warpkv
