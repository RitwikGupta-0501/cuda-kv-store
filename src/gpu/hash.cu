#include "hash.h"
#include <cstring>

namespace warpkv {

// GPU kernel: batch hash computation
// Input: array of keys
// Output: array of hash values
__global__ void warpkv_hash32_batch_kernel(
    const uint32_t* __restrict__ d_keys,
    uint32_t* __restrict__ d_hashes,
    uint32_t num_keys) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_keys) {
        d_hashes[idx] = warpkv_hash32(d_keys[idx]);
    }
}

// Host function: batch hash computation (GPU)
void warpkv_hash32_batch_gpu(
    const uint32_t* d_keys,
    uint32_t* d_hashes,
    uint32_t num_keys) {

    int block_size = 256;
    int grid_size = (num_keys + block_size - 1) / block_size;

    warpkv_hash32_batch_kernel<<<grid_size, block_size>>>(
        d_keys, d_hashes, num_keys);
}

// Host function: batch hash computation (CPU)
void warpkv_hash32_batch_cpu(
    const uint32_t* h_keys,
    uint32_t* h_hashes,
    uint32_t num_keys) {

    for (uint32_t i = 0; i < num_keys; ++i) {
        h_hashes[i] = warpkv_hash32_host(h_keys[i]);
    }
}

// Host function: compute hash pair (for testing)
HashPair compute_hash_pair_host(uint32_t key, uint32_t bucket_mask) {
    const uint32_t h = warpkv_hash32_host(key);

    uint32_t h2 = h;
    h2 ^= h2 >> 16;
    h2 *= 0x45d9f3bu;
    h2 ^= h2 >> 16;

    HashPair result;
    result.b1          = h  & bucket_mask;
    result.b2          = h2 & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 24);

    if (result.b2 == result.b1) {
        result.b2 = (result.b2 + 1) & bucket_mask;
    }

    return result;
}

}  // namespace warpkv
