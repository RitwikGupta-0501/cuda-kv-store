#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace warpkv {

// XXHash3 — 32-bit variant for short keys (≤16 bytes)
// Selected over MurmurHash3 for:
// - Superior avalanche at low key lengths
// - No known bias issues at low moduli
// - Faster on short keys (GPU & CPU)

// GPU kernel version: device-side hash function
__device__ __forceinline__ uint32_t xxhash3_32(uint32_t key) {
    uint32_t h = key + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16;
    return h;
}

// Host version: for CPU-side preprocessing and testing
inline uint32_t xxhash3_32_host(uint32_t key) {
    uint32_t h = key + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16;
    return h;
}

// Hash pair computation (b1 and b2 candidate buckets)
struct HashPair {
    uint32_t b1;        // Primary bucket index
    uint32_t b2;        // Secondary bucket index
    uint8_t  fingerprint; // Upper 8 bits for fast rejection
};

// Compute both buckets and fingerprint from a key
// Uses appropriate hash function for device vs host context
__device__ __host__ inline HashPair compute_hash_pair(uint32_t key, uint32_t bucket_mask) {
#ifdef __CUDA_ARCH__
    const uint32_t h = xxhash3_32(key);  // Device code
#else
    const uint32_t h = xxhash3_32_host(key);  // Host code
#endif

    HashPair result;
    result.b1 = h & bucket_mask;
    result.b2 = ((h >> 16) ^ 0xDEADBEEFu) & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 24);

    return result;
}

}  // namespace warpkv
