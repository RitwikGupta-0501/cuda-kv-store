#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace warpkv {

// GPU-Optimized Integer Hash — 32-bit variant for short keys (≤4 bytes)
//
// This is a Murmur3-derived finalizer (fmix32), selected for:
// - Excellent avalanche at low key lengths (all output bits flip with any input bit)
// - No known bias at low moduli
// - Minimal instruction count on GPU (4 multiplies, 3 XOR-shifts)
//
// Reference: Austin Appleby, MurmurHash3 (public domain)
// Constants: 0x85EBCA77 and 0xC2B2AE3D are from fmix32 in MurmurHash3.

// GPU device version: called from kernels
__device__ __forceinline__ uint32_t warpkv_hash32(uint32_t key) {
    uint32_t h = key + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16;
    return h;
}

// Host version: for CPU-side preprocessing and testing
inline uint32_t warpkv_hash32_host(uint32_t key) {
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
    uint32_t b1;         // Primary bucket index
    uint32_t b2;         // Secondary bucket index
    uint8_t  fingerprint; // Upper 8 bits for fast rejection
};

// Compute both buckets and fingerprint from a key.
//
// b1: primary hash, masked to table size.
// b2: independent second hash via an additional mixing step (not just XOR of b1).
//     Using a second full mix pass ensures good independence even for small tables.
//     If b1 == b2 (possible for very small tables), b2 is nudged by +1.
// fingerprint: upper 8 bits of h, used for fast rejection before key comparison.
__device__ __host__ inline HashPair compute_hash_pair(uint32_t key, uint32_t bucket_mask) {
#ifdef __CUDA_ARCH__
    const uint32_t h  = warpkv_hash32(key);          // Device code
#else
    const uint32_t h  = warpkv_hash32_host(key);     // Host code
#endif

    // Derive a second independent hash by running another mixing step on h.
    uint32_t h2 = h;
    h2 ^= h2 >> 16;
    h2 *= 0x45d9f3bu;
    h2 ^= h2 >> 16;

    HashPair result;
    result.b1          = h  & bucket_mask;
    result.b2          = h2 & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 24);

    // Ensure b1 != b2 for all table sizes (collision possible when mask is small).
    if (result.b2 == result.b1) {
        result.b2 = (result.b2 + 1) & bucket_mask;
    }

    return result;
}

}  // namespace warpkv
