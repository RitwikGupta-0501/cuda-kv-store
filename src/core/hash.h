#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace kvstore {

// Full MurmurHash3 64-bit implementation
__device__ __host__ inline uint64_t murmur3_hash64(uint64_t key,
                                                   uint64_t seed) {
  uint64_t h = seed;

  // Body
  uint64_t k = key;
  k *= 0x87c37b91114253d5ULL;
  k = (k << 31) | (k >> 33); // rotl64(k, 31)
  k *= 0x4cf5ad432745937fULL;

  h ^= k;
  h = (h << 27) | (h >> 37); // rotl64(h, 27)
  h = h * 5 + 0x52dce729;

  // Finalization
  h ^= 8; // length = 8 bytes
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;

  return h;
}

// Hash function 1: Seed = 0
__device__ __host__ inline uint32_t hash1(uint64_t key, uint32_t capacity) {
  uint64_t h = murmur3_hash64(key, 0);
  return h % capacity;
}

// Hash function 2: Different seed for independence
__device__ __host__ inline uint32_t hash2(uint64_t key, uint32_t capacity) {
  uint64_t h = murmur3_hash64(key, 0x9e3779b97f4a7c15ULL);
  return h % capacity;
}

// Combined hash computation
struct HashPair {
  uint32_t h1;
  uint32_t h2;
};

__device__ __host__ inline HashPair compute_hashes(uint64_t key,
                                                   uint32_t capacity) {
  return {hash1(key, capacity), hash2(key, capacity)};
}

} // namespace kvstore
