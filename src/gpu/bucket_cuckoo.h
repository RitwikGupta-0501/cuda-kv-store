#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace warpkv {

// ============================================================================
// Bucket Structure — 128 bytes = 1 L2 cache line (Bucket-AoS layout)
// ============================================================================

struct Bucket {
    // Keys: 32 bytes (8 × uint32_t)
    uint32_t keys[8];

    // Values: 32 bytes (8 × uint32_t)
    uint32_t values[8];

    // Fingerprints: 8 bytes (8 × uint8_t)
    uint8_t fingerprint[8];

    // Occupancy mask: 4 bytes (uint32_t, one bit per slot)
    uint32_t occupancy_mask;

    // Padding: 52 bytes (to reach 128 bytes total)
    uint8_t padding[52];
};

// Verify struct is exactly 128 bytes
static_assert(sizeof(Bucket) == 128, "Bucket must be exactly 128 bytes");

// ============================================================================
// Bucket Table Structure
// ============================================================================

struct BucketTable {
    // Pointer to contiguous array of buckets
    Bucket* buckets;

    // Number of buckets (power of 2)
    uint32_t num_buckets;

    // Bitmask for bucket indexing (num_buckets - 1)
    // Used instead of modulo: h & bucket_mask is one cycle vs. 20-40 cycles for %
    uint32_t bucket_mask;

    // Load factor limit (0.5 × num_buckets)
    // Insertion fails when entry count exceeds this
    uint32_t load_factor_limit;
};

// ============================================================================
// Stash Queue Structure — Mapped Pinned Memory
// ============================================================================

struct StashEntry {
    uint32_t key;
    uint32_t value;
};

struct StashQueue {
    // Next write position (atomically incremented by GPU)
    uint32_t head;

    // Next read position (incremented by CPU)
    uint32_t tail;

    // Flag: set by GPU if stash overflows or needs rehash
    uint32_t needs_rehash;

    // Stash entries: 16384 slots = BACKPRESSURE_THRESHOLD + 3 * BATCH_SIZE
    // = 4096 + 3 * 4096 = 16384
    StashEntry entries[16384];
};

// Verify size is reasonable
static_assert(sizeof(StashQueue) < 150000, "StashQueue should be < 150KB");

// ============================================================================
// Bucket Utility Functions (Host-side)
// ============================================================================

// Initialize a bucket to all-empty state
__host__ inline void bucket_init(Bucket* bucket) {
    bucket->occupancy_mask = 0;
    std::memset(bucket->keys, 0, sizeof(bucket->keys));
    std::memset(bucket->values, 0, sizeof(bucket->values));
    std::memset(bucket->fingerprint, 0, sizeof(bucket->fingerprint));
}

// Get a slot's occupancy bit
__host__ __device__ inline bool bucket_is_occupied(const Bucket* bucket, int slot) {
    return (bucket->occupancy_mask >> slot) & 1u;
}

// Set a slot's occupancy bit
__host__ __device__ inline void bucket_set_occupied(Bucket* bucket, int slot) {
    bucket->occupancy_mask |= (1u << slot);
}

// Clear a slot's occupancy bit
__host__ __device__ inline void bucket_clear_occupied(Bucket* bucket, int slot) {
    bucket->occupancy_mask &= ~(1u << slot);
}

// ============================================================================
// Constants for Phase 2+
// ============================================================================

// Maximum number of hops during insertion before key goes to stash
static constexpr uint32_t MAX_EVICTION_HOPS = 32;

// Backpressure threshold: triggers rehash when stash reaches this
static constexpr uint32_t BACKPRESSURE_THRESHOLD = 4096;

// Batch size: number of keys processed per GPU batch
static constexpr uint32_t BATCH_SIZE = 4096;

// Stash capacity: must be >= BACKPRESSURE_THRESHOLD + 3 * BATCH_SIZE
static constexpr uint32_t STASH_CAPACITY = 16384;

// Empty key marker (for CPU-side operations)
static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;

// Not found marker (for GPU kernels)
static constexpr uint32_t NOT_FOUND = 0xFFFFFFFFu;

}  // namespace warpkv
