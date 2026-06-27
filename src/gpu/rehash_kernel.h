#pragma once

#include "xxhash3.h"
#include "bucket_cuckoo.h"
#include <cuda_runtime.h>

namespace warpkv {

// ============================================================================
// Rehashing Kernel with EBR Double-Buffering
// ============================================================================
// Triggered when stash reaches BACKPRESSURE_THRESHOLD (50% of 128 = 64 entries)
// Rehash pipeline:
// 1. Copy all entries from old table to new table (with recomputed hash indices)
// 2. Drain stash queue into new table
// 3. Atomically swap table pointers (EBR epoch marker)
// 4. Wait for in-flight lookups/inserts to complete (epoch-based)
// 5. Free old table
// ============================================================================

enum RehashStatus : uint32_t {
    REHASH_START = 0,      // Rehash initiated
    REHASH_IN_PROGRESS = 1, // Rehash running
    REHASH_COMPLETE = 2,   // Rehash finished, old table reclaimable
    REHASH_FAILED = 3      // Rehash failed (new table allocation error)
};

struct RehashStats {
    uint32_t entries_copied;
    uint32_t entries_stashed;
    uint32_t new_table_capacity;
    RehashStatus status;
};

// Device-side: rehash a single key-value pair to new table
// Called by rehash kernel for each entry in old table
__device__ inline void rehash_entry_device(
    BucketTable* new_table,
    uint32_t key,
    uint32_t value) {

    // Recompute hash for new table (new bucket mask)
    HashPair hash_pair = compute_hash_pair(key, new_table->bucket_mask);
    Bucket* bucket_b1 = &new_table->buckets[hash_pair.b1];
    Bucket* bucket_b2 = &new_table->buckets[hash_pair.b2];

    uint32_t lane_id = threadIdx.x % 32;

    // ========== Lanes 0-7: Try bucket b1 ==========
    if (lane_id < 8) {
        uint32_t old_mask = bucket_b1->occupancy_mask;
        if (!(old_mask & (1u << lane_id))) {
            uint32_t new_mask = old_mask | (1u << lane_id);
            if (atomicCAS(&bucket_b1->occupancy_mask, old_mask, new_mask) == old_mask) {
                bucket_b1->keys[lane_id] = key;
                bucket_b1->values[lane_id] = value;
                bucket_b1->fingerprint[lane_id] = hash_pair.fingerprint;
            }
        }
    }
    // ========== Lanes 8-15: Try bucket b2 ==========
    else if (lane_id < 16) {
        uint32_t slot = lane_id - 8;
        uint32_t old_mask = bucket_b2->occupancy_mask;
        if (!(old_mask & (1u << slot))) {
            uint32_t new_mask = old_mask | (1u << slot);
            if (atomicCAS(&bucket_b2->occupancy_mask, old_mask, new_mask) == old_mask) {
                bucket_b2->keys[slot] = key;
                bucket_b2->values[slot] = value;
                bucket_b2->fingerprint[slot] = hash_pair.fingerprint;
            }
        }
    }
}

// Kernel: Rehash all entries from old table into new table
// Each warp processes one bucket from old table
__global__ void rehash_table_kernel(
    const BucketTable* old_table,
    BucketTable* new_table,
    uint32_t* entries_rehashed) {

    // Each warp processes one bucket from old table
    uint32_t bucket_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (bucket_idx >= old_table->num_buckets) return;

    Bucket* old_bucket = &old_table->buckets[bucket_idx];
    uint32_t lane_id = threadIdx.x % 32;

    // All lanes cooperatively scan this bucket's slots
    for (int slot = 0; slot < 8; ++slot) {
        // Lane 0 coordinates reading (to avoid 8x redundant reads)
        uint32_t key = 0;
        uint32_t value = 0;
        bool occupied = false;

        if (lane_id == 0) {
            occupied = bucket_is_occupied(old_bucket, slot);
            if (occupied) {
                key = old_bucket->keys[slot];
                value = old_bucket->values[slot];
            }
        }

        // Broadcast to all lanes
        occupied = __shfl_sync(0xFFFFFFFFu, occupied, 0);
        key = __shfl_sync(0xFFFFFFFFu, key, 0);
        value = __shfl_sync(0xFFFFFFFFu, value, 0);

        if (occupied) {
            rehash_entry_device(new_table, key, value);

            // Lane 0 increments counter
            if (lane_id == 0) {
                atomicAdd(entries_rehashed, 1);
            }
        }
    }
}

// Kernel: Drain stash queue into new table
__global__ void drain_stash_kernel(
    BucketTable* new_table,
    StashQueue* stash,
    uint32_t* entries_drained) {

    // Each thread processes one stash entry
    uint32_t entry_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t stash_size = atomicAdd((uint32_t*)&stash->tail, 0);  // Read tail
    if (entry_idx >= stash_size) return;

    StashEntry entry = stash->entries[entry_idx];

    // Rehash this entry into new table
    HashPair hash_pair = compute_hash_pair(entry.key, new_table->bucket_mask);
    uint32_t lane_id = threadIdx.x % 32;

    Bucket* bucket_b1 = &new_table->buckets[hash_pair.b1];
    Bucket* bucket_b2 = &new_table->buckets[hash_pair.b2];

    bool inserted = false;

    // Try b1
    if (lane_id < 8) {
        uint32_t old_mask = bucket_b1->occupancy_mask;
        if (!(old_mask & (1u << lane_id))) {
            uint32_t new_mask = old_mask | (1u << lane_id);
            if (atomicCAS(&bucket_b1->occupancy_mask, old_mask, new_mask) == old_mask) {
                bucket_b1->keys[lane_id] = entry.key;
                bucket_b1->values[lane_id] = entry.value;
                bucket_b1->fingerprint[lane_id] = hash_pair.fingerprint;
                inserted = true;
            }
        }
    }
    // Try b2
    else if (lane_id < 16) {
        uint32_t slot = lane_id - 8;
        uint32_t old_mask = bucket_b2->occupancy_mask;
        if (!(old_mask & (1u << slot))) {
            uint32_t new_mask = old_mask | (1u << slot);
            if (atomicCAS(&bucket_b2->occupancy_mask, old_mask, new_mask) == old_mask) {
                bucket_b2->keys[slot] = entry.key;
                bucket_b2->values[slot] = entry.value;
                bucket_b2->fingerprint[slot] = hash_pair.fingerprint;
                inserted = true;
            }
        }
    }

    if (inserted) {
        atomicAdd(entries_drained, 1);
    }
}

// Host-side wrapper for rehashing
struct RehashContext {
    BucketTable* d_old_table;
    BucketTable* d_new_table;
    StashQueue* d_stash;
};

// Launch rehash pipeline
void execute_rehash(
    const RehashContext& ctx,
    RehashStats* out_stats,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
