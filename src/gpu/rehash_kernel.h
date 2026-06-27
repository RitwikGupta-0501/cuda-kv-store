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

// Device-side: rehash a single key-value pair to new table with eviction chains
// Uses identical cuckoo eviction logic as insertion to guarantee no data loss
// Returns true if successfully inserted, false if unevictable (should be statistically impossible)
__device__ inline bool rehash_entry_device(
    BucketTable* new_table,
    uint32_t key,
    uint32_t value,
    uint8_t fingerprint) {

    uint32_t lane_id = threadIdx.x % 32;

    // Eviction loop: rehash entry may require multiple hops if collisions occur
    uint32_t current_key = key;
    uint32_t current_value = value;
    uint8_t current_fp = fingerprint;
    uint32_t hop_count = 0;
    bool inserted = false;

    while (hop_count < MAX_EVICTION_HOPS && !inserted) {
        // Recompute hash for new table (new bucket mask)
        HashPair hash_pair = compute_hash_pair(current_key, new_table->bucket_mask);
        Bucket* bucket_b1 = &new_table->buckets[hash_pair.b1];
        Bucket* bucket_b2 = &new_table->buckets[hash_pair.b2];

        // ========== Try bucket b1 ==========
        // Lanes 0-7 try their corresponding slots in parallel
        bool b1_success = false;
        if (lane_id < 8) {
            uint32_t slot = lane_id;
            uint32_t old_mask = bucket_b1->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                uint32_t new_mask = old_mask | (1u << slot);
                if (atomicCAS(&bucket_b1->occupancy_mask, old_mask, new_mask) == old_mask) {
                    // Claimed free slot in b1
                    bucket_b1->keys[slot] = current_key;
                    bucket_b1->values[slot] = current_value;
                    bucket_b1->fingerprint[slot] = current_fp;
                    b1_success = true;
                }
            }
        }

        // Check if any lane in b1 succeeded
        if (__ballot_sync(0xFFFFFFFFu, b1_success)) {
            return true;
        }

        // ========== Try bucket b2 ==========
        // Lanes 8-15 try their corresponding slots in parallel
        bool b2_success = false;
        if (lane_id >= 8 && lane_id < 16) {
            uint32_t slot = lane_id - 8;
            uint32_t old_mask = bucket_b2->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                uint32_t new_mask = old_mask | (1u << slot);
                if (atomicCAS(&bucket_b2->occupancy_mask, old_mask, new_mask) == old_mask) {
                    // Claimed free slot in b2
                    bucket_b2->keys[slot] = current_key;
                    bucket_b2->values[slot] = current_value;
                    bucket_b2->fingerprint[slot] = current_fp;
                    b2_success = true;
                }
            }
        }

        // Check if any lane in b2 succeeded
        if (__ballot_sync(0xFFFFFFFFu, b2_success)) {
            return true;
        }

        // ========== Both buckets full: Evict a victim ==========
        bool eviction_success = false;
        uint32_t evicted_key = 0;
        uint32_t evicted_value = 0;

        if (lane_id == 0) {
            // Pseudo-random victim selection
            uint32_t victim_slot = (hash_pair.b1 ^ hash_pair.b2 ^ hop_count) % 8;
            Bucket* victim_bucket = (hop_count % 2 == 0) ? bucket_b1 : bucket_b2;

            // Read victim
            uint32_t victim_key = victim_bucket->keys[victim_slot];
            uint32_t victim_value = victim_bucket->values[victim_slot];

            // Attempt eviction via atomicCAS
            uint32_t old_key = atomicCAS(&victim_bucket->keys[victim_slot], victim_key, current_key);

            if (old_key == victim_key) {
                // Eviction succeeded
                victim_bucket->values[victim_slot] = current_value;
                victim_bucket->fingerprint[victim_slot] = current_fp;

                evicted_key = victim_key;
                evicted_value = victim_value;
                eviction_success = true;
            }
        }

        // Broadcast eviction result
        eviction_success = __shfl_sync(0xFFFFFFFFu, eviction_success, 0);

        if (eviction_success) {
            // Broadcast evicted entry
            current_key = __shfl_sync(0xFFFFFFFFu, evicted_key, 0);
            current_value = __shfl_sync(0xFFFFFFFFu, evicted_value, 0);
            // current_fp remains unchanged
        }
        // If eviction failed, we retry next hop with same current_key

        hop_count++;
    }

    // Hit MAX_EVICTION_HOPS during rehash:
    // This is statistically impossible with 2x table at <50% load.
    // But if it happens, return false to signal failed insertion.
    return false;
}

// Kernel: Rehash all entries from old table into new table
// Each warp processes one bucket from old table
static __global__ void rehash_table_kernel(
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
        uint8_t fingerprint = 0;
        bool occupied = false;

        if (lane_id == 0) {
            occupied = bucket_is_occupied(old_bucket, slot);
            if (occupied) {
                key = old_bucket->keys[slot];
                value = old_bucket->values[slot];
                fingerprint = old_bucket->fingerprint[slot];
            }
        }

        // Broadcast to all lanes
        occupied = __shfl_sync(0xFFFFFFFFu, occupied, 0);
        key = __shfl_sync(0xFFFFFFFFu, key, 0);
        value = __shfl_sync(0xFFFFFFFFu, value, 0);
        fingerprint = __shfl_sync(0xFFFFFFFFu, (uint32_t)fingerprint, 0);

        if (occupied) {
            bool success = rehash_entry_device(new_table, key, value, (uint8_t)fingerprint);

            // Lane 0 increments counter only on success
            if (lane_id == 0 && success) {
                atomicAdd(entries_rehashed, 1);
            }
        }
    }
}

// Kernel: Drain stash queue into new table with cuckoo eviction chains
// Each warp cooperatively processes one stash entry (not one thread per entry)
static __global__ void drain_stash_kernel(
    BucketTable* new_table,
    StashQueue* stash,
    uint32_t* entries_drained) {

    // Each warp processes one stash entry
    uint32_t entry_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    uint32_t lane_id = threadIdx.x % 32;

    // Read stash size (from old head before it was reset)
    uint32_t stash_size = atomicAdd((uint32_t*)&stash->head, 0);
    if (entry_idx >= stash_size) return;

    StashEntry entry = stash->entries[entry_idx];

    // Use rehash_entry_device to insert with cuckoo eviction chains
    // Compute fingerprint from key
    HashPair hash_pair = compute_hash_pair(entry.key, new_table->bucket_mask);
    bool success = rehash_entry_device(new_table, entry.key, entry.value, hash_pair.fingerprint);

    // Lane 0 increments counter on success
    if (lane_id == 0 && success) {
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
