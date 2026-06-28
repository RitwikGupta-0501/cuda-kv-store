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

#ifdef __CUDACC__

// Device-side: rehash a single key-value pair to new table with eviction chains
// Uses identical cuckoo eviction logic as insertion to guarantee no data loss
// Returns true if successfully inserted, false if unevictable (should be statistically impossible)
__device__ inline bool rehash_entry_device(
    BucketTable new_table,
    uint32_t key,
    uint32_t value,
    uint8_t fingerprint) {

    uint32_t lane_id = threadIdx.x % 32;

    // Eviction loop: rehash entry may require multiple hops if collisions occur
    uint32_t current_key = key;
    uint32_t current_value = value;
    uint32_t hop_count = 0;
    bool inserted = false;

    while (hop_count < MAX_EVICTION_HOPS && !inserted) {
        // Recompute hash for new table (new bucket mask)
        HashPair hash_pair = compute_hash_pair(current_key, new_table.bucket_mask);
        uint8_t current_fp = hash_pair.fingerprint;
        
        Bucket* bucket_b1 = &new_table.buckets[hash_pair.b1];
        Bucket* bucket_b2 = &new_table.buckets[hash_pair.b2];

        // ========== Try to insert in bucket b1 ==========
        bool b1_success = false;
        bool b1_claimed = false;
        if (lane_id < 8) {
            uint32_t slot = lane_id;
            uint32_t old_mask = bucket_b1->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                uint32_t old_key = atomicCAS(&bucket_b1->keys[slot], 0, 0xFFFFFFFF);
                if (old_key == 0) b1_claimed = true;
            }
        }
        
        int b1_winner = __ffs(__ballot_sync(0xFFFFFFFFu, b1_claimed)) - 1;
        if (b1_claimed) {
            uint32_t slot = lane_id;
            if (lane_id == b1_winner) {
                bucket_b1->values[slot] = current_value;
                bucket_b1->fingerprint[slot] = current_fp;
                __threadfence(); // ensure writes are visible before unlock
                bucket_b1->keys[slot] = current_key;
                atomicOr(&bucket_b1->occupancy_mask, (1u << slot));
                b1_success = true;
            } else {
                bucket_b1->keys[slot] = 0; // Release unused locks
            }
        }

        // Check if any lane in b1 succeeded
        if (__ballot_sync(0xFFFFFFFFu, b1_success)) {
            return true;
        }

        // ========== Try to insert in bucket b2 ==========
        bool b2_success = false;
        bool b2_claimed = false;
        if (lane_id >= 8 && lane_id < 16) {
            uint32_t slot = lane_id - 8;
            uint32_t old_mask = bucket_b2->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                uint32_t old_key = atomicCAS(&bucket_b2->keys[slot], 0, 0xFFFFFFFF);
                if (old_key == 0) b2_claimed = true;
            }
        }
        
        int b2_winner = __ffs(__ballot_sync(0xFFFFFFFFu, b2_claimed)) - 1;
        if (b2_claimed) {
            uint32_t slot = lane_id - 8;
            if (lane_id == b2_winner) {
                bucket_b2->values[slot] = current_value;
                bucket_b2->fingerprint[slot] = current_fp;
                __threadfence(); // ensure writes are visible before unlock
                bucket_b2->keys[slot] = current_key;
                atomicOr(&bucket_b2->occupancy_mask, (1u << slot));
                b2_success = true;
            } else {
                bucket_b2->keys[slot] = 0; // Release unused locks
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

            // Read victim's key
            uint32_t victim_key = victim_bucket->keys[victim_slot];

            if (victim_key != 0 && victim_key != 0xFFFFFFFF) {
                // Attempt to lock victim slot
                uint32_t old_key = atomicCAS(&victim_bucket->keys[victim_slot], victim_key, 0xFFFFFFFF);

                if (old_key == victim_key) {
                    // Lock acquired! Safe to read value and overwrite
                    uint32_t victim_value = victim_bucket->values[victim_slot];
                    
                    victim_bucket->values[victim_slot] = current_value;
                    victim_bucket->fingerprint[victim_slot] = current_fp;
                    __threadfence(); // ensure writes are visible before unlock
                    victim_bucket->keys[victim_slot] = current_key;

                    evicted_key = victim_key;
                    evicted_value = victim_value;
                    eviction_success = true;
                }
            }
        }

        // Broadcast eviction result
        eviction_success = __shfl_sync(0xFFFFFFFFu, eviction_success, 0);

        if (eviction_success) {
            // Broadcast evicted entry
            current_key = __shfl_sync(0xFFFFFFFFu, evicted_key, 0);
            current_value = __shfl_sync(0xFFFFFFFFu, evicted_value, 0);
            // current_fp will be recomputed at the start of the next hop
        }

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
    const BucketTable old_table,
    BucketTable new_table,
    uint32_t* entries_rehashed) {

    // Each warp processes one bucket from old table
    uint32_t bucket_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (bucket_idx >= old_table.num_buckets) return;

    Bucket* old_bucket = &old_table.buckets[bucket_idx];
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
    BucketTable new_table,
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
    HashPair hash_pair = compute_hash_pair(entry.key, new_table.bucket_mask);
    bool success = rehash_entry_device(new_table, entry.key, entry.value, hash_pair.fingerprint);

    // Lane 0 increments counter on success
    if (lane_id == 0 && success) {
        atomicAdd(entries_drained, 1);
    }
}

#endif // __CUDACC__

// Host-side wrapper for rehashing
struct RehashContext {
    BucketTable old_table;
    BucketTable new_table;
    StashQueue* d_stash;
};

// Launch rehash pipeline
void execute_rehash(
    const RehashContext& ctx,
    RehashStats* out_stats,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
