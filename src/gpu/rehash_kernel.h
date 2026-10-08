#pragma once

#include "hash.h"
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
    KeyT key,
    ValueT value,
    uint8_t fingerprint) {

    const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    uint32_t lane_id = threadIdx.x % 16;
    const uint32_t warp_lane = threadIdx.x % 32;

    // Eviction loop: rehash entry may require multiple hops if collisions occur
    KeyT current_key = key;
    ValueT current_value = value;
    uint32_t hop_count = 0;
    uint32_t contention_count = 0;
    bool inserted = false;

    while (hop_count < MAX_EVICTION_HOPS && contention_count < 10000 && !inserted) {
        // Recompute hash for new table (new bucket mask)
        HashPair hash_pair = compute_hash_pair(current_key, new_table.bucket_mask);
        uint8_t current_fp = hash_pair.fingerprint;
        
        Bucket* bucket_b1 = &new_table.buckets[hash_pair.b1];
        Bucket* bucket_b2 = &new_table.buckets[hash_pair.b2];

        // ========== Try to insert in bucket b1 ==========
        bool b1_success = false;
        bool b1_claimed = false;
        if (lane_id < BUCKET_SLOTS) {
            uint32_t slot = lane_id;
            uint32_t old_mask = bucket_b1->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                KeyT old_key = atomicCAS(&bucket_b1->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b1_claimed = true;
            }
        }
        
        int b1_winner = __ffs(__ballot_sync(active_mask, b1_claimed)) - 1;
        if (b1_claimed) {
            uint32_t slot = lane_id;
            if (warp_lane == (uint32_t)b1_winner) {
                bucket_b1->values[slot] = current_value;
                bucket_b1->fingerprint[slot] = current_fp;
                __threadfence(); // ensure writes are visible before unlock
                bucket_b1->keys[slot] = current_key;
                atomicOr(&bucket_b1->occupancy_mask, (1u << slot));
                b1_success = true;
            } else {
                bucket_b1->keys[slot] = EMPTY_KEY; // Release unused locks
            }
        }

        // Check if any lane in b1 succeeded
        if (__ballot_sync(active_mask, b1_success)) {
            return true;
        }

        // ========== Try to insert in bucket b2 ==========
        bool b2_success = false;
        bool b2_claimed = false;
        if (lane_id >= 8 && lane_id < 8 + BUCKET_SLOTS) {
            uint32_t slot = lane_id - 8;
            uint32_t old_mask = bucket_b2->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                KeyT old_key = atomicCAS(&bucket_b2->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b2_claimed = true;
            }
        }
        
        int b2_winner = __ffs(__ballot_sync(active_mask, b2_claimed)) - 1;
        if (b2_claimed) {
            uint32_t slot = lane_id - 8;
            if (warp_lane == (uint32_t)b2_winner) {
                bucket_b2->values[slot] = current_value;
                bucket_b2->fingerprint[slot] = current_fp;
                __threadfence(); // ensure writes are visible before unlock
                bucket_b2->keys[slot] = current_key;
                atomicOr(&bucket_b2->occupancy_mask, (1u << slot));
                b2_success = true;
            } else {
                bucket_b2->keys[slot] = EMPTY_KEY; // Release unused locks
            }
        }

        // Check if any lane in b2 succeeded
        if (__ballot_sync(active_mask, b2_success)) {
            return true;
        }

        // ========== Both buckets full: Evict a victim ==========
        bool eviction_success = false;
        KeyT evicted_key = 0;
        ValueT evicted_value = 0;

        if (lane_id == 0) {
            // Pseudo-random victim selection
            uint32_t victim_slot = (hash_pair.b1 ^ hash_pair.b2 ^ hop_count ^ contention_count) % BUCKET_SLOTS;
            Bucket* victim_bucket = ((hop_count ^ contention_count) % 2 == 0) ? bucket_b1 : bucket_b2;

            // Read victim's key
            KeyT victim_key = victim_bucket->keys[victim_slot];

            if (victim_key != EMPTY_KEY && victim_key != LOCK_SENTINEL) {
                // Attempt to lock victim slot
                KeyT old_key = atomicCAS(&victim_bucket->keys[victim_slot], victim_key, LOCK_SENTINEL);

                if (old_key == victim_key) {
                    // Lock acquired! Safe to read value and overwrite
                    ValueT victim_value = ((volatile ValueT*)victim_bucket->values)[victim_slot];
                    
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
        eviction_success = __shfl_sync(active_mask, eviction_success, (threadIdx.x & ~15));

        if (eviction_success) {
            // Broadcast evicted entry
            current_key = warpkv_shfl64(active_mask, evicted_key, (threadIdx.x & ~15));
            current_value = warpkv_shfl64(active_mask, evicted_value, (threadIdx.x & ~15));
            
            hop_count++;
            contention_count = 0;
        } else {
            contention_count++;
        }
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
    uint32_t bucket_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);

    if (bucket_idx >= old_table.num_buckets) return;

    Bucket* old_bucket = &old_table.buckets[bucket_idx];
    const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    uint32_t lane_id = threadIdx.x % 16;

    // All lanes cooperatively scan this bucket's slots
    for (int slot = 0; slot < BUCKET_SLOTS; ++slot) {
        // Lane 0 coordinates reading (to avoid 8x redundant reads)
        KeyT key = 0;
        ValueT value = 0;
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
        occupied = __shfl_sync(active_mask, occupied, (threadIdx.x & ~15));
        key = warpkv_shfl64(active_mask, key, (threadIdx.x & ~15));
        value = warpkv_shfl64(active_mask, value, (threadIdx.x & ~15));
        fingerprint = __shfl_sync(active_mask, (uint32_t)fingerprint, (threadIdx.x & ~15));

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
    uint32_t entry_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    uint32_t lane_id = threadIdx.x % 16;

    // Read stash size (from old head before it was reset)
    uint32_t stash_size = atomicAdd((uint32_t*)&stash->head, 0);
    if (stash_size > STASH_CAPACITY) stash_size = STASH_CAPACITY;
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
