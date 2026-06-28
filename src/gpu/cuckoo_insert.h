#pragma once

#include "xxhash3.h"
#include "bucket_cuckoo.h"
#include <cuda_runtime.h>

namespace warpkv {

// ============================================================================
// Cuckoo Insertion Kernel with Eviction Chains
// ============================================================================
// One warp processes one insertion:
// 1. Try to insert in bucket b1 (lanes 0-7 find first free slot)
// 2. If full, try bucket b2 (lanes 8-15 find first free slot)
// 3. If both full, follow eviction chain (up to MAX_EVICTION_HOPS=32)
// 4. If eviction fails, insert into stash queue
// Returns insert status (success, stashed, or failed)
// ============================================================================

enum InsertStatus : uint32_t {
    INSERT_SUCCESS = 0,    // Inserted into bucket
    INSERT_STASHED = 1,    // Inserted into stash
    INSERT_FAILED = 2      // Both buckets full, stash full (should not happen)
};

struct InsertResult {
    InsertStatus status;
    uint32_t slot_used;    // Slot index in final bucket (0-7 if in bucket, ignored if stashed)
    uint32_t hops;         // Number of eviction hops required
};

#ifdef __CUDACC__

// Device-side insertion function with cuckoo eviction chains
// Implements full cuckoo hashing: try b1, try b2, then evict up to MAX_EVICTION_HOPS
__device__ inline InsertResult warp_insert_device(
    BucketTable table,
    StashQueue* stash,
    uint32_t key,
    uint32_t value,
    uint8_t fingerprint) {

    uint32_t lane_id = threadIdx.x % 32;
    InsertResult result = {INSERT_FAILED, 0, 0};

    // Eviction loop: current_key/value/fp get evicted and re-inserted up to MAX_EVICTION_HOPS times
    uint32_t current_key = key;
    uint32_t current_value = value;
    uint32_t hop_count = 0;

    while (hop_count < MAX_EVICTION_HOPS) {
        // Compute buckets for the current key
        HashPair hash_pair = compute_hash_pair(current_key, table.bucket_mask);
        uint8_t current_fp = hash_pair.fingerprint;
        
        Bucket* bucket_b1 = &table.buckets[hash_pair.b1];
        Bucket* bucket_b2 = &table.buckets[hash_pair.b2];

        // ========== Try to insert in bucket b1 ==========
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
                
                result.status = INSERT_SUCCESS;
                result.slot_used = slot;
                result.hops = hop_count;
            } else {
                bucket_b1->keys[slot] = 0; // Release unused locks
            }
        }

        // Broadcast success from any lane
        int success_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.status == INSERT_SUCCESS)) - 1;
        if (success_lane >= 0) {
            result.status = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, success_lane);
            result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, success_lane);
            result.hops = __shfl_sync(0xFFFFFFFFu, result.hops, success_lane);
            return result;
        }

        // ========== Try to insert in bucket b2 ==========
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
                
                result.status = INSERT_SUCCESS;
                result.slot_used = slot;
                result.hops = hop_count;
            } else {
                bucket_b2->keys[slot] = 0; // Release unused locks
            }
        }

        // Broadcast success from any lane
        success_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.status == INSERT_SUCCESS)) - 1;
        if (success_lane >= 0) {
            result.status = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, success_lane);
            result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, success_lane);
            result.hops = __shfl_sync(0xFFFFFFFFu, result.hops, success_lane);
            return result;
        }

        // ========== Both buckets full: Evict a victim ==========
        bool eviction_success = false;
        uint32_t evicted_key = 0;
        uint32_t evicted_value = 0;

        if (lane_id == 0) {
            // Pseudo-random victim selection: use hash bits XOR'd with hop count
            uint32_t victim_slot = (hash_pair.b1 ^ hash_pair.b2 ^ hop_count) % 8;

            // Alternate between b1 and b2 based on hop count
            Bucket* victim_bucket = (hop_count % 2 == 0) ? bucket_b1 : bucket_b2;

            // Read the victim's key
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

        // Broadcast eviction success to all lanes
        eviction_success = __shfl_sync(0xFFFFFFFFu, eviction_success, 0);

        if (eviction_success) {
            // Broadcast the evicted key and value to all lanes
            current_key = __shfl_sync(0xFFFFFFFFu, evicted_key, 0);
            current_value = __shfl_sync(0xFFFFFFFFu, evicted_value, 0);
            // current_fp will be recomputed at the start of the next hop
        }
        // If eviction failed, current_key/value/fp remain unchanged and we loop again

        hop_count++;
    }

    // ========== Hit MAX_EVICTION_HOPS: Dump final evicted key to stash ==========
    if (lane_id == 0) {
        // Cast std::atomic<uint32_t>* to uint32_t* for CUDA intrinsics
        uint32_t head = atomicAdd((uint32_t*)&stash->head, 1);

        if (head < STASH_CAPACITY) {
            stash->entries[head].key = current_key;
            stash->entries[head].value = current_value;
            result.status = INSERT_STASHED;
            result.hops = hop_count;
        } else {
            // Stash overflow: set needs_rehash flag
            atomicExch((uint32_t*)&stash->needs_rehash, 1u);
            result.status = INSERT_FAILED;
            result.hops = hop_count;
        }
    }

    // Broadcast final status and hops to all lanes
    result.status = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, 0);
    result.hops = __shfl_sync(0xFFFFFFFFu, result.hops, 0);

    return result;
}

// Kernel: process a batch of insertion keys
static __global__ void warp_insert_kernel(
    BucketTable table,
    StashQueue* stash,
    const uint32_t* keys,
    const uint32_t* values,
    InsertStatus* statuses,
    uint32_t* hops,
    uint32_t num_keys) {

    // Each warp processes one key
    uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (key_idx >= num_keys) return;

    uint32_t key = keys[key_idx];
    if (key == EMPTY_KEY) return;
    
    uint32_t value = values[key_idx];
    uint8_t fp = compute_hash_pair(key, table.bucket_mask).fingerprint;

    InsertResult result = warp_insert_device(table, stash, key, value, fp);

    // Lane 0 writes result
    if ((threadIdx.x % 32) == 0) {
        statuses[key_idx] = result.status;
        if (hops) hops[key_idx] = result.hops;
    }
}

#endif // __CUDACC__

struct InsertBatch {
    uint32_t* h_keys;      // Host input: keys
    uint32_t* h_values;    // Host input: values
    InsertStatus* h_statuses;  // Host output: insertion statuses
    uint32_t* h_hops;      // Host output: hops required (can be nullptr if not needed)
    uint32_t num_keys;
};

// Launch insertion kernel for a batch of keys
void warp_insert_batch(
    BucketTable table,
    StashQueue* d_stash,
    const InsertBatch& batch,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
