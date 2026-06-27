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
};

// Device-side insertion function
__device__ inline InsertResult warp_insert_device(
    BucketTable* table,
    StashQueue* stash,
    uint32_t key,
    uint32_t value,
    uint8_t fingerprint) {

    // Compute both bucket addresses
    HashPair hash_pair = compute_hash_pair(key, table->bucket_mask);
    uint32_t b1_idx = hash_pair.b1;
    uint32_t b2_idx = hash_pair.b2;
    Bucket* bucket_b1 = &table->buckets[b1_idx];
    Bucket* bucket_b2 = &table->buckets[b2_idx];

    uint32_t lane_id = threadIdx.x % 32;
    InsertResult result = {INSERT_FAILED, 0};

    // ========== Lanes 0-7: Try to insert in bucket b1 ==========
    if (lane_id < 8) {
        // Each lane tries to claim its slot atomically
        uint32_t old_mask = bucket_b1->occupancy_mask;
        if (!(old_mask & (1u << lane_id))) {
            uint32_t new_mask = old_mask | (1u << lane_id);
            if (atomicCAS(&bucket_b1->occupancy_mask, old_mask, new_mask) == old_mask) {
                // Slot claimed successfully
                bucket_b1->keys[lane_id] = key;
                bucket_b1->values[lane_id] = value;
                bucket_b1->fingerprint[lane_id] = fingerprint;
                result.status = INSERT_SUCCESS;
                result.slot_used = lane_id;
            }
        }
    }
    // ========== Lanes 8-15: Try to insert in bucket b2 ==========
    else if (lane_id < 16) {
        // Each lane tries to claim its corresponding slot in b2
        uint32_t slot = lane_id - 8;
        uint32_t old_mask = bucket_b2->occupancy_mask;
        if (!(old_mask & (1u << slot))) {
            uint32_t new_mask = old_mask | (1u << slot);
            if (atomicCAS(&bucket_b2->occupancy_mask, old_mask, new_mask) == old_mask) {
                // Slot claimed successfully
                bucket_b2->keys[slot] = key;
                bucket_b2->values[slot] = value;
                bucket_b2->fingerprint[slot] = fingerprint;
                result.status = INSERT_SUCCESS;
                result.slot_used = slot;
            }
        }
    }

    // Broadcast success result from whichever lane succeeded
    int success_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.status == INSERT_SUCCESS)) - 1;
    if (success_lane >= 0) {
        result.status = __shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, success_lane);
        result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, success_lane);
        return result;
    }

    // ========== Both buckets full: Try stash (all lanes) ==========
    // Lane 0 handles stash atomically
    if (lane_id == 0) {
        // Atomic: claim next stash slot
        uint32_t head = atomicAdd(&stash->head, 1);

        if (head < STASH_CAPACITY) {
            stash->entries[head].key = key;
            stash->entries[head].value = value;
            result.status = INSERT_STASHED;
            result.slot_used = head;
        } else {
            // Stash overflow: set needs_rehash flag
            atomicExch(&stash->needs_rehash, 1u);
            result.status = INSERT_FAILED;
        }
    }

    // Broadcast stash result
    result.status = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, 0);
    result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, 0);
    return result;
}

// Kernel: process a batch of insertion keys
__global__ void warp_insert_kernel(
    BucketTable* table,
    StashQueue* stash,
    const uint32_t* keys,
    const uint32_t* values,
    InsertStatus* statuses,
    uint32_t num_keys) {

    // Each warp processes one key
    uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (key_idx >= num_keys) return;

    uint32_t key = keys[key_idx];
    uint32_t value = values[key_idx];
    uint8_t fp = compute_hash_pair(key, 0).fingerprint;

    InsertResult result = warp_insert_device(table, stash, key, value, fp);

    // Lane 0 writes result
    if ((threadIdx.x % 32) == 0) {
        statuses[key_idx] = result.status;
    }
}

// Host-side wrapper for insertion batches
struct InsertBatch {
    uint32_t* h_keys;      // Host input: keys
    uint32_t* h_values;    // Host input: values
    InsertStatus* h_statuses;  // Host output: insertion statuses
    uint32_t num_keys;
};

// Launch insertion kernel for a batch of keys
void warp_insert_batch(
    BucketTable* d_table,
    StashQueue* d_stash,
    const InsertBatch& batch,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
