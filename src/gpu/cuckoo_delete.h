#pragma once

#include "hash.h"
#include "bucket_cuckoo.h"
#include <cuda_runtime.h>

namespace warpkv {

// ============================================================================
// Cuckoo Deletion Kernel
// ============================================================================
// One warp processes one deletion:
// 1. Lanes 0-7 scan b1 for the key.
// 2. Lanes 8-15 scan b2 for the key.
// 3. If found, lock slot, verify key, clear data, set EMPTY_KEY, unlock.
// 4. If not found in buckets, lane 0 scans the stash linearly.
// ============================================================================

#ifdef __CUDACC__

__device__ inline bool warp_delete_device(
    BucketTable table,
    StashQueue* stash,
    uint32_t key,
    uint8_t fingerprint) {
    
    uint32_t lane_id = threadIdx.x % 32;
    HashPair hash_pair = compute_hash_pair(key, table.bucket_mask);
    
    Bucket* bucket_b1 = &table.buckets[hash_pair.b1];
    Bucket* bucket_b2 = &table.buckets[hash_pair.b2];

    bool found = false;
    uint32_t slot = 0;
    Bucket* target_bucket = nullptr;

    // Check b1 (lanes 0-7)
    if (lane_id < 8) {
        if ((bucket_b1->occupancy_mask & (1u << lane_id)) && 
            bucket_b1->fingerprint[lane_id] == fingerprint &&
            bucket_b1->keys[lane_id] == key) {
            found = true;
            slot = lane_id;
            target_bucket = bucket_b1;
        }
    }
    
    // Check b2 (lanes 8-15)
    if (lane_id >= 8 && lane_id < 16) {
        uint32_t local_slot = lane_id - 8;
        if ((bucket_b2->occupancy_mask & (1u << local_slot)) && 
            bucket_b2->fingerprint[local_slot] == fingerprint &&
            bucket_b2->keys[local_slot] == key) {
            found = true;
            slot = local_slot;
            target_bucket = bucket_b2;
        }
    }

    // Determine if any lane found it
    int winner = __ffs(__ballot_sync(0xFFFFFFFFu, found)) - 1;
    
    bool delete_success = false;
    if (winner >= 0) {
        // Someone found it in the buckets!
        if (lane_id == winner) {
            // Attempt to lock the slot
            uint32_t old_key = atomicCAS(&target_bucket->keys[slot], key, LOCK_SENTINEL);
            if (old_key == key) {
                // Lock acquired. Clear the occupancy mask first so readers don't see incomplete states
                atomicAnd(&target_bucket->occupancy_mask, ~(1u << slot));
                target_bucket->values[slot] = 0;
                target_bucket->fingerprint[slot] = 0;
                __threadfence();
                // Release the lock by writing EMPTY_KEY
                target_bucket->keys[slot] = EMPTY_KEY;
                delete_success = true;
            }
        }
        delete_success = __shfl_sync(0xFFFFFFFFu, delete_success, winner);
        return delete_success;
    }

    // If not found in buckets, scan the stash
    if (lane_id == 0 && stash != nullptr) {
        // Linear scan of the stash
        uint32_t current_head = *(volatile uint32_t*)&stash->head;
        uint32_t count = min(current_head, STASH_CAPACITY);
        
        for (uint32_t i = 0; i < count; ++i) {
            if (stash->entries[i].key == key) {
                // Attempt to "delete" by marking the key as EMPTY_KEY.
                // We use atomicCAS to prevent races if multiple deletes hit the same stash entry.
                uint32_t old_stash_key = atomicCAS(&stash->entries[i].key, key, EMPTY_KEY);
                if (old_stash_key == key) {
                    delete_success = true;
                    break;
                }
            }
        }
    }
    
    delete_success = __shfl_sync(0xFFFFFFFFu, delete_success, 0);
    return delete_success;
}

static __global__ void warp_delete_kernel(
    BucketTable table,
    StashQueue* stash,
    const uint32_t* keys,
    uint32_t* deleted_flags,
    uint32_t num_keys) {
    
    uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (key_idx >= num_keys) return;

    uint32_t key = keys[key_idx];
    if (key == EMPTY_KEY) {
        if ((threadIdx.x % 32) == 0) deleted_flags[key_idx] = 0;
        return;
    }
    
    uint8_t fp = compute_hash_pair(key, table.bucket_mask).fingerprint;
    bool success = warp_delete_device(table, stash, key, fp);

    if ((threadIdx.x % 32) == 0) {
        deleted_flags[key_idx] = success ? 1 : 0;
    }
}

#endif // __CUDACC__

struct DeleteBatch {
    const uint32_t* h_keys;   // Host input: keys
    uint32_t* h_deleted;      // Host output: 1 if deleted, 0 if not found
    uint32_t num_keys;
};

// Launch deletion kernel for a batch of keys (Synchronous/Test Wrapper)
void warp_delete_batch_sync(
    BucketTable table,
    StashQueue* d_stash,
    const DeleteBatch& batch,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
