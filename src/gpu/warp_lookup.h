#pragma once

#include "xxhash3.h"
#include "bucket_cuckoo.h"
#include <cuda_runtime.h>

namespace warpkv {

// ============================================================================
// Warp-Cooperative Lookup Kernel
// ============================================================================
// One warp processes one key:
// - Lanes 0-7: scan bucket b1 (hash[0])
// - Lanes 8-15: scan bucket b2 (hash[1])
// - Parallel filtering via fingerprint comparison before key comparison
// - Returns value on hit, NOT_FOUND on miss
// ============================================================================

struct LookupResult {
    uint32_t value;
    bool found;
};

// Device-side lookup function (called by kernel)
__device__ inline LookupResult warp_lookup_device(
    const BucketTable* table,
    uint32_t key,
    uint8_t fingerprint) {

    // Compute both bucket addresses
    HashPair hash_pair = compute_hash_pair(key, table->bucket_mask);
    Bucket* bucket_b1 = &table->buckets[hash_pair.b1];
    Bucket* bucket_b2 = &table->buckets[hash_pair.b2];

    uint32_t lane_id = threadIdx.x % 32;
    uint32_t slot_id = lane_id;  // Each lane scans one slot in parallel

    LookupResult result = {NOT_FOUND, false};

    // ========== Lanes 0-7: Scan bucket b1 ==========
    if (lane_id < 8) {
        // Check if slot is occupied
        if (bucket_b1->occupancy_mask & (1u << slot_id)) {
            // Fast path: check fingerprint first
            if (bucket_b1->fingerprint[slot_id] == fingerprint) {
                // Fingerprint match: verify actual key
                if (bucket_b1->keys[slot_id] == key) {
                    result.value = bucket_b1->values[slot_id];
                    result.found = true;
                }
            }
        }
    }
    // ========== Lanes 8-15: Scan bucket b2 in parallel ==========
    else if (lane_id < 16) {
        uint32_t b2_slot = lane_id - 8;

        // Check if slot is occupied
        if (bucket_b2->occupancy_mask & (1u << b2_slot)) {
            // Fast path: check fingerprint first
            if (bucket_b2->fingerprint[b2_slot] == fingerprint) {
                // Fingerprint match: verify actual key
                if (bucket_b2->keys[b2_slot] == key) {
                    result.value = bucket_b2->values[b2_slot];
                    result.found = true;
                }
            }
        }
    }

    // Broadcast result from whichever lane found it
    // Use warp shuffle to combine results: if any lane found it, that value wins
    int found_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.found)) - 1;
    if (found_lane >= 0) {
        result.value = __shfl_sync(0xFFFFFFFFu, result.value, found_lane);
    }

    return result;
}

// Kernel: process a batch of lookup keys
// Assumes one warp per key (32 threads per key)
__global__ void warp_lookup_kernel(
    const BucketTable* table,
    const uint32_t* keys,
    uint32_t* values,
    uint32_t* found_flags,
    uint32_t num_keys) {

    // Each warp processes one key
    uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);

    if (key_idx >= num_keys) return;

    uint32_t key = keys[key_idx];
    uint8_t fp = compute_hash_pair(key, 0).fingerprint;

    LookupResult result = warp_lookup_device(table, key, fp);

    // Lane 0 writes result
    if ((threadIdx.x % 32) == 0) {
        values[key_idx] = result.value;
        found_flags[key_idx] = result.found ? 1u : 0u;
    }
}

// Host-side wrapper for lookup batches
struct LookupBatch {
    uint32_t* h_keys;         // Host input: keys
    uint32_t* h_values;       // Host output: values
    uint32_t* h_found;        // Host output: found flags (0=not found, 1=found)
    uint32_t num_keys;
};

// Launch lookup kernel for a batch of keys
void warp_lookup_batch(
    const BucketTable* d_table,
    const LookupBatch& batch,
    cudaStream_t stream = nullptr);

}  // namespace warpkv
