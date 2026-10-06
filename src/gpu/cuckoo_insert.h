// =============================================================================
// cuckoo_insert.h — Cuckoo insertion kernel (host-side wrappers)
// =============================================================================
// The device-side insertion logic (warp_insert_device, InsertStatus,
// InsertResult) has been consolidated into warpkv_device.cuh. This header
// re-exports those symbols and retains the host-side batch struct, the global
// kernel entry point, and the synchronous test wrapper declaration.
// =============================================================================

#pragma once

#include "../../include/warpkv/warpkv_device.cuh"
#include <cuda_runtime.h>

namespace warpkv {

#ifdef __CUDACC__

// ============================================================================
// Cuckoo Insertion Kernel (batch entry point)
// ============================================================================
// One warp (32 threads) processes one insertion:
//   1. Lanes 0-7:   attempt to claim a free slot in bucket b1
//   2. Lanes 8-15:  attempt to claim a free slot in bucket b2
//   3. Lanes 16-31: idle during slot claims
//   4. Lane 0:      selects and locks eviction victim when both buckets are full
//   5. Lane 0:      writes to stash after MAX_EVICTION_HOPS
//
// Lane 0 writes the InsertStatus and hop count to global memory.
// ============================================================================
static __global__ void warp_insert_kernel(
    BucketTable          table,
    StashQueue*          stash,
    uint32_t*            d_needs_rehash_flag,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ values,
    InsertStatus*        statuses,
    uint32_t*            hops,
    uint32_t             num_keys)
{
    const uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (key_idx >= num_keys) return;

    const uint32_t key = keys[key_idx];
    if (key == EMPTY_KEY) return; // Key 0 is reserved and cannot be inserted.

    const uint32_t value = values[key_idx];
    const uint8_t  fp    = compute_hash_pair(key, table.bucket_mask).fingerprint;

    const InsertResult result =
        warp_insert_device(table, stash, d_needs_rehash_flag, key, value, fp);

    if ((threadIdx.x % 32) == 0) {
        statuses[key_idx] = result.status;
        if (hops) hops[key_idx] = result.hops;
    }
}

#endif // __CUDACC__

// ============================================================================
// Host-side batch descriptor (used by test wrappers and the engine)
// ============================================================================

struct InsertBatch {
    uint32_t*     h_keys;     ///< Host input: keys to insert
    uint32_t*     h_values;   ///< Host input: corresponding values
    InsertStatus* h_statuses; ///< Host output: per-key InsertStatus
    uint32_t*     h_hops;     ///< Host output: eviction hop count (may be nullptr)
    uint32_t      num_keys;
};

// ============================================================================
// Synchronous test wrapper
// ============================================================================
// NOTE: Allocates and frees device memory on every call. Unit tests only.
// Production code (WarpKVEngine) uses CUDA Graphs with pre-allocated buffers.
// ============================================================================
void warp_insert_batch_sync(
    BucketTable          table,
    StashQueue*          d_stash,
    uint32_t*            d_needs_rehash_flag,
    const InsertBatch&   batch,
    cudaStream_t         stream = nullptr);

} // namespace warpkv
