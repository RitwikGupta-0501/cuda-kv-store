// =============================================================================
// cuckoo_delete.h — Cuckoo deletion kernel (host-side wrappers)
// =============================================================================
// The device-side deletion logic (warp_delete_device) has been consolidated
// into warpkv_device.cuh. This header re-exports those symbols and retains
// the host-side batch struct, the global kernel entry point, and the
// synchronous test wrapper declaration.
// =============================================================================

#pragma once

#include "../../include/warpkv/warpkv_device.cuh"
#include <cuda_runtime.h>

namespace warpkv {

#ifdef __CUDACC__

// ============================================================================
// Cuckoo Deletion Kernel (batch entry point)
// ============================================================================
// One warp (32 threads) processes one deletion:
//   Lanes  0-7:  scan b1 for the key
//   Lanes  8-15: scan b2 for the key in parallel
//   Lanes 16-31: idle during bucket scan
//   Winner lane: locks the slot and clears the entry atomically
//   Lane 0:      scans the stash linearly if key not found in buckets
//
// Lane 0 writes the deleted flag to global memory.
//
// Known limitation: stash scan uses only lane 0 (31 threads idle).
// This will be addressed in Phase 3 (warp utilization fix).
// ============================================================================
static __global__ void warp_delete_kernel(
    BucketTable          table,
    StashQueue*          stash,
    const uint32_t* __restrict__ keys,
    uint32_t*            deleted_flags,
    uint32_t             num_keys)
{
    const uint32_t key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (key_idx >= num_keys) return;

    const uint32_t key = keys[key_idx];
    if (key == EMPTY_KEY) {
        if ((threadIdx.x % 32) == 0) deleted_flags[key_idx] = 0;
        return;
    }

    const uint8_t fp      = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const bool    success = warp_delete_device(table, stash, key, fp);

    if ((threadIdx.x % 32) == 0) {
        deleted_flags[key_idx] = success ? 1u : 0u;
    }
}

#endif // __CUDACC__

// ============================================================================
// Host-side batch descriptor (used by test wrappers and the engine)
// ============================================================================

struct DeleteBatch {
    const uint32_t* h_keys;    ///< Host input: keys to delete
    uint32_t*       h_deleted; ///< Host output: 1 if deleted, 0 if not found
    uint32_t        num_keys;
};

// ============================================================================
// Synchronous test wrapper
// ============================================================================
// NOTE: Allocates and frees device memory on every call. Unit tests only.
// ============================================================================
void warp_delete_batch_sync(
    BucketTable         table,
    StashQueue*         d_stash,
    const DeleteBatch&  batch,
    cudaStream_t        stream = nullptr);

} // namespace warpkv
