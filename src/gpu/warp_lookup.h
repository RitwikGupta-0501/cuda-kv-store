// =============================================================================
// warp_lookup.h — Warp-cooperative lookup kernel (host-side wrappers)
// =============================================================================
// The device-side lookup logic (warp_lookup_device, LookupResult) has been
// consolidated into warpkv_device.cuh. This header re-exports those symbols
// and retains the host-side batch struct and synchronous test wrapper.
// =============================================================================

#pragma once

#include "../../include/warpkv/warpkv_device.cuh"
#include <cuda_runtime.h>

namespace warpkv {

#ifdef __CUDACC__

// ============================================================================
// Warp-Cooperative Lookup Kernel (batch entry point)
// ============================================================================
// One warp (32 threads) processes one key:
//   Lanes  0-7:  scan primary bucket b1
//   Lanes  8-15: scan secondary bucket b2 in parallel
//   Lanes 16-31: idle during bucket scan; cooperative during stash scan
//
// Lane 0 writes the result to global memory.
// ============================================================================
static __global__ void warp_lookup_kernel(
    BucketTable  table,
    StashQueue*  stash,
    const KeyT* __restrict__ keys,
    ValueT*      values,
    uint32_t*    found_flags,
    uint32_t     num_keys)
{
    const uint32_t key_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    if (key_idx >= num_keys) return;

    const KeyT key = keys[key_idx];
    if (key == EMPTY_KEY) {
        if ((threadIdx.x % 16) == 0) {
            values[key_idx]      = NOT_FOUND;
            found_flags[key_idx] = 0;
        }
        return;
    }

    const uint8_t fp = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const LookupResult result = warp_lookup_device(table, stash, key, fp);

    if ((threadIdx.x % 16) == 0) {
        values[key_idx]      = result.value;
        found_flags[key_idx] = result.found ? 1u : 0u;
    }
}

#endif // __CUDACC__

// ============================================================================
// Host-side batch descriptor (used by test wrappers and the engine)
// ============================================================================

struct LookupBatch {
    KeyT*     h_keys;   ///< Host input: keys to look up
    ValueT*   h_values; ///< Host output: found values (NOT_FOUND on miss)
    uint32_t* h_found;  ///< Host output: 1 if found, 0 if not found
    uint32_t  num_keys;
};

// ============================================================================
// Synchronous test wrapper
// ============================================================================
// NOTE: This wrapper allocates and frees device memory on every call.
// It is intended for unit tests only. Production code (WarpKVEngine) uses
// CUDA Graphs with pre-allocated device buffers.
// ============================================================================
void warp_lookup_batch_sync(
    BucketTable         table,
    StashQueue*         d_stash,
    const LookupBatch&  batch,
    cudaStream_t        stream = nullptr);

} // namespace warpkv
