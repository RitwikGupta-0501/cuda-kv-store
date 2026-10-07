// =============================================================================
// warpkv_entry.h — C-linkage host declarations for PTX entry kernels
// =============================================================================
//
// This header declares the extern "C" __global__ kernels defined in
// warpkv_entry_kernels.cu for consumers using the CUDA Driver API directly.
//
// For Python / Rust consumers loading the .ptx file at runtime, use the
// function names:
//   "warpkv_lookup_kernel_c"
//   "warpkv_insert_kernel_c"
//   "warpkv_delete_kernel_c"
//
// See warpkv_entry_kernels.cu for full parameter documentation.
// =============================================================================

#pragma once

#include "warpkv_device.cuh"
#include <cuda_runtime.h>

#ifdef __CUDACC__

extern "C" __global__ void warpkv_lookup_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    const warpkv::KeyT* __restrict__ d_keys,
    warpkv::ValueT*              d_values_out,
    uint32_t*                    d_found,
    uint32_t                     num_keys);

extern "C" __global__ void warpkv_insert_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    uint32_t*                    d_needs_rehash_flag,
    const warpkv::KeyT* __restrict__ d_keys,
    const warpkv::ValueT* __restrict__ d_values,
    uint32_t*                    d_statuses_out,
    uint32_t                     num_keys);

extern "C" __global__ void warpkv_delete_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    const warpkv::KeyT* __restrict__ d_keys,
    uint32_t*                    d_deleted_out,
    uint32_t                     num_keys);

#endif // __CUDACC__
