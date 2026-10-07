// =============================================================================
// warpkv_entry_kernels.cu — extern "C" __global__ entry points for PTX export
// =============================================================================
//
// PURPOSE:
//   These thin wrappers expose WarpKV kernels with C linkage and no C++ name
//   mangling, making them loadable by foreign runtimes that consume PTX:
//     - Python (Numba via cuModuleLoad / cuLaunchKernel)
//     - Python (CuPy via cupy.RawModule)
//     - Rust  (Rust-CUDA via cust::module::Module)
//     - C     (Any host code using the CUDA Driver API directly)
//
// HOW TO BUILD THE PTX MODULE:
//   nvcc -arch=sm_70 --ptx -o warpkv_kernels.ptx \
//        src/gpu/warpkv_entry_kernels.cu          \
//        -I include
//
// HOW TO LOAD IN PYTHON (Numba example):
//   from numba import cuda
//   module = cuda.driver.CudaModule("warpkv_kernels.ptx")
//   fn = module.get_function("warpkv_insert_kernel_c")
//
// HOW TO LOAD IN PYTHON (CuPy example):
//   import cupy as cp
//   with open("warpkv_kernels.ptx") as f: ptx = f.read()
//   module = cp.RawModule(code=ptx, backend='ptx')
//   fn = module.get_function("warpkv_lookup_kernel_c")
//
// KERNEL CONTRACTS:
//   - Grid/block configuration: 1 warp (32 threads) per key.
//     Launch with: grid=(num_keys/8, 1, 1), block=(256, 1, 1)  [8 keys/block]
//   - Key 0 (EMPTY_KEY = 0x00000000) is reserved — cannot be inserted/deleted.
//   - Key 0xFFFFFFFF (LOCK_SENTINEL) is reserved — cannot be inserted/deleted.
//   - All device pointers must be allocated with cudaMalloc or equivalent.
//
// PARAMETER LAYOUTS:
//   Lookup:
//     table        — BucketTable struct (24 bytes, passed by value)
//     d_stash      — StashQueue* device pointer (nullable)
//     d_keys       — const uint32_t* [num_keys] — input keys
//     d_values_out — uint32_t*       [num_keys] — output values (NOT_FOUND=0xFFFFFFFF on miss)
//     d_found      — uint32_t*       [num_keys] — output flags (1=found, 0=miss)
//     num_keys     — uint32_t
//
//   Insert:
//     table               — BucketTable
//     d_stash             — StashQueue*
//     d_needs_rehash_flag — uint32_t* mapped host flag (set to 1 when rehash needed)
//     d_keys              — const uint32_t* [num_keys]
//     d_values            — const uint32_t* [num_keys]
//     d_statuses_out      — uint32_t*       [num_keys] — InsertStatus (0=success,1=stashed,2=failed)
//     num_keys            — uint32_t
//
//   Delete:
//     table          — BucketTable
//     d_stash        — StashQueue*
//     d_keys         — const uint32_t* [num_keys]
//     d_deleted_out  — uint32_t*       [num_keys] — 1=deleted, 0=not found
//     num_keys       — uint32_t
//
// =============================================================================

#include "warpkv/warpkv_device.cuh"

// =============================================================================
// Lookup entry point
// =============================================================================

extern "C" __global__ void warpkv_lookup_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    const warpkv::KeyT* __restrict__ d_keys,
    warpkv::ValueT*              d_values_out,
    uint32_t*                    d_found,
    uint32_t                     num_keys)
{
    using namespace warpkv;

    const uint32_t key_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    if (key_idx >= num_keys) return;

    const KeyT key = d_keys[key_idx];
    if (key == EMPTY_KEY) {
        if ((threadIdx.x % 16) == 0) {
            d_values_out[key_idx] = NOT_FOUND;
            d_found[key_idx]      = 0u;
        }
        return;
    }

    const uint8_t      fp     = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const LookupResult result = warp_lookup_device(table, d_stash, key, fp);

    if ((threadIdx.x % 16) == 0) {
        d_values_out[key_idx] = result.value;
        d_found[key_idx]      = result.found ? 1u : 0u;
    }
}

// =============================================================================
// Insert entry point
// =============================================================================

extern "C" __global__ void warpkv_insert_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    uint32_t*                    d_needs_rehash_flag,
    const warpkv::KeyT* __restrict__ d_keys,
    const warpkv::ValueT* __restrict__ d_values,
    uint32_t*                    d_statuses_out,
    uint32_t                     num_keys)
{
    using namespace warpkv;

    const uint32_t key_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    if (key_idx >= num_keys) return;

    const KeyT key = d_keys[key_idx];
    if (key == EMPTY_KEY) return; // reserved key — silently skip

    const ValueT value  = d_values[key_idx];
    const uint8_t  fp     = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const InsertResult result =
        warp_insert_device(table, d_stash, d_needs_rehash_flag, key, value, fp);

    if ((threadIdx.x % 16) == 0) {
        d_statuses_out[key_idx] = static_cast<uint32_t>(result.status);
    }
}

// =============================================================================
// Delete entry point
// =============================================================================

extern "C" __global__ void warpkv_delete_kernel_c(
    warpkv::BucketTable          table,
    warpkv::StashQueue*          d_stash,
    const warpkv::KeyT* __restrict__ d_keys,
    uint32_t*                    d_deleted_out,
    uint32_t                     num_keys)
{
    using namespace warpkv;

    const uint32_t key_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    if (key_idx >= num_keys) return;

    const uint32_t key = d_keys[key_idx];
    if (key == EMPTY_KEY) {
        if ((threadIdx.x % 16) == 0) d_deleted_out[key_idx] = 0u;
        return;
    }

    const uint8_t fp      = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const bool    success = warp_delete_device(table, d_stash, key, fp);

    if ((threadIdx.x % 16) == 0) {
        d_deleted_out[key_idx] = success ? 1u : 0u;
    }
}
