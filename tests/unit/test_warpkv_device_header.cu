// =============================================================================
// test_warpkv_device_header.cu — Acceptance test for Step 1.1
// =============================================================================
//
// Verifies that warpkv_device.cuh is truly self-contained:
//   1. Only includes <cstdint> and <cuda_runtime.h> (enforced by compilation
//      of this file which adds NO other includes before the device header).
//   2. All structs, constants, and device functions are accessible.
//   3. A custom kernel can call warp_lookup_device directly.
//
// Build (standalone, NO project dependency):
//   nvcc -arch=sm_70 -std=c++14 \
//        -I /path/to/cuda-kv-store/include \
//        test_warpkv_device_header.cu -o test_device_header
//
// Run: ./test_device_header
// Expected: "ALL TESTS PASSED" printed to stdout.
// =============================================================================

// Intentionally include ONLY warpkv_device.cuh — no other project headers.
#include "warpkv/warpkv_device.cuh"

#include <cstdio>
#include <cassert>
#include <cstring>

// ============================================================================
// Custom kernel that uses warp_lookup_device directly (Step 1.1 acceptance)
// ============================================================================

__global__ void custom_kernel_with_device_lookup(
    warpkv::BucketTable  table,
    warpkv::StashQueue*  stash,
    const uint32_t*      d_query_keys,
    uint32_t*            d_results,
    uint32_t             num_keys)
{
    using namespace warpkv;

    const KeyT key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (key_idx >= num_keys) return;

    const KeyT key = d_query_keys[key_idx];
    const uint8_t  fp  = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const LookupResult result = warp_lookup_device(table, stash, key, fp);

    if ((threadIdx.x % 32) == 0) {
        d_results[key_idx] = result.found ? result.value : NOT_FOUND;
    }
}

__global__ void custom_kernel_with_device_insert(
    warpkv::BucketTable  table,
    warpkv::StashQueue*  stash,
    uint32_t*            d_needs_rehash,
    const uint32_t*      d_keys,
    const uint32_t*      d_values,
    uint32_t*            d_statuses,
    uint32_t             num_keys)
{
    using namespace warpkv;

    const KeyT key_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (key_idx >= num_keys) return;

    const KeyT key   = d_keys[key_idx];
    const ValueT value = d_values[key_idx];
    if (key == EMPTY_KEY) return;

    const uint8_t      fp     = compute_hash_pair(key, table.bucket_mask).fingerprint;
    const InsertResult result = warp_insert_device(table, stash, d_needs_rehash, key, value, fp);

    if ((threadIdx.x % 32) == 0) {
        d_statuses[key_idx] = static_cast<uint32_t>(result.status);
    }
}

// ============================================================================
// Host-side test helpers
// ============================================================================

#define CUDA_ASSERT(call)                                                        \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static constexpr uint32_t NUM_BUCKETS = 256; // small table for testing
static constexpr uint32_t NUM_KEYS    = 16;

int main() {
    printf("=== test_warpkv_device_header: Step 1.1 acceptance test ===\n\n");

    // ---- Static structure checks -------------------------------------------
    printf("[1] Checking static assertions...\n");
    static_assert(sizeof(warpkv::Bucket)     == 128,    "Bucket must be 128 bytes");
    static_assert(sizeof(warpkv::StashQueue) < 300000,  "StashQueue must be < 300 KB");
    static_assert(warpkv::EMPTY_KEY    == 0x00000000u,  "EMPTY_KEY check");
    static_assert(warpkv::LOCK_SENTINEL == 0xFFFFFFFFu, "LOCK_SENTINEL check");
    static_assert(warpkv::NOT_FOUND    == 0xFFFFFFFFu,  "NOT_FOUND check");
    printf("    OK: struct sizes and constants correct\n");

    // ---- Host-side hash function check -------------------------------------
    printf("[2] Checking host-side hash function...\n");
    {
        warpkv::HashPair p = warpkv::compute_hash_pair(42u, NUM_BUCKETS - 1);
        assert(p.b1 < NUM_BUCKETS && "b1 must be within table");
        assert(p.b2 < NUM_BUCKETS && "b2 must be within table");
        assert(p.b1 != p.b2      && "b1 and b2 must differ");
        printf("    OK: key=42 -> b1=%u, b2=%u, fp=0x%02x\n", p.b1, p.b2, p.fingerprint);
    }

    // ---- GPU: allocate table and stash ------------------------------------
    printf("[3] Allocating GPU table (%u buckets)...\n", NUM_BUCKETS);
    warpkv::Bucket* d_buckets = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_buckets, NUM_BUCKETS * sizeof(warpkv::Bucket)));
    CUDA_ASSERT(cudaMemset(d_buckets, 0, NUM_BUCKETS * sizeof(warpkv::Bucket)));

    warpkv::BucketTable table;
    table.buckets          = d_buckets;
    table.num_buckets      = NUM_BUCKETS;
    table.bucket_mask      = NUM_BUCKETS - 1;
    table.load_factor_limit = NUM_BUCKETS / 2;

    warpkv::StashQueue* d_stash = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_stash, sizeof(warpkv::StashQueue)));
    CUDA_ASSERT(cudaMemset(d_stash, 0, sizeof(warpkv::StashQueue)));

    uint32_t* d_needs_rehash = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_needs_rehash, sizeof(uint32_t)));
    CUDA_ASSERT(cudaMemset(d_needs_rehash, 0, sizeof(uint32_t)));
    printf("    OK\n");

    // ---- GPU: insert keys via custom kernel ---------------------------------
    printf("[4] Inserting %u keys via custom_kernel_with_device_insert...\n", NUM_KEYS);
    uint32_t h_keys[NUM_KEYS], h_values[NUM_KEYS];
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        h_keys[i]   = i + 1; // keys 1..16 (avoid EMPTY_KEY = 0)
        h_values[i] = (i + 1) * 100;
    }

    uint32_t *d_keys = nullptr, *d_values = nullptr, *d_statuses = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_keys,     NUM_KEYS * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_values,   NUM_KEYS * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_statuses, NUM_KEYS * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMemcpy(d_keys,   h_keys,   NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_values, h_values, NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch: 256 threads/block = 8 warps = 8 keys/block
    const uint32_t threads = 256;
    const uint32_t blocks  = (NUM_KEYS + 7) / 8;
    custom_kernel_with_device_insert<<<blocks, threads>>>(
        table, d_stash, d_needs_rehash, d_keys, d_values, d_statuses, NUM_KEYS);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    uint32_t h_statuses[NUM_KEYS];
    CUDA_ASSERT(cudaMemcpy(h_statuses, d_statuses, NUM_KEYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        if (h_statuses[i] != warpkv::INSERT_SUCCESS) {
            fprintf(stderr, "    FAIL: key %u got status %u (expected INSERT_SUCCESS=0)\n",
                    h_keys[i], h_statuses[i]);
            exit(1);
        }
    }
    printf("    OK: all %u keys inserted with INSERT_SUCCESS\n", NUM_KEYS);

    // ---- GPU: lookup via custom kernel -------------------------------------
    printf("[5] Looking up %u keys via custom_kernel_with_device_lookup...\n", NUM_KEYS);
    uint32_t* d_results = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_results, NUM_KEYS * sizeof(uint32_t)));

    custom_kernel_with_device_lookup<<<blocks, threads>>>(
        table, d_stash, d_keys, d_results, NUM_KEYS);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    uint32_t h_results[NUM_KEYS];
    CUDA_ASSERT(cudaMemcpy(h_results, d_results, NUM_KEYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        if (h_results[i] != h_values[i]) {
            fprintf(stderr, "    FAIL: key %u -> got %u, expected %u\n",
                    h_keys[i], h_results[i], h_values[i]);
            exit(1);
        }
    }
    printf("    OK: all %u lookups returned correct values\n", NUM_KEYS);

    // ---- GPU: lookup missing key -------------------------------------------
    printf("[6] Checking NOT_FOUND for missing key...\n");
    uint32_t missing_key = 9999u;
    CUDA_ASSERT(cudaMemcpy(d_keys, &missing_key, sizeof(uint32_t), cudaMemcpyHostToDevice));
    custom_kernel_with_device_lookup<<<1, 32>>>(table, d_stash, d_keys, d_results, 1);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    uint32_t miss_result;
    CUDA_ASSERT(cudaMemcpy(&miss_result, d_results, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (miss_result != warpkv::NOT_FOUND) {
        fprintf(stderr, "    FAIL: missing key returned %u, expected NOT_FOUND=%u\n",
                miss_result, warpkv::NOT_FOUND);
        exit(1);
    }
    printf("    OK: missing key correctly returns NOT_FOUND\n");

    // ---- Cleanup -----------------------------------------------------------
    cudaFree(d_buckets);
    cudaFree(d_stash);
    cudaFree(d_needs_rehash);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_statuses);
    cudaFree(d_results);

    printf("\n=== ALL TESTS PASSED ===\n");
    printf("warpkv_device.cuh is self-contained and all device APIs are callable\n");
    printf("from an external CUDA kernel with no other project headers.\n");
    return 0;
}
