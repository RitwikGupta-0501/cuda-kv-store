# WarpKV Implementation Tasks

## Phase 1: Device API Extraction & Async Foundation
*Started: 2026-10-07*

### Step 1.1 — Extract `warpkv_device.cuh` (self-contained header)
- [x] Create `include/warpkv/warpkv_device.cuh`
  - [x] Move `Bucket`, `BucketTable`, `StashEntry`, `StashQueue` structs
  - [x] Move all constants (`EMPTY_KEY`, `LOCK_SENTINEL`, `NOT_FOUND`, `BATCH_SIZE`, etc.)
  - [x] Move `bucket_init()`, `bucket_is_occupied()`, `bucket_set_occupied()`, `bucket_clear_occupied()`
  - [x] Move `warpkv_hash32()`, `warpkv_hash32_host()`, `compute_hash_pair()`
  - [x] Move `warp_lookup_device()` with all dependencies
  - [x] Move `warp_insert_device()` with all dependencies
  - [x] Move `warp_delete_device()` with all dependencies
  - [x] Verify: only `<cstdint>` and `<cuda_runtime.h>` as dependencies
- [x] Refactor existing headers to thin wrappers over `warpkv_device.cuh`
  - [x] `src/gpu/bucket_cuckoo.h`
  - [x] `src/gpu/hash.h`
  - [x] `src/gpu/warp_lookup.h`
  - [x] `src/gpu/cuckoo_insert.h`
  - [x] `src/gpu/cuckoo_delete.h`
- [x] Update `CMakeLists.txt` to add `include/` directory
- [x] Acceptance test: standalone `.cu` file includes only `warpkv_device.cuh` and compiles (`tests/unit/test_warpkv_device_header.cu`)

### Step 1.2 — PTX module with `extern "C" __global__` entry points
- [x] Create `src/gpu/warpkv_entry_kernels.cu`
  - [x] `extern "C" __global__ void warpkv_lookup_kernel_c(...)`
  - [x] `extern "C" __global__ void warpkv_insert_kernel_c(...)`
  - [x] `extern "C" __global__ void warpkv_delete_kernel_c(...)`
- [x] Create `include/warpkv/warpkv_entry.h` — C-linkage host declarations
- [x] Add CMake target: `warpkv_device_ptx` (nvcc --ptx)
- [x] Verification: PTX generation and entry points checked in test harness (`scripts/run_phase1_tests.sh`)

### Step 1.3 — Remove blocking syncs / enable async pipeline
- [x] Remove `cudaStreamSynchronize` at **end** of `submit_insert_batch`
- [x] Remove `cudaStreamSynchronize` at **end** of `submit_lookup_batch`
- [x] Remove `cudaStreamSynchronize` at **end** of `submit_delete_batch`
- [x] Change submit functions to return `std::future<void>` and `std::future<LookupFutureResult>`
- [x] Maintain backwards-compatible buffer overloads: `submit_lookup_batch(keys, values_out, count)`
- [x] Add blocking synchronous wrappers (`submit_*_batch_sync`)
- [x] Update Python binding to call `.get()` via synchronous wrappers (blocking behavior preserved at binding layer)
- [x] Acceptance test: async overlap, futures, and sync wrappers verified in `tests/unit/test_async_pipeline.cu`
- [x] Verification test script: `scripts/run_phase1_tests.sh` with full suite automation

---

## Phase 2 & 3: 64-bit Type System + Half-Warp Optimization
*Status: IN PROGRESS*

### Sub-Phase 2.1: Type Abstraction (Plumbing)
- [x] Define `KeyT` and `ValueT` in `warpkv_device.cuh` (default to `uint32_t`).
- [x] Refactor `Bucket`, `StashEntry`, and constants to use `KeyT`/`ValueT`.
- [x] Refactor GPU kernel signatures and host batch structs.
- [x] Refactor `WarpKVEngine` buffers and submit API.
- [x] Update tests to use abstract types.
- [x] Verify: Tests pass with exactly identical binary behavior.

### Sub-Phase 2.2: 64-bit Hash Function Upgrade
- [x] Implement `warpkv_hash64` (64-bit avalanche/mixer).
- [x] Update `compute_hash_pair` to consume 64-bit hash.
- [x] Verify: Distribution tests and correctness tests pass.

### Sub-Phase 2.3: 7-Slot Layout & 64-bit Switch
- [x] Switch `KeyT = uint64_t` and `ValueT = uint64_t`.
- [x] Redesign `Bucket` to 7 slots (128 bytes total).
- [x] Update bucket scan loops from 8 to 7 slots.
- [x] Update Python bindings to `uint64`.
- [x] Verify: 64-bit keys/values functional, cache-line alignment preserved.

### Sub-Phase 3.1: Half-Warp Compute Optimization
- [x] Change dispatch: 2 Keys per 32-thread Warp.
- [x] Lanes 0-15 process Key A, Lanes 16-31 process Key B.
- [x] Update lane indexing math in lookup, insert, and delete.
- [x] Verify: 87.5% compute utilization, basic functionality passes.

### Sub-Phase 3.2: Sub-Warp Synchronization & Stash Polish
- [x] Update `__ballot_sync` / `__shfl_sync` masks to `0x0000FFFF` and `0xFFFF0000`.
- [x] Parallelize stash scan cooperatively across 16 threads.
- [x] Verify: `test_engine_concurrent` passes (no deadlocks or races).

---

## Phase 4: PyTorch Integration
*Status: PENDING (blocked on Phase 2+3)*

## Phase 5: Async CPU API & Dynamic Batching
*Status: PENDING (blocked on Phase 2+3)*
