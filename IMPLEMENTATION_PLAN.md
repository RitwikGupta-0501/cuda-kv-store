# WarpKV v2 — Production Implementation Plan

**Status:** Complete technical specification, misaligned initial codebase  
**Target:** SOTA-level GPU-accelerated KV store on NVIDIA MX130  
**Deadline:** Production ready with benchmarks  

---

## Executive Summary

The current implementation uses:
- ❌ MurmurHash3 (spec requires XXHash3)
- ❌ Per-slot naive cuckoo (spec requires bucket cuckoo, 128B cache-line aligned)
- ❌ CPU-only operations (no GPU kernels)
- ❌ No CUDA Graphs, no EBR, no pybind11 binding

**Plan scope:** 7 implementation phases + comprehensive testing + competitive benchmarking.  
**Estimated effort:** ~400–600 engineering hours for 1–2 FTE.  
**Critical path:** Phases 1–4 (kernels) → Phase 5 (EBR) → Phase 6 (pipeline) → Phase 7 (Python + benchmarks).

---

## Phase 0: Foundation & Validation (Week 1)

**Goal:** Validate hardware, build toolchain, establish test infrastructure.

### Tasks

1. **CUDA Compute Capability validation**
   - Verify MX130 detection: `nvidia-smi`, `nvcc -V`
   - Test PCIe bandwidth measurement with a simple memory bandwidth benchmark
   - Baseline: measure actual vs. spec ~6 GB/s bidirectional effective bandwidth
   - Document in `docs/hardware_baseline.md`

2. **Build system setup**
   - CMakeLists.txt: organize source into `src/core/`, `src/gpu/`, `src/python/`
   - CUDA compute capability flags: `-arch=sm_50` (CC 5.0)
   - C++17 standard, CUDA 11.0+ (or latest available)
   - Unit test framework: Google Test (gtest)

3. **Test infrastructure**
   - Create `tests/unit/` for kernel unit tests
   - Create `tests/integration/` for end-to-end tests
   - Benchmark harness skeleton in `benchmarks/`
   - CI/CD: basic `cmake && make test` workflow

4. **Documentation**
   - `docs/architecture.md` — mirrors spec section I–VI
   - `docs/build.md` — compilation steps, dependencies, troubleshooting
   - `docs/testing.md` — test categories, running tests, interpreting results

### Deliverables
- Clean CMakeLists.txt with gtest integration
- Hardware baseline report (PCIe BW, memory latency)
- Test skeleton with first passing unit test
- Estimated hours: **40–60h**

---

## Phase 1: XXHash3 Kernel (Week 1–2)

**Goal:** Implement spec-compliant hash function, validate avalanche and bit distribution.

### Spec Requirements
- 32-bit finalizer variant for short keys ≤ 16 bytes (only 4B keys in v1)
- Power-of-two bucket indexing via bitmask (no modulo)
- Two independent hash positions: `b1 = h & mask`, `b2 = ((h >> 16) ^ 0xDEADBEEF) & mask`
- Verify `b1 != b2` property across 10M random keys (degenerate case prevention)

### Tasks

1. **Implement `src/gpu/xxhash3.cu`**
   ```cuda
   __device__ __forceinline__ uint32_t xxhash3_32(uint32_t key) {
       uint32_t h = key + 0x9E3779B9u;
       h ^= h >> 15;
       h *= 0x85EBCA77u;
       h ^= h >> 13;
       h *= 0xC2B2AE3Du;
       h ^= h >> 16;
       return h;
   }
   ```

2. **Unit tests: `tests/unit/test_xxhash3.cpp`**
   - ✓ Known key→hash mappings (golden values)
   - ✓ Avalanche property: flipping each input bit affects ~50% of output bits
   - ✓ No correlation between input and output bits (chi-square test)
   - ✓ `b1 != b2` property: sweep 10M random keys, verify no collisions when `b1 == b2`
   - ✓ Modulo elimination: verify bitmask is faster than `%` operator

3. **Benchmark: `benchmarks/hash_throughput.cu`**
   - Measure hash function throughput: keys/ns on 1M random keys
   - Compare vs. MurmurHash3 (current)
   - Expected result: XXHash3 ≥ 2.5× faster on short keys

### Deliverables
- `src/gpu/xxhash3.cu` (< 20 lines)
- `src/gpu/xxhash3.h` (header + host-side version for preprocessing)
- Unit tests: 100% pass rate, avalanche verified
- Benchmark report: `docs/benchmarks/xxhash3_vs_murmur.txt`
- Estimated hours: **20–30h**

---

## Phase 2: Bucket Cuckoo Data Structure & Allocator (Week 2–3)

**Goal:** Implement spec-compliant bucket layout, SoA allocation, power-of-two sizing.

### Spec Requirements
- **Bucket layout** (128 bytes = 1 cache line):
  - `keys[8]` — 32B (uint32_t × 8)
  - `values[8]` — 32B (uint32_t × 8)
  - `fingerprint[8]` — 8B (uint8_t × 8, upper 8 bits of hash)
  - `occupancy_mask` — 4B (uint32_t, one bit per slot)
  - Padding — 52B (unused)
- **SoA layout at table level:** keys, values, fingerprints, occupancy masks stored contiguously
- **Static arena allocation:** cudaMalloc once at startup, sub-allocate within arena
- **Load factor:** 0.50 (50% utilization for eviction chain bounds)
- **Max eviction hops:** 32
- **CPU stash:** 128 slots in pinned memory for un-placeable keys

### Tasks

1. **Implement `src/gpu/bucket_cuckoo.h`**
   ```cpp
   struct Bucket {
       uint32_t keys[8];
       uint32_t values[8];
       uint8_t fingerprint[8];
       uint32_t occupancy_mask;
       uint8_t padding[52];  // Total: 128 bytes
   };
   
   struct BucketTable {
       Bucket* buckets;           // SoA: all buckets
       uint32_t num_buckets;      // Power of 2
       uint32_t bucket_mask;      // num_buckets - 1 (for bitmask indexing)
       uint32_t load_factor_limit; // 0.5 × num_buckets for insertion cap
   };
   ```

2. **Implement `src/gpu/arena_allocator.cu`**
   - `WarpKVEngine::init()`: 
     - Query free VRAM, allocate 80% as arena
     - Round arena_size down to power-of-two bucket count
     - Sub-allocate: buckets, d_keys[3], d_vals[3]
     - Pre-allocate CPU stash (pinned, 128 slots)
   - Zero-copy sub-allocation (no runtime cudaMalloc)

3. **Host-side CPU table builder: `src/core/bucket_cuckoo_cpu.cpp`**
   - Build table on CPU for testing/preprocessing
   - Insert up to load_factor=0.5
   - Copy to GPU via cudaMemcpy

4. **Unit tests: `tests/unit/test_bucket_layout.cpp`**
   - ✓ Bucket struct is exactly 128 bytes
   - ✓ Occupancy mask bit operations (set, clear, test bit)
   - ✓ Fingerprint extraction from hash (upper 8 bits)
   - ✓ Arena allocation: correct bucket count (power of 2)
   - ✓ Bucket indexing via bitmask

5. **Unit tests: `tests/unit/test_arena_allocator.cpp`**
   - ✓ cudaMalloc called exactly once
   - ✓ Available buckets = arena_size / sizeof(Bucket), rounded down to power of 2
   - ✓ No buffer overrun on max capacity
   - ✓ Pinned memory allocated and accessible from device code

### Deliverables
- `src/gpu/bucket_cuckoo.h` + struct definitions
- `src/gpu/arena_allocator.cu` (allocator logic)
- `src/core/bucket_cuckoo_cpu.cpp` (CPU builder for testing)
- Unit tests: 100% pass rate
- Memory layout validation via `cuda-memcheck`
- Estimated hours: **50–80h**

---

## Phase 3: Single-Warp Lookup Kernel (Week 3–4)

**Goal:** Implement zero-divergence warp-cooperative lookup on 2 buckets per warp.

### Spec Requirements
- **One warp (32 threads) per key lookup**
  - Lanes 0–15 scan bucket b1
  - Lanes 16–31 scan bucket b2
- **Fingerprint fast-path:** `__ballot_sync` rejects non-matching fingerprints before value array read
- **Result aggregation:** `__ffs` (find first set) lowest-lane match wins
- **Lane mapping:** Correct `__shfl_sync` for result broadcast
- **Guaranteed 2-bucket lookup bound:** No probe chain, worst case = 2 cache-line reads

### Tasks

1. **Implement `src/gpu/warp_lookup.cu`**
   ```cuda
   __global__ void warp_lookup_kernel(
       const BucketTable* table,
       const uint32_t* d_keys,
       uint32_t* d_vals,
       uint32_t batch_size) {
       uint32_t lane = threadIdx.x & 31;
       uint32_t warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
       if (warp_id >= batch_size) return;
       
       uint32_t key = d_keys[warp_id];
       uint32_t h = xxhash3_32(key);
       uint32_t b1 = h & table->bucket_mask;
       uint32_t b2 = ((h >> 16) ^ 0xDEADBEEFu) & table->bucket_mask;
       uint8_t fp = (uint8_t)(h >> 24);
       
       const int slot = lane & 15;
       uint32_t bucket_idx = (lane < 16) ? b1 : b2;
       const Bucket* bkt = &table->buckets[bucket_idx];
       
       // Fingerprint fast-path
       bool hit = (bkt->occupancy_mask >> slot & 1u) &&
                  (bkt->fingerprint[slot] == fp) &&
                  (bkt->keys[slot] == key);
       
       uint32_t match_mask = __ballot_sync(0xFFFFFFFFu, hit);
       
       if (match_mask == 0u) {
           if (lane == 0) d_vals[warp_id] = WARPKV_NOT_FOUND;
           return;
       }
       
       int winner = __ffs(match_mask) - 1;
       uint32_t result = __shfl_sync(0xFFFFFFFFu, bkt->values[slot], winner);
       
       if (lane == 0) d_vals[warp_id] = result;
   }
   ```

2. **Unit tests: `tests/unit/test_warp_lookup.cpp`**
   - ✓ Single key: b1 hit, b2 hit (hardcoded small table)
   - ✓ Key not found → WARPKV_NOT_FOUND
   - ✓ Fingerprint collision, key mismatch correctly rejects
   - ✓ Full warp: 32 distinct keys in parallel, no cross-lane corruption
   - ✓ Load factor 0.5 edge case
   - ✓ `b1 == b2` degenerate case (XOR decorrelation prevents it)
   - ✓ Worst-case 2 cache-line reads verified via profiler

3. **Integration tests: `tests/integration/test_lookup_correctness.cpp`**
   - Insert 1000 keys CPU-side
   - Copy table to GPU
   - Lookup all 1000 keys, verify correctness
   - Lookup 1000 non-existent keys, verify all return NOT_FOUND

4. **Profiling**
   - NVIDIA Nsight: measure L2 hit rate, cache-line utilization
   - Verify ≤2 cache-line reads per lookup (worst case)

### Deliverables
- `src/gpu/warp_lookup.cu` (main kernel)
- Unit tests: 100% pass rate, no warp divergence on lookup path
- Integration tests: 100% pass rate
- Profiler output: cache behavior report
- Estimated hours: **60–100h** (most complex kernel)

---

## Phase 4: Warp-Cooperative Insert Kernel (Week 4–5)

**Goal:** Implement CAS-based insertion with eviction chain, stash fallback, load factor cap.

### Spec Requirements
- **Lane 0 drives insertion** (CAS atomic operations)
- **Warp broadcasts key/val via `__shfl_sync`** to all lanes
- **Max eviction hops:** 32 (bounded worst-case insertion time)
- **Max load factor:** 0.50 (eviction chains statistically short)
- **CPU stash:** 128 slots for unplaceable keys, triggers async rehash
- **Stash overflow action:** Full-table rehash on dedicated CUDA stream (visible via EBR in Phase 5)

### Tasks

1. **Implement `src/gpu/warp_insert.cu`**
   ```cuda
   __device__ bool warp_insert(BucketTable* table, uint32_t key, uint32_t val) {
       int lane = threadIdx.x & 31;
       uint32_t cur_key = (lane == 0) ? key : 0u;
       uint32_t cur_val = (lane == 0) ? val : 0u;
       cur_key = __shfl_sync(0xFFFFFFFFu, cur_key, 0);
       cur_val = __shfl_sync(0xFFFFFFFFu, cur_val, 0);
       
       for (int hop = 0; hop < MAX_EVICTION_HOPS; ++hop) {
           uint32_t h = xxhash3_32(cur_key);
           uint32_t b1 = h & table->bucket_mask;
           uint32_t b2 = ((h >> 16) ^ 0xDEADBEEFu) & table->bucket_mask;
           uint8_t fp = (uint8_t)(h >> 24);
           
           uint32_t evicted_key = 0u, evicted_val = 0u;
           bool success = false;
           
           if (lane == 0) {
               success = try_cuckoo_insert(&table->buckets[b1], &table->buckets[b2],
                                           cur_key, cur_val, fp,
                                           &evicted_key, &evicted_val);
           }
           
           success = __shfl_sync(0xFFFFFFFFu, success, 0);
           if (success) return true;
           
           evicted_key = __shfl_sync(0xFFFFFFFFu, evicted_key, 0);
           evicted_val = __shfl_sync(0xFFFFFFFFu, evicted_val, 0);
           cur_key = evicted_key;
           cur_val = evicted_val;
       }
       
       return false;  // Push to CPU stash
   }
   ```

2. **Implement CAS helper: `src/gpu/bucket_cuckoo_cas.cu`**
   - `try_cuckoo_insert()`: atomic CAS on occupancy_mask
   - Try b1 first, then b2
   - Return evicted key/val on collision
   - Handle concurrent inserts (same key from different threads)

3. **Unit tests: `tests/unit/test_warp_insert.cpp`**
   - ✓ Single key insert, lookup retrieves it
   - ✓ Eviction chain: insert sequence that triggers 5+ evictions
   - ✓ Load factor 0.5 boundary: insert succeeds at exactly 50%
   - ✓ Load factor 0.5 + 1: key pushed to stash (returns false from GPU insert)
   - ✓ Multiple concurrent inserts (simulated via multi-block kernel)
   - ✓ Max eviction hops: 32 evictions reached, falls back to stash

4. **Integration tests: `tests/integration/test_insert_correctness.cpp`**
   - Insert 1000 random keys up to load factor 0.5
   - Verify all keys retrievable
   - Insert one more key → stash
   - Verify stashed key in CPU memory

5. **Stash management: `src/core/cpu_stash.cpp`**
   - Maintain 128-slot pinned buffer
   - Track occupancy, push/pop operations
   - Trigger rehash when full

### Deliverables
- `src/gpu/warp_insert.cu` (main kernel)
- `src/gpu/bucket_cuckoo_cas.cu` (CAS helper)
- `src/core/cpu_stash.cpp` (CPU stash management)
- Unit tests: 100% pass rate
- Integration tests: correctness + stash behavior verified
- Estimated hours: **70–120h** (atomic operations, CAS reasoning)

---

## Phase 5: Epoch-Based Reclamation (EBR) & Rehash (Week 5–6)

**Goal:** Implement safe pointer swap for async table rehash, zero-copy reader path.

### Spec Requirements
- **Double-buffered arenas:** Two full table allocations at startup
- **Reader-count protocol:** `acquire()` increments, `release()` decrements
- **Atomic pointer swap:** Readers acquire stable pointer before swap sees new table
- **Rehash thread:** Drains old readers, then reclaims old arena
- **Async rehash:** Dedicated CPU thread triggered by stash overflow, invisible to read path

### Tasks

1. **Implement `src/core/epoch_table.cpp`**
   ```cpp
   struct EpochTable {
       std::atomic<BucketTable*> current;
       BucketTable* arenas[2];
       std::atomic<uint64_t> epoch;
       std::atomic<int32_t> readers;
   };
   
   BucketTable* WarpKVEngine::acquire() {
       epoch_table.readers.fetch_add(1, std::memory_order_acquire);
       return epoch_table.current.load(std::memory_order_acquire);
   }
   
   void WarpKVEngine::release() {
       epoch_table.readers.fetch_sub(1, std::memory_order_release);
   }
   
   void WarpKVEngine::rehash() {
       uint64_t old_epoch = epoch_table.epoch.load(std::memory_order_acquire);
       BucketTable* old_table = epoch_table.arenas[old_epoch & 1];
       BucketTable* new_table = epoch_table.arenas[(old_epoch + 1) & 1];
       
       rebuild_table(old_table, new_table);
       epoch_table.current.store(new_table, std::memory_order_release);
       epoch_table.epoch.fetch_add(1, std::memory_order_acq_rel);
       
       while (epoch_table.readers.load(std::memory_order_acquire) > 0)
           std::this_thread::yield();
       
       clear_arena(old_table);
   }
   ```

2. **Implement rehash logic: `src/core/rehash.cpp`**
   - `rebuild_table()`: Copy all entries from old table + stash to new table
   - Resize if needed (double buckets if stash was active)
   - Reset stash for next cycle

3. **Unit tests: `tests/unit/test_ebr.cpp`**
   - ✓ Double-buffered arena allocation (2 arenas)
   - ✓ Reader count increment/decrement
   - ✓ Pointer swap: old pointer remains stable for in-flight readers
   - ✓ EBR: read during rehash does not segfault
   - ✓ Epoch counter increments correctly

4. **Integration tests: `tests/integration/test_rehash.cpp`**
   - Insert keys up to stash threshold
   - Trigger rehash (stash overflow)
   - Verify all pre-rehash keys still findable
   - Verify new table visible to subsequent batches
   - Concurrent reads during rehash: no crashes

5. **Stress test: `tests/integration/test_ebr_stress.cpp`**
   - 10 parallel read threads (acquire/release loops)
   - 1 rehash thread (triggered every 10K ops)
   - 1000 total operations
   - Verify no crashes, no stale pointer reads

### Deliverables
- `src/core/epoch_table.cpp` (EBR protocol)
- `src/core/rehash.cpp` (table rebuild)
- Unit tests: 100% pass rate
- Integration tests + stress test: no crashes, correct behavior
- Memory safety validated (AddressSanitizer)
- Estimated hours: **50–80h**

---

## Phase 6: CUDA Graphs & Triple-Buffered Pipeline (Week 6–7)

**Goal:** Eliminate per-batch CPU overhead, implement 3-stage pipelined architecture.

### Spec Requirements
- **Pinned host buffers:** Triple-buffered (3 slots), H→D input, D→H output
- **Three streams:** `stream_h2d`, `stream_compute`, `stream_d2h`
- **CUDA Graph:** Capture entire pipeline topology once at startup
- **Per-batch overhead:** Single `cudaGraphLaunch` call (~1µs CPU)
- **Pipeline stages:** H→D → Compute → D→H, all overlapping

### Tasks

1. **Implement `src/gpu/pinned_buffers.cu`**
   - Allocate 3 slots of pinned memory per direction
   - Track active slot (round-robin)
   - `next_slot()`: increment counter mod 3

2. **Implement `src/gpu/cuda_graph.cu`**
   ```cpp
   void WarpKVEngine::build_graph() {
       cudaStreamBeginCapture(stream_h2d, cudaStreamCaptureModeGlobal);
       
       for (int slot = 0; slot < 3; ++slot) {
           // Stage 1: H→D
           cudaMemcpyAsync(d_keys[slot], h_keys_in[slot], BATCH_SIZE * sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream_h2d);
           cudaEventRecord(ev_h2d[slot], stream_h2d);
           
           // Stage 2: Compute (waits on H→D)
           cudaStreamWaitEvent(stream_compute, ev_h2d[slot], 0);
           warp_lookup_kernel<<<(BATCH_SIZE + 31) / 32, 32, 0, stream_compute>>>(
               epoch_table.current.load(), d_keys[slot], d_vals[slot], BATCH_SIZE);
           cudaEventRecord(ev_compute[slot], stream_compute);
           
           // Stage 3: D→H (waits on Compute)
           cudaStreamWaitEvent(stream_d2h, ev_compute[slot], 0);
           cudaMemcpyAsync(h_vals_out[slot], d_vals[slot], BATCH_SIZE * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream_d2h);
           cudaEventRecord(ev_d2h[slot], stream_d2h);
       }
       
       cudaStreamEndCapture(stream_h2d, &graph);
       cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
   }
   ```

3. **Hot-path submission: `src/gpu/submit_batch.cu`**
   ```cpp
   void WarpKVEngine::submit_batch(const uint32_t* h_keys, uint32_t batch_size) {
       BucketTable* tbl = acquire();
       int slot = next_slot();
       
       // Copy keys to pinned buffer
       memcpy(h_keys_in[slot], h_keys, batch_size * sizeof(uint32_t));
       
       // Single graph launch
       cudaGraphLaunch(graphExec, stream_h2d);
       
       // Wait for D→H completion
       cudaEventSynchronize(ev_d2h[slot]);
       
       release();
       return h_vals_out[slot];
   }
   ```

4. **Unit tests: `tests/unit/test_cuda_graph.cpp`**
   - ✓ Graph instantiation succeeds
   - ✓ Graph launch executes without error
   - ✓ Results correct after single graph replay
   - ✓ Triple buffer rotation: 1000 replays, no aliasing

5. **Integration tests: `tests/integration/test_pipeline.cpp`**
   - Insert 10K keys, lookup in batches of 4096
   - Measure pipeline efficiency: all 3 stages overlapping
   - Verify correctness: all values match expected
   - Profile: H→D, Compute, D→H times independently

6. **Performance profiling: `benchmarks/pipeline_overhead.cu`**
   - Measure CPU overhead: `cudaGraphLaunch` call (~1µs expected)
   - Compare vs. manual `cudaLaunchKernel` per batch
   - Expected speedup: 5–10× CPU overhead reduction

### Deliverables
- `src/gpu/pinned_buffers.cu` (allocation + rotation)
- `src/gpu/cuda_graph.cu` (graph capture)
- `src/gpu/submit_batch.cu` (hot path)
- Unit tests: 100% pass rate
- Integration tests: correctness + pipeline profiling
- Benchmark: graph vs. manual overhead comparison
- Estimated hours: **50–80h**

---

## Phase 7: Python Interface & GIL Bypass (Week 7–8)

**Goal:** Implement pybind11 binding, release GIL before CUDA submission.

### Spec Requirements
- **GIL release:** `py::gil_scoped_release` before `cudaGraphLaunch`
- **Python as pure routing layer:** Key marshalling → acquire → submit → release → result marshalling
- **Batch operations:** Accept Python list, return Python list
- **Thread-safe:** Multiple concurrent batches via GIL release

### Tasks

1. **Implement `src/python/warpkv_binding.cpp`**
   ```cpp
   #include <pybind11/pybind11.h>
   
   py::list warpkv_batch_lookup(WarpKVEngine& engine, py::list keys) {
       int n = keys.size();
       int slot = engine.next_slot();
       
       {
           py::gil_scoped_release release;
           
           for (int i = 0; i < n; ++i)
               engine.h_keys_in[slot][i] = keys[i].cast<uint32_t>();
           
           engine.submit_batch(slot);
       }
       
       py::list results;
       for (int i = 0; i < n; ++i)
           results.append(engine.h_vals_out[slot][i]);
       return results;
   }
   
   PYBIND11_MODULE(warpkv, m) {
       py::class_<WarpKVEngine>(m, "WarpKVEngine")
           .def(py::init<>())
           .def("lookup", &warpkv_batch_lookup)
           .def("insert", &warpkv_batch_insert);
   }
   ```

2. **Python test suite: `tests/python/test_warpkv.py`**
   - ✓ Module imports
   - ✓ Engine initializes
   - ✓ Batch lookup: 512 keys → correct results
   - ✓ GIL released: concurrent Python threads (simulated CPU work during CUDA)
   - ✓ Error handling: invalid key types, out-of-range values

3. **Integration: `tests/integration/test_python_gpu.cpp`**
   - Python calls GPU lookup
   - Verify results correct
   - No GIL deadlock

4. **CMakeLists.txt updates**
   - Find pybind11, add Python module target
   - Link against CUDA, curand (optional)
   - Install target: Python wheel or direct `.so`

### Deliverables
- `src/python/warpkv_binding.cpp` (pybind11 binding)
- `tests/python/test_warpkv.py` (Python tests)
- Python module: importable via `import warpkv`
- CMakeLists.txt: Python module build target
- Unit tests: 100% pass rate
- Estimated hours: **30–50h**

---

## Phase 8: Comprehensive Testing & Validation (Week 8–9)

**Goal:** Unit + integration + stress + property-based tests; validate all spec invariants.

### Test Categories

1. **Unit Tests (existing phases 1–7)**
   - Total: ~50 test cases
   - Coverage: hash, bucket layout, kernels, EBR, graph, Python

2. **Integration Tests**
   - End-to-end: insert 10K keys, lookup all, verify correctness
   - Pipeline: concurrent H→D, Compute, D→H stages
   - Rehash: trigger stash overflow, verify all keys still findable
   - Python: batch lookups from Python, GIL release verified

3. **Stress Tests**
   - 100K key insertion/lookup cycles
   - Concurrent reader threads + rehash thread
   - AddressSanitizer: memory safety
   - ThreadSanitizer: race condition detection

4. **Property-Based Tests** (QuickCheck-style)
   - `prop_insert_lookup_commute`: insert(X) then lookup(X) == X
   - `prop_load_factor_bounded`: load_factor ≤ 0.5 always
   - `prop_no_duplicate_buckets`: `b1 != b2` always (across 1M random keys)
   - `prop_fingerprint_fast_path`: fingerprint rejection avoids false positives

5. **Spec Compliance Checklist**
   - ✓ Hash function: XXHash3, no MurmurHash3
   - ✓ Bucket layout: exactly 128 bytes, SoA within bucket
   - ✓ Lookup bound: ≤2 cache-line reads worst case
   - ✓ Insert bound: ≤32 eviction hops before stash
   - ✓ Load factor: capped at 0.50
   - ✓ EBR: no use-after-free under concurrent rehash
   - ✓ CUDA Graph: single `cudaGraphLaunch` per batch
   - ✓ GIL: released before CUDA work
   - ✓ PCIe bandwidth: ≤6 GB/s not exceeded

### Tasks

1. **Write comprehensive test suite**
   - `tests/unit/test_all.cpp` (runs all unit tests)
   - `tests/integration/test_all.cpp` (runs all integration tests)
   - `benchmarks/stress_test.cu` (100K cycles)

2. **Spec compliance checklist: `docs/compliance_checklist.md`**
   - Every requirement from spec Section I–VII
   - Test case that validates each requirement
   - Pass/fail status

3. **Code coverage report**
   - Target: ≥90% line coverage on GPU kernels
   - Use gcov + lcov (or LLVM coverage)

### Deliverables
- Test suite: 100+ test cases, 100% pass rate
- Spec compliance checklist: all items validated
- Code coverage report: ≥90% coverage
- AddressSanitizer: zero leaks/errors
- ThreadSanitizer: zero race conditions
- Estimated hours: **60–100h**

---

## Phase 9: Benchmark Suite & Competitive Analysis (Week 9–10)

**Goal:** Measure performance vs. Redis, FASTER, cuCollections; publish benchmark results.

### Benchmarks

1. **Throughput Benchmark: `benchmarks/throughput_sweep.cu`**
   - Batch sizes: [128, 256, 512, 1024, 2048, 4096, 8192]
   - Measure: batches/sec, keys/sec, latency (µs/batch)
   - Hardware: MX130 (target), H100 (aspirational)
   - Expected WarpKV result: ≥512 entries break-even, 4–6× Redis at 4096 entries

2. **Latency Benchmark: `benchmarks/latency_profile.cu`**
   - Single batch latency vs. batch size
   - Breakdown: H→D (µs), Compute (µs), D→H (µs)
   - P50, P99, P99.9 latency

3. **Competitive Baselines**
   - **Redis 7.x local:**
     - Setup: single-threaded Redis instance
     - Workload: pipelined GET commands, same batch sizes
     - Result: publish comparison table
   - **FASTER (Microsoft):**
     - CPU-only KV store, epoch-based concurrency
     - Same workload: pipelined 512–8192 item batches
     - Result: CPU vs. GPU comparison
   - **cuCollections `cuco::static_map`:**
     - GPU peer baseline
     - Same batch sizes, same data
     - Result: bucket cuckoo vs. linear probing comparison

4. **Memory Footprint: `benchmarks/memory_footprint.cu`**
   - Table size vs. num keys
   - Expected: load_factor=0.5 → 2× overhead vs. dense packing
   - VRAM budget on MX130: 2GB total, how much for table?

5. **L2 Cache Analysis: `benchmarks/cache_behavior.cu`**
   - L2 hit rate for batch sizes [512, 1024, 4096, 16384]
   - Expected: ≥95% L2 hit rate at ≤4096 keys (fits in 512 KB)
   - Profile with Nsight

### Tasks

1. **Implement benchmark harness: `benchmarks/harness.cpp`**
   - Command-line args: `--batch-size`, `--num-batches`, `--dataset`
   - Warm-up: 10 batches before timing
   - Measurement: 1000 batches, record each
   - Output: CSV (batch_size, latency_us, throughput_keys_sec)

2. **Generate datasets: `benchmarks/datasets.cpp`**
   - Random 32-bit keys, uniform distribution
   - Real-world datasets: if available (e.g., YCSB)
   - Save to file for reproducibility

3. **Redis baseline: `benchmarks/redis_baseline.sh`**
   - Start local Redis instance
   - Run pipelined benchmark
   - Collect results

4. **FASTER baseline: `benchmarks/faster_baseline.cpp`**
   - Link against FASTER library
   - Run same workload
   - Collect results

5. **Benchmark report: `docs/benchmarks/RESULTS.md`**
   - Table: WarpKV vs. Redis vs. FASTER vs. cuCollections
   - Charts: throughput vs. batch size
   - Analysis: break-even point, scalability, memory efficiency
   - Caveats: MX130 constraints, single-node scope, write-cold workload

### Deliverables
- Benchmark harness: compilable, runs on MX130
- Benchmark results: CSV + plots
- Competitive analysis report: `docs/benchmarks/RESULTS.md`
- Charts: throughput, latency, L2 hit rate
- Estimated hours: **80–120h** (includes FASTER/Redis setup)

---

## Phase 10: Documentation & Production Readiness (Week 10–11)

**Goal:** Complete documentation, CI/CD pipeline, release-ready code.

### Tasks

1. **API Documentation**
   - Doxygen: all public functions in `src/gpu/`, `src/core/`, `src/python/`
   - Usage examples: CPU insert, GPU lookup, Python interface
   - Header file: `include/warpkv/warpkv.h` (public API)

2. **User Guide: `docs/USER_GUIDE.md`**
   - Installation from source
   - Build instructions (CMake)
   - Example C++ code: initialize, insert, lookup
   - Example Python code: batch lookup
   - Troubleshooting: CUDA not found, VRAM limits, etc.

3. **Architecture Deep-Dive: `docs/ARCHITECTURE.md`**
   - Mirrors spec (Sections I–VII)
   - Implementation details: actual code paths, optimizations
   - Design decisions: why bucket cuckoo, why XXHash3, why EBR

4. **Performance Tuning: `docs/PERFORMANCE_TUNING.md`**
   - Batch size selection: break-even analysis
   - Load factor impact: insertion speed vs. space efficiency
   - PCIe bandwidth limits: when to use GPU vs. CPU

5. **CI/CD Pipeline**
   - GitHub Actions: `cmake && make && make test` on each commit
   - Automated benchmark runs (if hardware available)
   - Code coverage reports (codecov.io)

6. **Release Package**
   - `CMakeLists.txt`: install targets
   - Python wheel: `pip install warpkv`
   - Versioning: `v2.0.0` tag, CHANGELOG.md

7. **Code Cleanup**
   - Remove old naive cuckoo code (from misaligned implementation)
   - Consistent style: clang-format with `.clang-format` config
   - Comment pass: clarify non-obvious code

### Deliverables
- Doxygen documentation: HTML API reference
- User guide + architecture guide + tuning guide
- CI/CD workflow file (GitHub Actions YAML)
- Release package: source tarball + Python wheel
- Changelog: migration guide from v1 → v2
- Estimated hours: **40–60h**

---

## Critical Path & Risk Mitigation

### Critical Path (Longest Dependency Chain)
1. **Phase 0** (Foundation) → **Phase 1** (XXHash3)
2. → **Phase 2** (Bucket layout)
3. → **Phase 3** (Lookup kernel) [HIGH RISK]
4. → **Phase 4** (Insert kernel)
5. → **Phase 5** (EBR) [MEDIUM RISK]
6. → **Phase 6** (CUDA Graphs)
7. → **Phase 7** (Python)
8. → **Phase 9** (Benchmarks) [MEDIUM RISK: requires FASTER setup]

**Parallel tracks:**
- Phase 8 (Testing) can start after Phase 4
- Phase 10 (Documentation) can start after Phase 7

### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Warp divergence in Phase 3** | Extensive unit tests on `__ballot_sync` logic; profile with Nsight to verify zero divergence |
| **CAS race conditions in Phase 4** | Stress test with concurrent inserts; use ThreadSanitizer; design review of atomic semantics |
| **EBR memory safety (Phase 5)** | AddressSanitizer on all tests; explicit reader-count verification; hazard pointer alternative ready |
| **CUDA Graph replay state leakage (Phase 6)** | Test 1000 consecutive replays; verify no corruption; profiler check for unintended state persistence |
| **GIL deadlock (Phase 7)** | Test concurrent Python threads during CUDA work; explicit release timing check |
| **Benchmark variability** | Warm-up runs, multiple trials (N≥5), publish error bars; control for system noise (power scaling disabled) |

---

## Testing Strategy Summary

| Test Type | Phase | Count | Goal |
|-----------|-------|-------|------|
| **Unit** | 1–7 | 50+ | Spec compliance per component |
| **Integration** | 3–7 | 20+ | End-to-end correctness |
| **Stress** | 8 | 3 | Concurrent + 100K cycles |
| **Property** | 8 | 5 | Invariant validation |
| **Benchmark** | 9 | 3 | Competitive analysis |
| **Total** | — | **80+** | 100% coverage + SOTA validation |

---

## Success Criteria

- ✓ All spec requirements (Sections I–VII) met exactly as written
- ✓ 100% test pass rate: 80+ test cases
- ✓ Zero sanitizer errors: AddressSanitizer, ThreadSanitizer, MemorySanitizer
- ✓ Code coverage: ≥90% on GPU kernels, ≥85% overall
- ✓ Benchmark results: published with error bars, comparable to Redis/FASTER
- ✓ Documentation: complete, Doxygen + user guide + architecture guide
- ✓ Production readiness: CI/CD passing, release package available, no known issues

---

## Estimated Effort & Timeline

| Phase | Hours | FTE-weeks | Critical Path |
|-------|-------|-----------|---|
| Phase 0 (Foundation) | 50 | 1.25 | Yes |
| Phase 1 (XXHash3) | 25 | 0.6 | Yes |
| Phase 2 (Bucket layout) | 65 | 1.6 | Yes |
| Phase 3 (Lookup) | 80 | 2.0 | **Yes** ⚠️ |
| Phase 4 (Insert) | 95 | 2.4 | Yes |
| Phase 5 (EBR) | 65 | 1.6 | Yes |
| Phase 6 (CUDA Graphs) | 65 | 1.6 | Yes |
| Phase 7 (Python) | 40 | 1.0 | Yes |
| Phase 8 (Testing) | 80 | 2.0 | Parallel |
| Phase 9 (Benchmarks) | 100 | 2.5 | End |
| Phase 10 (Polish) | 50 | 1.25 | End |
| **Total** | **715** | **17.8 FTE-weeks** | — |

**For 1 FTE:** ~4.5 calendar months (18 weeks)  
**For 2 FTE:** ~2.25 calendar months (9 weeks, with parallelization)  
**Buffer:** +20% for unknowns (Phase 3 & 5 historically most complex)

---

## Next Steps

1. **Week 1 Start:**
   - Set up build system (Phase 0)
   - Implement XXHash3 + unit tests (Phase 1)
   - Establish hardware baseline

2. **Checkpoint after Phase 2:**
   - Bucket layout complete, arena allocator working
   - Decision: proceed to Phase 3 or adjust plan

3. **Checkpoint after Phase 5:**
   - EBR + rehash complete, pointer safety validated
   - Decision: benchmarking timeline, release date

4. **Final release:**
   - All 10 phases complete
   - Spec compliance checklist: 100% ✓
   - Benchmark report published
   - Ready for production deployment

---

**Document version: 1.0**  
**Last updated: 2026-06-27**  
**Prepared by:** Claude Code — WarpKV v2 Implementation Planner
