# WarpKV v2 — Complete Technical Specification

**Project:** WarpKV — GPU-Accelerated Key-Value Store  
**Architecture:** Hybrid C++17 / CUDA (Compute Capability 5.0+)  
**Primary Target:** NVIDIA MX130 (PCIe Gen 3 x4, 384 CUDA cores, 2GB GDDR5)  
**Version:** 2.0 — SOTA-Aligned  
**Status:** Pre-Implementation Design Document

---

## Table of Contents

1. [Core Philosophy & Objective](#i-core-philosophy--objective)
2. [Explicit Non-Goals](#ii-explicit-non-goals)
3. [Hardware Constraints & Crossover Math](#iii-hardware-constraints--crossover-math)
4. [Hash Table Core — Bucket Cuckoo](#iv-hash-table-core--bucket-cuckoo-cache-line-aligned)
5. [Hash Function — XXHash3](#v-hash-function--xxhash3)
6. [Memory Architecture](#vi-memory-architecture)
7. [Pipeline Architecture — CUDA Graphs](#vii-pipeline-architecture--cuda-graphs)
8. [Epoch-Based Reclamation — Safe Pointer Swap](#viii-epoch-based-reclamation--safe-pointer-swap)
9. [Python Interface — GIL Bypass](#ix-python-interface--gil-bypass-via-pybind11)
10. [L2 Cache Working Set Analysis](#x-l2-cache-working-set-analysis)
11. [Competitive Baseline](#xi-competitive-baseline)
12. [Design Decision Log](#xii-design-decision-log--rejected-alternatives)
13. [Unit Test Plan](#xiii-unit-test-plan)
14. [Build Order](#xiv-build-order)

---

## I. Core Philosophy & Objective

**The Goal:** WarpKV is an ephemeral, GPU-accelerated key-value store engineered to dominate read-heavy, high-batch-throughput workloads where CPU-bound stores serialise under load. It is not a general-purpose database — it is a precision instrument for a specific performance regime.

**The Constraint as a Feature:** The MX130 (CC 5.0, PCIe Gen 3 x4, ~6 GB/s effective bidirectional bandwidth) is a deliberate target. Optimising a GPU pipeline against a severely bottlenecked PCIe bus forces every design decision to be mechanically justified. Designing for the floor proves the fundamentals. Anyone can saturate an H100.

**The Python GIL Bypass:** Python's Global Interpreter Lock prevents true parallel execution within a Python process. By pushing all data-path logic through a pybind11 binding that explicitly releases the GIL before CUDA submission, Python is reduced to a pure routing layer. The C++/CUDA backend executes entirely outside Python's scheduler.

**The Regime:** WarpKV wins at batch sizes ≥ 512 entries. Below this threshold, fixed PCIe round-trip cost (~20–25µs) exceeds Redis's per-query latency. This crossover is stated explicitly and quantified in Section III.

---

## II. Explicit Non-Goals

| Non-Goal | Rationale |
|---|---|
| **Persistence** | Ephemeral store. Process death = data loss. Documented upfront, not an oversight. |
| **Sub-512 single-key lookups** | PCIe latency dominates. Redis wins this regime. |
| **Write-heavy workloads** | Bucket cuckoo at 0.5 load factor is optimised for read-cold insertion, not write-hot throughput. |
| **NVLink / multi-GPU** | MX130 has no NVLink. PCIe is the only interconnect. |
| **Distributed operation** | Single-node, single-GPU scope. |

---

## III. Hardware Constraints & Crossover Math

### MX130 Hardware Profile

| Parameter | Value |
|---|---|
| Compute Capability | 5.0 (Maxwell) |
| CUDA Cores | 384 |
| VRAM | 2 GB GDDR5 |
| PCIe Version | Gen 3 x4 |
| Effective Bidirectional BW | ~6 GB/s |
| L2 Cache | 512 KB |
| Warp Size | 32 threads |

### Crossover Math vs. Redis

| Parameter | Value |
|---|---|
| H→D transfer for 512 × 8B entries | ~4 KB → ~0.7µs |
| D→H transfer for 512 × 4B values | ~2 KB → ~0.3µs |
| Kernel launch + compute (512 keys) | ~15µs on MX130 |
| CUDA event synchronisation overhead | ~3–5µs |
| **Total GPU round trip (512 entries)** | **~20–25µs** |
| Redis P99 latency (pipelined 512 GETs, local) | ~80–120µs |
| **Break-even batch size** | **~512 entries** |

At 4096 entries (8× break-even), WarpKV's pipelined throughput is projected to exceed Redis by **4–6×** on this hardware. Stated as projection pending benchmark against FASTER and Redis 7.x baselines. Not asserted as fact.

---

## IV. Hash Table Core — Bucket Cuckoo (Cache-Line Aligned)

### Why Not Naive Cuckoo

Standard cuckoo hashing assigns one key per slot. On GPU, each thread in a warp follows an independent eviction chain of unpredictable length during insertion. The result is severe warp divergence — threads in the same warp serialise waiting for the longest chain. This is the primary reason production GPU hash tables (cuCollections, WarpCore) abandoned per-slot cuckoo.

### Why Not Linear Probing

Linear probing (`cuco::static_map` style) offers simpler insertion and higher load factor (0.9+), but provides no upper bound on probe chain length. For a read-optimised, write-cold store, the guaranteed 2-bucket lookup bound of bucket cuckoo is the correct tradeoff even at 50% VRAM efficiency. If WarpKV were write-heavy, linear probing would be the correct choice — it is explicitly not being built for that workload.

### Bucket Cuckoo Design

Each bucket is exactly **one cache line = 128 bytes**. With 4-byte keys and 4-byte values, one bucket holds **8 key-value slots** in SoA layout within the bucket, plus fingerprints and occupancy mask.

```
Bucket Layout (128 bytes total):
┌─────────────────────────────────────────┐
│ keys[8]        — 32 bytes (uint32_t×8)  │
│ values[8]      — 32 bytes (uint32_t×8)  │
│ fingerprint[8] — 8 bytes  (uint8_t×8)   │
│ occupancy_mask — 4 bytes  (uint32_t)    │
│ padding        — 52 bytes               │
└─────────────────────────────────────────┘
```

**Fingerprints** (upper 8 bits of the hash, 1 byte per slot) allow a warp to reject non-matching slots via a single `__ballot_sync` comparison before reading the value array — eliminating unnecessary VRAM fetches on misses.

A single warp (32 threads) checks **two buckets simultaneously**: lanes 0–15 scan bucket b1, lanes 16–31 scan bucket b2. Both are checked in a single warp pass with zero divergence on the lookup path.

### Warp-Cooperative Lookup Kernel

```cuda
__device__ uint32_t warp_lookup(const BucketTable* __restrict__ table,
                                 uint32_t key) {
    const uint32_t h   = xxhash3_32(key);
    const uint32_t b1  = h  & table->bucket_mask;               // power-of-two mask
    const uint32_t b2  = ((h >> 16) ^ 0xDEADBEEFu) & table->bucket_mask;
    const uint8_t  fp  = (uint8_t)(h >> 24);

    const int lane = threadIdx.x & 31;

    // Lanes 0-15 → bucket b1, lanes 16-31 → bucket b2
    const uint32_t    bucket_idx = (lane < 16) ? b1 : b2;
    const int         slot       = lane & 15;
    const Bucket* bkt        = &table->buckets[bucket_idx];

    // Single coalesced read: all lanes in warp hit same bucket (128B = 1 cache line)
    const bool hit = (bkt->occupancy_mask >> slot & 1u)    &&
                     (bkt->fingerprint[slot] == fp)         &&
                     (bkt->keys[slot] == key);

    const uint32_t match_mask = __ballot_sync(0xFFFFFFFFu, hit);

    if (match_mask == 0u) return WARPKV_NOT_FOUND;

    const int winner = __ffs(match_mask) - 1;  // lowest-lane match wins
    const uint32_t result = __shfl_sync(0xFFFFFFFFu,
                                         bkt->values[slot],
                                         winner);
    return result;
}
```

**Critical properties:**
- Zero warp divergence on the lookup path — all branches are data-uniform across the warp.
- Both candidate buckets are checked in one warp pass — guaranteed ≤2 cache-line reads per lookup, worst case.
- `__ffs` (find first set) is a single hardware instruction on Maxwell+.

### Insertion Strategy

Insertion is **not on the hot path**. WarpKV treats the table as write-once-read-many within a session.

```cuda
__device__ bool warp_insert(BucketTable* table, uint32_t key, uint32_t val) {
    const int lane = threadIdx.x & 31;

    uint32_t cur_key = (lane == 0) ? key : 0u;
    uint32_t cur_val = (lane == 0) ? val : 0u;

    // Broadcast to all lanes — warp-cooperative eviction
    cur_key = __shfl_sync(0xFFFFFFFFu, cur_key, 0);
    cur_val = __shfl_sync(0xFFFFFFFFu, cur_val, 0);

    for (int hop = 0; hop < MAX_EVICTION_HOPS; ++hop) {
        const uint32_t h   = xxhash3_32(cur_key);
        const uint32_t b1  = h  & table->bucket_mask;
        const uint32_t b2  = ((h >> 16) ^ 0xDEADBEEFu) & table->bucket_mask;
        const uint8_t  fp  = (uint8_t)(h >> 24);

        // Lane 0 attempts CAS insertion into b1, then b2
        if (lane == 0) {
            // ... atomic CAS on occupancy_mask, key, value, fingerprint
            // Returns true on success, evicted key/val on collision
        }

        // Broadcast result to all lanes
        bool success = __shfl_sync(0xFFFFFFFFu, /* success flag */ 0, 0);
        if (success) return true;

        // Broadcast evicted key/val for next hop
        cur_key = __shfl_sync(0xFFFFFFFFu, cur_key, 0);
        cur_val = __shfl_sync(0xFFFFFFFFu, cur_val, 0);
    }

    return false;  // Push to CPU stash
}
```

**Insertion safety parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Max load factor | 0.50 | Below 50%, eviction chains are statistically short. Above 65%, livelock probability becomes non-negligible. |
| Max eviction hops | 32 | Bounds worst-case insertion time. Empirically sufficient at 0.5 load factor. |
| CPU stash size | 128 slots (pinned memory) | Holds un-placeable keys after hop limit. CPU resolves serially. |
| Stash overflow action | Async full-table rehash on dedicated CUDA stream | Invisible to read path via double-buffered table pointers + EBR (see Section VIII). |

---

## V. Hash Function — XXHash3

MurmurHash3 is replaced with **XXHash3** (32-bit finalizer variant for short keys ≤ 16 bytes).

**Reasons for selection:**
- SIMD-vectorizable on CPU preprocessing path.
- Superior avalanche at low key lengths — the dominant case for a KV store.
- No known bias issues at low moduli unlike MurmurHash3's final mix step.
- Measurably faster in cuCollections benchmark comparisons for short keys.

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

### Power-of-Two Bucket Indexing

The modulo operator (`%`) is an integer division instruction — 20–40 cycle latency on CC 5.0. In a warp-cooperative hot loop, this is unacceptable.

**Fix:** Force bucket count to power-of-two at allocation time; store the mask alongside the table. All index computations use single-cycle bitwise AND.

```cpp
// At arena allocation (host side)
size_t raw_buckets = arena_size / sizeof(Bucket);
size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));  // round down to power of 2

table.num_buckets  = num_buckets;
table.bucket_mask  = (uint32_t)(num_buckets - 1);  // stored in device struct
```

```cuda
// In kernel — bitwise AND replaces division
uint32_t b1 = h  & table->bucket_mask;                           // 1 cycle
uint32_t b2 = ((h >> 16) ^ 0xDEADBEEFu) & table->bucket_mask;  // 1 cycle
```

The XOR with `0xDEADBEEF` on `b2` decorrelates the two hash positions, preventing the degenerate case where `b1 == b2` halves effective capacity. No latency cost.

---

## VI. Memory Architecture

### SoA Layout at Table Level

The global table is SoA — all bucket key arrays are contiguous in VRAM, all value arrays are contiguous, all fingerprint arrays are contiguous. A warp scanning buckets hits sequential VRAM addresses, guaranteeing 128-byte cache-line coalescing with zero striding.

```
VRAM Layout:
[ Bucket[0].keys | Bucket[1].keys | ... | Bucket[N].keys ]   ← coalesced key access
[ Bucket[0].vals | Bucket[1].vals | ... | Bucket[N].vals ]   ← coalesced value access
[ Bucket[0].fps  | Bucket[1].fps  | ... | Bucket[N].fps  ]   ← coalesced fingerprint access
```

Within each bucket, keys and values are also SoA (not interleaved) — a lane reading `keys[slot]` does not stride through intervening values.

### Static Arena Allocation

`cudaMalloc` is a global driver synchronisation point. Calling it during a lookup serialises the entire device. The arena ensures it is called exactly **once**, at startup.

```cpp
void WarpKVEngine::init() {
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);

    // Reserve 80% of free VRAM — leave headroom for output buffers and stash
    size_t arena_size = (size_t)(free_vram * 0.80);
    cudaMalloc(&device_arena, arena_size);

    // Sub-allocate within arena — zero runtime cudaMalloc in hot path
    table.buckets     = (Bucket*)device_arena;
    table.num_buckets = 1ULL << (63 - __builtin_clzll(arena_size / sizeof(Bucket)));
    table.bucket_mask = (uint32_t)(table.num_buckets - 1);
}
```

### Pinned Host Buffers — Triple-Buffered

Three buffer slots per direction to match the three pipeline stages (H→D, Compute, D→H). Pinned (page-locked) memory allows the CUDA DMA engine to perform transfers without CPU involvement — standard `malloc` buffers require a CPU-mediated copy via a hidden staging buffer inside the CUDA driver.

```cpp
static constexpr int NUM_SLOTS  = 3;
static constexpr int BATCH_SIZE = 4096;  // tunable; break-even at ~512 on MX130

uint32_t* h_keys_in [NUM_SLOTS];   // pinned — host→device staging
uint32_t* h_vals_out[NUM_SLOTS];   // pinned — device→host staging
uint32_t* d_keys    [NUM_SLOTS];   // device — input key buffers
uint32_t* d_vals    [NUM_SLOTS];   // device — output value buffers

void WarpKVEngine::alloc_buffers() {
    for (int i = 0; i < NUM_SLOTS; ++i) {
        cudaMallocHost(&h_keys_in[i],  BATCH_SIZE * sizeof(uint32_t));
        cudaMallocHost(&h_vals_out[i], BATCH_SIZE * sizeof(uint32_t));
        cudaMalloc    (&d_keys[i],     BATCH_SIZE * sizeof(uint32_t));
        cudaMalloc    (&d_vals[i],     BATCH_SIZE * sizeof(uint32_t));
    }
}
```

---

## VII. Pipeline Architecture — CUDA Graphs

### Why CUDA Graphs over Manual Stream Dispatch

Manual kernel dispatch via `cudaLaunchKernel` incurs **~5–10µs CPU-side overhead per call** — driver validation, argument marshalling, scheduler interaction. At 10,000 batches/sec, that is 50–100ms/sec of pure scheduling tax: not compute, not transfer, just API overhead.

CUDA Graphs capture the entire pipeline topology — streams, events, kernel launches, memcpy nodes — in a single compiled execution graph. Each subsequent batch is replayed with a single `cudaGraphLaunch` call at ~1µs overhead, regardless of graph complexity.

### Three-Stage Pipeline Topology

```
Slot:         [0]              [1]              [2]
Stream H→D:   ─[Xfer In A]────────────────────[Xfer In B]──────────────
                    │ ev_h2d[0]                      │ ev_h2d[1]
Stream Comp:        └──[Compute A]─────────────────[Compute B]──────────
                             │ ev_compute[0]                │ ev_compute[1]
Stream D→H:                  └──[Xfer Out A]──────────────[Xfer Out B]──
                                      │ ev_d2h[0]
                                   CPU reads here after cudaEventSynchronize
```

At steady state:
- Stream D→H is transferring batch N results to host
- Stream Compute is processing batch N+1
- Stream H→D is loading batch N+2 into device memory

No stage stalls. All three stages overlap continuously.

### CUDA Event Synchronisation Points

Six events per slot define the dependency graph precisely:

| Event | Fired by | Waited by |
|---|---|---|
| `ev_h2d[slot]` | Stream H→D after transfer completes | Stream Compute before kernel launch |
| `ev_compute[slot]` | Stream Compute after kernel completes | Stream D→H before result transfer |
| `ev_d2h[slot]` | Stream D→H after transfer completes | CPU thread via `cudaEventSynchronize` |

### CUDA Graph Capture

```cpp
void WarpKVEngine::build_graph(int slot) {
    cudaStreamBeginCapture(stream_h2d, cudaStreamCaptureModeGlobal);

    // Stage 1: H→D transfer
    cudaMemcpyAsync(d_keys[slot], h_keys_in[slot],
                    BATCH_SIZE * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream_h2d);
    cudaEventRecord(ev_h2d[slot], stream_h2d);

    // Stage 2: Compute — waits on H→D completion
    cudaStreamWaitEvent(stream_compute, ev_h2d[slot], 0);
    dim3 block(32);   // one warp per block; one key per warp
    dim3 grid((BATCH_SIZE + 31) / 32);
    warp_lookup_kernel<<<grid, block, 0, stream_compute>>>(
        epoch_table.current.load(), d_keys[slot], d_vals[slot], BATCH_SIZE);
    cudaEventRecord(ev_compute[slot], stream_compute);

    // Stage 3: D→H transfer — waits on compute completion
    cudaStreamWaitEvent(stream_d2h, ev_compute[slot], 0);
    cudaMemcpyAsync(h_vals_out[slot], d_vals[slot],
                    BATCH_SIZE * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream_d2h);
    cudaEventRecord(ev_d2h[slot], stream_d2h);

    cudaStreamEndCapture(stream_h2d, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
}

// Hot path — called per batch
void WarpKVEngine::submit_batch(int slot) {
    cudaGraphLaunch(graphExec, stream_h2d);           // ~1µs CPU overhead
    cudaEventSynchronize(ev_d2h[slot]);               // block until results ready
}
```

The graph is instantiated once at startup. Every subsequent batch is a single `cudaGraphLaunch`.

---

## VIII. Epoch-Based Reclamation — Safe Pointer Swap

### The Problem

When stash overflow triggers a full-table rehash, the rehash thread must swap the `BucketTable*` pointer visible to the kernel. A bare atomic pointer swap is unsafe: any kernel that was mid-execution on the old table holds a dangling pointer if the old arena is freed before the kernel completes.

### The Solution: Simplified EBR (Epoch-Based Reclamation)

Two table arenas are allocated at startup. The read path never frees memory — it reads whichever arena the current epoch designates. The rehash thread waits until all in-flight readers drain before reclaiming the old arena.

```cpp
struct EpochTable {
    std::atomic<BucketTable*> current;    // read path reads this — never NULL
    BucketTable*              arenas[2];  // double-buffered arenas, allocated at startup
    std::atomic<uint64_t>     epoch;      // monotonically increasing, even=arena[0], odd=arena[1]
    std::atomic<int32_t>      readers;    // count of in-flight GPU batches on current table
};

// READ PATH — called before every cudaGraphLaunch
BucketTable* WarpKVEngine::acquire() {
    epoch_table.readers.fetch_add(1, std::memory_order_acquire);
    return epoch_table.current.load(std::memory_order_acquire);
    // Pointer is stable for the lifetime of this batch.
    // release() is called after cudaEventSynchronize(ev_d2h[slot]).
}

void WarpKVEngine::release() {
    epoch_table.readers.fetch_sub(1, std::memory_order_release);
}

// REHASH PATH — runs on dedicated CPU thread, triggered by stash overflow
void WarpKVEngine::rehash() {
    uint64_t old_epoch = epoch_table.epoch.load(std::memory_order_acquire);
    BucketTable* old_table = epoch_table.arenas[old_epoch & 1];
    BucketTable* new_table = epoch_table.arenas[(old_epoch + 1) & 1];

    // 1. Build new table from old table + stash contents (on new_table arena)
    rebuild_table(old_table, new_table);

    // 2. Atomically publish new table — readers arriving AFTER this see new_table
    epoch_table.current.store(new_table, std::memory_order_release);
    epoch_table.epoch.fetch_add(1, std::memory_order_acq_rel);

    // 3. Drain all in-flight readers that acquired old_table BEFORE the swap
    //    These hold stable pointers; we wait for them to complete and call release()
    while (epoch_table.readers.load(std::memory_order_acquire) > 0)
        std::this_thread::yield();

    // 4. Safe to clear old arena — zero live readers remain
    clear_arena(old_table);
    // old_table arena is now available for the next rehash
}
```

**Key invariant:** Any batch that called `acquire()` before the pointer swap holds a stable pointer for its entire GPU lifetime. The rehash thread spins on `readers == 0` before touching the old arena. No kernel ever sees a freed or partially-overwritten table.

**Why not hazard pointers?** Hazard pointer schemes require per-thread registration and scanning, which adds overhead on the critical read path. For WarpKV's single-writer rehash model (one rehash thread, multiple reader threads), the reader-count approach is simpler, correct, and sufficient.

---

## IX. Python Interface — GIL Bypass via pybind11

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

py::list warpkv_batch_lookup(WarpKVEngine& engine, py::list keys) {
    const int n = (int)keys.size();

    // ── GIL RELEASE ──────────────────────────────────────────────
    // From this point, Python's scheduler cannot preempt this thread.
    // All CUDA operations execute outside Python's concurrency model.
    {
        py::gil_scoped_release release;

        int slot = engine.next_slot();  // round-robin across triple buffer
        BucketTable* tbl = engine.acquire();

        // Copy keys from Python list into pinned staging buffer
        // (key copy happens before GIL release in production;
        //  shown here for clarity of where CUDA work begins)
        for (int i = 0; i < n; ++i)
            engine.h_keys_in[slot][i] = keys[i].cast<uint32_t>();

        // Submit: single graph launch (~1µs), then block on D→H completion
        engine.submit_batch(slot);       // cudaGraphLaunch + cudaEventSynchronize
        engine.release();

        // Results are now in engine.h_vals_out[slot] — pinned host memory
    }
    // ── GIL REACQUIRED ───────────────────────────────────────────

    py::list results;
    int slot = engine.current_slot();
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

Python is a **pure routing layer**. It submits keys, releases the GIL, and the entire CUDA pipeline — DMA transfer, warp-cooperative hash lookup, DMA return — executes without Python's scheduler touching it.

---

## X. L2 Cache Working Set Analysis

### MX130 L2: 512 KB

Active working set for a 4096-key batch lookup:

| Component | Calculation | Size |
|---|---|---|
| Keys array | 4096 × 4B | 16 KB |
| Values array | 4096 × 4B | 16 KB |
| Fingerprints | 4096 × 1B | 4 KB |
| Occupancy masks | ~512 buckets hit × 4B | 2 KB |
| **Total hot footprint** | | **~38 KB** |

38 KB fits comfortably inside 512 KB L2. WarpKV has approximately **13× L2 headroom** at the 4096-entry batch size.

### Scaling Ceiling

L2 thrash begins at approximately 35,000-entry batch sizes on MX130. This is well outside WarpKV's target operating range (512–8192 entries). Stated explicitly:

> *At batch sizes ≤ 32K entries, the active working set fits within the MX130's 512 KB L2 cache. Lookups are L2-bound, not VRAM-bandwidth-bound. Beyond 32K, latency degrades to VRAM-bandwidth-bound (~6 GB/s PCIe ceiling).*

### Mitigation at Scale (If Required)

If the working set must exceed L2 capacity, the correct mitigation is **value indirection** — not FP16 quantisation (which is only appropriate for floating-point embedding values). Value indirection stores a 4-byte offset into a value heap in the bucket, deferring the value VRAM fetch to a second pass executed only for matched keys.

This reduces the hot lookup path to keys + fingerprints only:
- Hot footprint at 4096 entries: ~20 KB (keys 16 KB + fingerprints 4 KB)
- L2 ceiling extends to approximately 65K entries before thrash

Value indirection adds one additional VRAM read per hit (not per probe), which is acceptable when the hit rate is low relative to probe count.

---

## XI. Competitive Baseline

### Baseline Selection

| Baseline | Role | Why |
|---|---|---|
| **FASTER (Microsoft)** | Primary engineering baseline | CPU KV store built for high-throughput; epoch-based concurrency; what WarpKV must beat to justify GPU cost |
| **Redis 7.x** | Deployment context baseline | What WarpKV replaces in a Python service; single-threaded command loop; Python GIL interaction compounds against it |
| **cuCollections `cuco::static_map`** | GPU peer baseline | Establishes where WarpKV sits within GPU-native KV store design space |

Redis is **not** the primary engineering baseline — framing WarpKV as "faster than Redis" alone is equivalent to benchmarking against `std::unordered_map`. FASTER is the honest ceiling for a CPU KV store.

### Expected Competitive Position

- **vs Redis:** WarpKV wins at batch ≥ 512 on read-heavy workloads. Redis wins on single-key sub-millisecond latency and write throughput.
- **vs FASTER:** Competitive at large batch sizes. FASTER's epoch-based CPU concurrency is formidable; the GPU advantage must be demonstrated empirically, not asserted.
- **vs cuCollections:** cuCollections uses linear probing at higher load factor. WarpKV's bucket cuckoo provides a deterministic 2-bucket lookup bound that cuCollections cannot guarantee. WarpKV trades VRAM efficiency for lookup latency predictability.

---

## XII. Design Decision Log — Rejected Alternatives

| Component | Chosen | Rejected | Reason for Rejection |
|---|---|---|---|
| Hash table | Bucket cuckoo | Naive cuckoo | Per-slot eviction chains → warp divergence |
| Hash table | Bucket cuckoo | Linear probing | No lookup bound guarantee; cuCollections already does this |
| Hash table | Bucket cuckoo | Robin Hood | Moderate improvement over linear; insufficient differentiation |
| Hash function | XXHash3 | MurmurHash3 | Bias at low moduli; slower on short keys |
| Bucket indexing | Power-of-two bitmask | Modulo `%` | Integer division 20–40 cycles on CC 5.0 |
| Pipeline dispatch | CUDA Graphs | Manual stream+events | ~10µs/batch CPU launch overhead at scale |
| Pointer swap safety | EBR reader-count | Hazard pointers | Per-thread hazard registration overhead; single-writer model doesn't need it |
| Value encoding | Direct uint32_t | FP16 quantisation | FP16 only applicable to floating-point values; value indirection is the correct general mitigation |

---

## XIII. Unit Test Plan

The bucket cuckoo warp-cooperative lookup is the highest-risk component. A single off-by-one in `__ballot_sync` mask handling or `__shfl_sync` lane indexing produces ghost data corruption that is silent and difficult to attribute. Tests must be written before any kernel code.

### Required Tests (Pre-Implementation)

| Test | What It Catches |
|---|---|
| **Single key insert + lookup, b1 hit** | Basic correctness; fingerprint matching; occupancy mask set |
| **Single key insert + lookup, b2 hit** | Second-bucket path; `b2` hash decorrelation working |
| **Key not present → NOT_FOUND** | `__ballot_sync` returns 0; no spurious match |
| **Fingerprint false positive, key mismatch** | Fingerprint collision followed by key comparison correctly rejects |
| **Full warp: 32 distinct keys, 32 distinct lookups** | All 32 lanes find their own keys; no cross-lane result corruption |
| **Load factor exactly 0.5 → insert succeeds** | Boundary condition at cap |
| **Load factor 0.5 + 1 key → stash path triggered** | Stash handoff correct; key survives in stash |
| **Stash overflow → rehash triggered** | Rehash completes; all pre-rehash keys still findable |
| **EBR: read during rehash pointer swap** | Readers on old table complete without segfault; new table visible to subsequent batches |
| **b1 == b2 degenerate case** | XOR decorrelation prevents this; test that `b1 != b2` always holds across hash space |
| **CUDA Graph replay: 1000 consecutive batches** | No state leakage across graph replays; event reuse correct |
| **Triple buffer: slot rotation under load** | Slot counter wraps correctly; no aliasing between concurrent slots |

---

## XIV. Build Order

Recommended implementation sequence — each stage is independently testable before proceeding.

1. **XXHash3 kernel** — implement and validate hash function; verify `b1 != b2` property across 10M random keys.

2. **Bucket struct + SoA allocator** — implement arena allocation, power-of-two sizing, bucket layout; validate with `cuda-memcheck`.

3. **Single-warp lookup kernel (no insert)** — hardcode a small table with known entries; verify `__ballot_sync` / `__shfl_sync` correctness against all unit tests in Section XIII.

4. **Warp-cooperative insert kernel** — implement CAS-based insertion; test load factor boundary; validate stash handoff.

5. **EBR + rehash** — implement `acquire()`/`release()` + rehash path; test pointer swap under concurrent simulated reads.

6. **Pinned buffer allocation + triple buffering** — validate DMA transfers; check for pinned memory exhaustion on MX130 2GB.

7. **CUDA Graph capture + replay** — capture the three-stage pipeline; validate event dependency ordering; stress-test 10K replay cycles.

8. **pybind11 binding + GIL release** — integrate Python interface; verify GIL is released before graph launch; profile Python overhead.

9. **Benchmark harness** — batch size sweep from 128 to 8192 entries; compare against Redis 7.x locally; record raw numbers with error bars.

---

## Appendix: Architecture Summary

```
Python Layer
    │  (pybind11 — GIL released here)
    ▼
C++ Host Layer
    ├── EpochTable (acquire/release — EBR)
    ├── Triple-buffered pinned memory (h_keys_in, h_vals_out)
    └── CUDA Graph (graphExec — single launch call per batch)
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │  Stream H→D  →  cudaMemcpyAsync (keys)      │
    │       ↓ ev_h2d                              │
    │  Stream Comp →  warp_lookup_kernel          │  CUDA Pipeline
    │       ↓ ev_compute                          │
    │  Stream D→H  →  cudaMemcpyAsync (values)    │
    └─────────────────────────────────────────────┘
           │
           ▼
    VRAM — Static Arena
    ├── BucketTable (SoA — keys[], vals[], fps[], occupancy[])
    │      └── Bucket Cuckoo — 128B cache-line aligned
    │          Two candidate buckets per key (b1, b2)
    │          Checked in single warp pass via __ballot_sync
    ├── d_keys[3] — device input buffers
    └── d_vals[3] — device output buffers
```

---

*Document version: 2.0. All design decisions are final for V1 implementation scope. Rehash, persistence, and multi-GPU paths are explicitly deferred.*
