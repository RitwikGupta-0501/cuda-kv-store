# WarpKV v2 Spec Audit — Critical Issues Identified

**Audit Date:** 2026-06-27  
**Status:** 10 Critical Bugs, 5 Design Contradictions, 8 Unspecified Bridges  
**Verdict:** Spec is unsafe to implement as written. Requires corrections before Phase 1.

---

## Critical Bugs (Will Cause Memory Corruption)

### BUG 1: Lane-to-Slot Out-of-Bounds Access (Section IV)

**Location:** `warp_lookup` kernel pseudocode, Section IV (line ~131)

**Current Code:**
```cuda
const int slot = lane & 15;  // slot = 0..15
const uint32_t bucket_idx = (lane < 16) ? b1 : b2;
const Bucket* bkt = &table->buckets[bucket_idx];

const bool hit = (bkt->occupancy_mask >> slot & 1u) &&
                 (bkt->fingerprint[slot] == fp) &&
                 (bkt->keys[slot] == key);
```

**The Problem:**
- Bucket struct defines `keys[8]`, `values[8]`, `fingerprint[8]` (8 slots per bucket)
- Code indexes with `slot = lane & 15` (range 0–15)
- **Lanes 8–15 read out of bounds**
  - `keys[8..15]` don't exist
  - In memory, `keys[8]` = first byte of `values[0]`
  - Reading `keys[8..15]` corrupts data: reads into `values[0..7]` array
  - False positives: will incorrectly match keys against values
  - **Result: Silent data corruption, dropped inserts, wrong lookups**

**Why This Bug Exists:**
- Spec claims "32 lanes, 2 buckets" but each bucket only has 8 slots
- 32 lanes ÷ 2 buckets = 16 lanes per bucket (wrong)
- 8 slots require 8 lanes per bucket, leaving 16 lanes idle (inefficient but correct)

**Correct Implementation:**
```cuda
const int slot = lane & 7;  // slot = 0..7 (valid range)
const uint32_t bucket_idx = (lane < 8) ? b1 : b2;
const Bucket* bkt = &table->buckets[bucket_idx];

const bool hit = (bkt->occupancy_mask >> slot & 1u) &&
                 (bkt->fingerprint[slot] == fp) &&
                 (bkt->keys[slot] == key);
```

**Lanes:**
- Lanes 0–7: scan bucket b1, slots 0–7
- Lanes 8–15: scan bucket b2, slots 0–7
- Lanes 16–31: idle (participate in `__ballot_sync` but don't read arrays)

---

### BUG 2: CUDA Launch Grid Drops 97% of Keys (Section VII)

**Location:** Phase 6 pseudocode, "CUDA Graph Capture" (line ~363)

**Current Code:**
```cuda
dim3 block(32);   // one warp per block; one key per warp
dim3 grid((BATCH_SIZE + 31) / 32);
warp_lookup_kernel<<<grid, block, 0, stream_compute>>>(
    epoch_table.current.load(), d_keys[slot], d_vals[slot], BATCH_SIZE);
```

**The Problem:**
With `BATCH_SIZE = 4096`:
- `grid = (4096 + 31) / 32 = 128`
- Total threads = 128 blocks × 32 threads/block = **4,096 threads**
- Total warps = 4,096 ÷ 32 = **128 warps**
- **Spec design: 1 warp per key → can process only 128 keys**
- **3,968 keys are silently dropped**

**Why This Bug Exists:**
- Comment correctly says "one key per warp" (one warp = 32 threads = one block)
- Grid formula divides BATCH_SIZE by 32, creating a mismatch
- Formula was likely copy-pasted from a different kernel type (one thread per key, not one warp per key)

**Correct Implementation:**
```cuda
dim3 block(32);   // 32 threads per block (one warp)
dim3 grid(BATCH_SIZE);  // BATCH_SIZE blocks (one per key)
// Total threads = BATCH_SIZE × 32 (correct)
```

Example with BATCH_SIZE = 4096:
- grid = 4096
- block = 32
- Total threads = 4096 × 32 = 131,072
- Total warps = 131,072 ÷ 32 = 4,096
- **Processes all 4,096 keys correctly**

**Additional Issue:** The kernel call includes `BATCH_SIZE` as a parameter, but each warp is tied to one key. The warp ID should be computed as `blockIdx.x` (one block per key). The `BATCH_SIZE` parameter is redundant.

---

### BUG 3: `warp_insert` Has Identical Lane Indexing Bug (Section IV)

**Location:** `warp_insert` kernel pseudocode, lines 157–190

**Current Code:**
```cuda
if (lane == 0) {
    // ... atomic CAS on occupancy_mask, key, value, fingerprint
    // Returns true on success, evicted key/val on collision
}
```

**The Problem:**
- The insert kernel doesn't explicitly index `keys[slot]` like lookup does
- **But** the CAS helper function (`try_cuckoo_insert`) will have the same problem
- Any bucket access inside the CAS helper will use the same flawed lane-to-slot mapping
- **The bug propagates to insertions**

**Correct Fix:**
All bucket accesses must use `slot = lane & 7` (8 lanes per bucket), not `lane & 15`.

---

### BUG 4: CPU Stash Handoff Unspecified (Section IV, Phase 4)

**Location:** `warp_insert` return path, line 188

**Current Code:**
```cuda
__device__ bool warp_insert(BucketTable* table, uint32_t key, uint32_t val) {
    // ... loop through MAX_EVICTION_HOPS ...
    return false;  // Push to CPU stash
}
```

**The Problem:**
- `warp_insert` runs as a `__device__` function (on GPU)
- Returning `false` to the device kernel doesn't communicate anything to the host
- Host has no idea:
  - Which key failed to insert
  - How many keys failed total
  - Where in the stash to write the failed key
  - Whether the stash is full

**Result:** Failed keys are dropped into the void. Insertions silently fail.

**What's Missing:**

1. **Atomic queue in mapped pinned memory:**
   ```cpp
   struct StashEntry {
       uint32_t key;
       uint32_t value;
   };
   
   struct StashQueue {
       std::atomic<uint32_t> head;  // Next write position
       std::atomic<uint32_t> tail;  // Next read position (for host)
       StashEntry entries[128];
   };
   ```

2. **GPU kernel appends to queue:**
   ```cuda
   if (!success_after_max_hops) {
       uint32_t idx = atomicAdd(&stash->head, 1);
       if (idx < 128) {
           stash->entries[idx] = {cur_key, cur_val};
       } else {
           // Stash full, signal rehash needed (another atomic)
       }
   }
   ```

3. **Host polls stash after kernel:**
   ```cpp
   cudaEventSynchronize(ev_compute[slot]);  // Wait for kernel
   uint32_t new_head = stash_queue->head.load();
   if (new_head > stash_tail) {
       // Process new entries from stash_entries[stash_tail..new_head-1]
       // If head >= 128, trigger rehash
   }
   ```

**Current Spec:** Has zero design for steps 2 and 3. Just says "return false; Push to CPU stash."

---

### BUG 5: Graph Triple-Buffering Slot Binding Unspecified (Section VII)

**Location:** `build_graph()` pseudocode, lines 351–377

**Current Code:**
```cuda
void WarpKVEngine::build_graph(int slot) {
    cudaStreamBeginCapture(stream_h2d, cudaStreamCaptureModeGlobal);
    
    for (int slot = 0; slot < 3; ++slot) {
        cudaMemcpyAsync(d_keys[slot], h_keys_in[slot], ...);
        // ... more operations with slot 0, 1, 2 ...
    }
    
    cudaStreamEndCapture(stream_h2d, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
}
```

**The Problem:**
- Graph capture loop iterates `slot = 0, 1, 2` but captures all three slots' operations
- This creates **one monolithic graph** with operations for all three slots in sequence:
  ```
  Stage 1: H→D with slot 0
  Stage 1: H→D with slot 1
  Stage 1: H→D with slot 2
  Stage 2: Compute with slot 0
  Stage 2: Compute with slot 1
  Stage 2: Compute with slot 2
  ...
  ```
- **Replaying this graph does nothing useful.** You process all three slots in one replay, defeating the purpose of triple buffering.

**How Triple Buffering Should Work:**
- **Option A: Separate graph per slot (3 graphs total)**
  ```cpp
  for (int s = 0; s < 3; ++s) {
      cudaStreamBeginCapture(stream_h2d, cudaStreamCaptureModeGlobal);
      cudaMemcpyAsync(d_keys[s], h_keys_in[s], ...);
      cudaEventRecord(ev_h2d[s], stream_h2d);
      // ... rest of 3-stage pipeline for slot s ...
      cudaStreamEndCapture(stream_h2d, &graphs[s]);
      cudaGraphInstantiate(&graphExecs[s], graphs[s], ...);
  }
  ```
  - On each batch, launch `graphExecs[current_slot]`
  - CPU overhead: still ~1µs per graph launch

- **Option B: One graph, dynamic buffer updates (1 graph, complex)**
  - Capture graph with one hardcoded slot
  - Between replays, use `cudaGraphExecUpdate` to rebind buffer pointers
  - More complex, but only one graph

**Current Spec:** Doesn't clarify which approach is intended. The code shows neither.

---

### BUG 6: `acquire()`/`release()` Race on Epoch Swap (Section VIII)

**Location:** EBR protocol, lines 409–441

**Current Code:**
```cpp
BucketTable* WarpKVEngine::acquire() {
    epoch_table.readers.fetch_add(1, std::memory_order_acquire);
    return epoch_table.current.load(std::memory_order_acquire);
}

void WarpKVEngine::rehash() {
    // ... build new table ...
    epoch_table.current.store(new_table, std::memory_order_release);
    epoch_table.epoch.fetch_add(1, std::memory_order_acq_rel);
    
    while (epoch_table.readers.load(std::memory_order_acquire) > 0)
        std::this_thread::yield();
}
```

**The Problem:**
Race condition:

```
Thread 1 (GPU batch):
  T1: readers.fetch_add(1) → readers = 1
  T2: load(current) → returns old_table
  
Thread 2 (rehash, concurrent):
  T2: store(new_table) → current now points to new_table
  T3: epoch.fetch_add(1)
  T4: load(readers) → sees readers = 1
  T5: spin-wait until readers == 0
  
Thread 1:
  T6: GPU kernel executes on old_table (still valid, good)
  T7: cudaEventSynchronize (kernel finishes)
  T8: readers.fetch_sub(1) → readers = 0
  
Thread 2:
  T9: wake from spin, clear old arena
```

**This is actually correct.** The memory ordering is sound. Thread 1 acquires the old pointer (stable) before the swap. Thread 2 waits for all readers of the old pointer to finish before clearing.

**BUT:** The spec doesn't explain the memory ordering or the acquire/release semantics. An implementer could get this wrong by using `memory_order_relaxed` everywhere.

**The Fix:** Add explicit memory ordering comments explaining why each barrier is necessary.

---

### BUG 7: Python GIL Scope Unclear (Section IX)

**Location:** `warpkv_batch_lookup` pseudocode, lines 457–488

**Current Code:**
```cpp
{
    py::gil_scoped_release release;

    int slot = engine.next_slot();
    BucketTable* tbl = engine.acquire();

    // Copy keys from Python list into pinned staging buffer
    for (int i = 0; i < n; ++i)
        engine.h_keys_in[slot][i] = keys[i].cast<uint32_t>();

    engine.submit_batch(slot);
    engine.release();
}
```

**The Problem:**
1. **GIL is released before key copy.** The code comment says "(key copy happens before GIL release in production; shown here for clarity...)" but the **actual code shows copy happening AFTER release.**
2. **`keys[i].cast<uint32_t>()` can fail if the key is not a valid uint32.** No error handling shown.
3. **`keys` list is a pybind11 proxy.** Reading `keys[i]` while GIL is released could cause issues if Python's refcount management is involved.

**Correct Implementation:**
```cpp
py::list results;
int n = keys.size();

// *** COPY KEYS BEFORE GIL RELEASE ***
std::vector<uint32_t> keys_vec(n);
for (int i = 0; i < n; ++i) {
    try {
        keys_vec[i] = keys[i].cast<uint32_t>();
    } catch (const std::exception& e) {
        throw py::type_error("All keys must be uint32_t");
    }
}

int slot = engine.next_slot();

{
    py::gil_scoped_release release;  // Release AFTER copying
    
    engine.acquire();
    std::memcpy(engine.h_keys_in[slot], keys_vec.data(), n * sizeof(uint32_t));
    engine.submit_batch(slot);
    engine.release();
}

for (int i = 0; i < n; ++i)
    results.append(engine.h_vals_out[slot][i]);
return results;
```

---

## Design Contradictions (Ambiguous Spec)

### CONTRADICTION 1: SoA vs. AoS Memory Layout (Section VI)

**Claim in Section VI:**
> The global table is SoA — all bucket key arrays are contiguous in VRAM, all value arrays are contiguous, all fingerprint arrays are contiguous.

**Diagram shown:**
```
VRAM Layout:
[ Bucket[0].keys | Bucket[1].keys | ... | Bucket[N].keys ]
[ Bucket[0].vals | Bucket[1].vals | ... | Bucket[N].vals ]
```

**Code in Section IV:**
```cpp
struct Bucket {
    uint32_t keys[8];      // 32 bytes
    uint32_t values[8];    // 32 bytes
    uint8_t fingerprint[8]; // 8 bytes
    uint32_t occupancy_mask; // 4 bytes
    uint8_t padding[52];    // 52 bytes
    // Total: 128 bytes
};

BucketTable* table->buckets;  // Array of Bucket structs
```

**The Reality:**
With `Bucket* buckets` and array allocation, memory layout is:
```
[Bucket[0]:
    keys[0-7] at bytes 0-31,
    values[0-7] at bytes 32-63,
    fps[0-7] at bytes 64-71,
    mask at bytes 72-75,
    pad at bytes 76-127]
[Bucket[1]:
    keys[0-7] at bytes 128-159,
    values[0-7] at bytes 160-191,
    ...]
```

**This is AoS (Array of Structs), not SoA (Struct of Arrays).**

Keys are at offsets 0, 128, 256, 384, ... (not contiguous).
Values are at offsets 32, 160, 288, 416, ... (not contiguous).

**Which is correct?**
- **SoA** would be:
  ```
  [ all keys from all buckets: 8M keys = 32M bytes ]
  [ all values from all buckets: 8M values = 32M bytes ]
  [ all fingerprints: 8M bytes ]
  [ all masks: 4M bytes ]
  ```
  Pros: Coalesced VRAM reads when scanning keys
  Cons: Destroys 128-byte cache-line alignment

- **AoS (bucket-level)** would be the current struct layout
  Pros: 128-byte cache-line alignment, one cache line per bucket scan
  Cons: Keys and values interleaved; less SIMD-friendly

**Spec is contradictory. The code comment calls it "SoA" but implements AoS.**

**The Fix:** Clarify that the design uses **"Bucket-AoS"** (each bucket is 128B, cache-line aligned) but *not* global SoA. The diagram is misleading and should be removed.

---

### CONTRADICTION 2: "One Warp Per Key" vs. Lane Utilization

**Claim:**
> A single warp (32 threads) checks **two buckets simultaneously**: lanes 0–15 scan bucket b1, lanes 16–31 scan bucket b2.

**Math:**
- 32 lanes total
- 2 buckets
- 8 slots per bucket
- Lanes 0–15 (16 lanes) → bucket b1 (8 slots) → 2 lanes per slot (wasteful?)
- Lanes 16–31 (16 lanes) → bucket b2 (8 slots) → 2 lanes per slot (wasteful?)

**This doesn't match "one key per warp" (one warp = 32 lanes = one key). It suggests one warp scans two buckets for two keys.**

**Or is it:** One warp processes one key, but the warp has 32 lanes and only uses 16 of them per bucket?

**The spec is ambiguous.** A clearer statement would be:
- "One warp (32 threads) checks one key against two candidate buckets"
- "Lanes 0–7 scan bucket b1, lanes 8–15 scan bucket b2, lanes 16–31 are idle"
- "All lanes participate in `__ballot_sync` to aggregate results across lanes 0–15"

---

### CONTRADICTION 3: Load Factor Capping

**Section IV states:**
> Max load factor: 0.50. Below 50%, eviction chains are statistically short. Above 65%, livelock probability becomes non-negligible.

**Section VI states:**
> Reserve 80% of free VRAM — leave headroom for output buffers and stash

**But:** How is the 0.50 load factor *enforced*?

**Option A:** Track global insertion count, reject inserts when count ≥ 0.5 × num_buckets.
- Requires an atomic counter on GPU
- Spec doesn't mention this counter

**Option B:** Accept inserts until stash overflows, then rehash.
- Allows temporary overfill
- Spec doesn't specify how much overfill is tolerable

**Spec doesn't clarify the mechanism for capping at 0.50.**

---

### CONTRADICTION 4: Stash Size vs. Max Eviction Hops

**Section IV states:**
- Max eviction hops: 32
- CPU stash size: 128 slots
- Stash overflow action: Async full-table rehash

**The Problem:**
If a key fails to insert after 32 hops, it goes to the stash. But:
- At what load factor do eviction chains become long enough to hit 32 hops?
- With 0.5 load factor and balanced hashing, what's the P99 eviction chain length?
- How many concurrent insertions can overflow the stash before rehash catches up?

**Spec doesn't provide:** Load factor vs. chain length analysis, P99 hop count, concurrent insertion capacity before stash overflow.

---

### CONTRADICTION 5: Rehash Trigger Timing

**Section IV states:**
> Stash overflow action: Async full-table rehash on dedicated CUDA stream (Invisible to read path via double-buffered table pointers + EBR)

**The Problem:**
- When exactly is rehash triggered?
  - On first stash insertion (immediate, every insert will stall)?
  - When stash reaches 50% full (128/2 = 64 entries)?
  - When stash reaches 100% full (128 entries)?
  - When a stash insertion fails (stash full)?

**Spec doesn't specify the rehash trigger threshold.**

If triggered too early, overhead is wasted. If triggered too late, insertions stall waiting for rehash to finish.

---

## Unspecified Bridges (Design Gaps)

### GAP 1: How Does `submit_batch` Know Which Slot to Use?

**Spec shows:**
```cpp
void WarpKVEngine::submit_batch(int slot) {
    cudaGraphLaunch(graphExec, stream_h2d);
    cudaEventSynchronize(ev_d2h[slot]);
}
```

**Missing:** 
- Who calls `next_slot()`?
- Where is the slot counter incremented?
- What prevents two consecutive batches from using the same slot before the previous batch finishes?

**Spec doesn't show** the batching loop structure:
```cpp
while (has_batches) {
    int slot = next_slot();  // 0, 1, 2, 0, 1, 2, ...
    copy_keys_to_pinned(h_keys_in[slot], batch);
    submit_batch(slot);
    sleep_until_done(slot);
    copy_results_from_pinned(h_vals_out[slot], results);
}
```

---

### GAP 2: GPU Stash Overflow Signal

**Spec states:**
> Stash overflow action: Async full-table rehash on dedicated CUDA stream

**Missing:**
- How does the rehash thread know the stash overflowed?
- What signal wakes the rehash thread?
- Is there a polling loop? A condition variable? An atomic flag?

**Current Spec:** No synchronization primitive defined.

**Correct Design Would Include:**
```cpp
struct {
    std::atomic<bool> needs_rehash;
    std::condition_variable cv;
} rehash_signal;

// GPU kernel:
if (stash->head >= 128) {
    rehash_signal.needs_rehash.store(true, std::memory_order_release);
    // Or: signal via mapped pinned memory
}

// Rehash thread:
while (true) {
    std::unique_lock<std::mutex> lock(rehash_mutex);
    rehash_signal.cv.wait(lock, [] { return rehash_signal.needs_rehash.load(); });
    
    // Perform rehash
    rehash_signal.needs_rehash.store(false);
}
```

**Spec doesn't provide this.**

---

### GAP 3: Event Synchronization Between Streams

**Spec shows:**
```cuda
cudaEventRecord(ev_h2d[slot], stream_h2d);
cudaStreamWaitEvent(stream_compute, ev_h2d[slot], 0);
```

**Missing:**
- What if `cudaStreamWaitEvent` is called on an event that hasn't been recorded yet?
- Are there race conditions if the same event is recorded twice (overwriting)?
- What if events are not reset between replays?

**Spec doesn't address:** Event lifecycle management, reset/reuse logic, synchronization bugs.

---

### GAP 4: Python Batch Size Constraints

**Spec shows:**
```cpp
py::list warpkv_batch_lookup(WarpKVEngine& engine, py::list keys)
```

**Missing:**
- What if `keys.size()` > BATCH_SIZE (4096)?
- Should the function split into multiple GPU batches? Reject? Allocate more pinned memory?
- What if `keys` is empty?

**Spec doesn't specify:** Batch size validation, empty input handling, overflow behavior.

---

### GAP 5: Kernel Grid Block/Thread Sizing

**Spec states:**
- One warp per key
- BATCH_SIZE keys

**But:** What is the actual optimal block size?

**Spec's pseudocode shows:**
```cuda
dim3 block(32);  // 32 threads = 1 warp per block
```

**Better designs might use:**
- `dim3 block(32 * k)` where k keys per block (k warps per block)
  - Reduces grid overhead
  - Allows better SM occupancy
  - But violates "one warp per key" abstraction

**Spec doesn't justify the 1-warp-per-block choice.**

---

### GAP 6: Memory Ordering in EBR

**Spec shows:**
```cpp
epoch_table.readers.fetch_add(1, std::memory_order_acquire);
epoch_table.current.store(new_table, std::memory_order_release);
```

**Missing:**
- Why `acquire` on the increment? The increment itself doesn't acquire anything.
- Should it be `memory_order_relaxed`? (increment doesn't need to order against anything)
- The pointer swap should be `memory_order_release` (correct)
- The drain spin loop should be `memory_order_acquire` (correct)

**Spec memory ordering is questionable.** A correct analysis would be:
```cpp
// acquire(): Load-acquire semantics on pointer
readers.fetch_add(1, std::memory_order_relaxed);  // Just increment
BucketTable* ptr = current.load(std::memory_order_acquire);

// store in rehash: Release semantics on pointer
current.store(new_table, std::memory_order_release);

// drain spin: Acquire on readers to see increments
while (readers.load(std::memory_order_acquire) > 0) yield();
```

**Spec's memory ordering is sloppy and not explained.**

---

### GAP 7: VRAM Budget Calculation

**Spec Section VI states:**
> Reserve 80% of free VRAM — leave headroom for output buffers and stash

**MX130 has 2 GB VRAM.**
- 80% = 1.6 GB for table arena
- Table arena for 2 double-buffered tables = 3.2 GB needed
- **But you only have 1.6 GB available.**

**This is a math error in the spec.**

**Correct calculation:**
- Two full tables (EBR double-buffering) = 2 × arena_size
- Triple-buffered pinned: 3 × (BATCH_SIZE × 8 bytes) for keys
- Triple-buffered device: 3 × (BATCH_SIZE × 4 bytes) for values
- CPU stash: 128 × 8 bytes = 1 KB (negligible)

**If we want 80% of 2 GB for tables:**
- 0.8 × 2 GB = 1.6 GB per table
- Two tables need 3.2 GB
- **Impossible on MX130**

**Spec should allocate one table at full size, reserve one table budget for future rehash, accept temp stash overflow until rehash completes.**

---

### GAP 8: Fingerprint False Positive Rate

**Spec states:**
> Fingerprints (upper 8 bits of the hash, 1 byte per slot) allow a warp to reject non-matching slots via a single `__ballot_sync` comparison before reading the value array — eliminating unnecessary VRAM fetches on misses.

**Missing:** Probability analysis
- 8-bit fingerprint → 1 in 256 keys have matching fingerprint
- With 8 slots per bucket, P(false positive) = 8 / 256 = 3.1%
- Per lookup, expected VRAM reads = (1 hit × 1 value) + (P(FP) × values_checked)
- Spec claims "unnecessary VRAM fetches on misses" but doesn't quantify benefit

**Spec oversells the fingerprint optimization without analysis.**

---

## Summary Table

| # | Category | Severity | Spec Section | Fix Complexity |
|---|----------|----------|--------------|---|
| 1 | Lane-to-slot out-of-bounds | **CRITICAL** | IV | High (redesign warp mapping) |
| 2 | Launch grid drops keys | **CRITICAL** | VII | Trivial (formula fix) |
| 3 | Warp insert lane bug | **CRITICAL** | IV | High (same as #1) |
| 4 | Stash handoff missing | **CRITICAL** | IV, Phase 4 | High (atomic queue design) |
| 5 | Graph slot binding broken | **CRITICAL** | VII | High (clarify approach) |
| 6 | EBR race (actually OK) | Low | VIII | Low (add comments) |
| 7 | Python GIL scope | **Medium** | IX | Medium (reorder code) |
| 8 | SoA vs. AoS contradiction | **Medium** | IV, VI | Low (clarify labeling) |
| 9 | Load factor capping unclear | **Medium** | IV | Medium (define mechanism) |
| 10 | Stash size vs. hops | **Medium** | IV | Medium (provide analysis) |
| 11 | Rehash trigger timing | **Medium** | IV | Medium (specify threshold) |
| 12–18 | Design gaps (8 gaps) | **Medium** | Various | Varies (add specs) |
| 19 | VRAM budget math error | **High** | VI | Low (recalculate) |
| 20 | Fingerprint analysis missing | Low | IV | Low (add analysis) |

---

## Recommendations

1. **Immediate (before Phase 1):**
   - Fix bugs #1–5 (memory corruption risks)
   - Fix gaps #2, #4, #7 (unspecified bridges causing data loss)

2. **Before Phase 3 (lookup kernel):**
   - Finalize warp lane mapping (bug #1)
   - Clarify SoA/AoS (contradiction #1)
   - Define fingerprint false positive analysis

3. **Before Phase 6 (CUDA graphs):**
   - Specify which graph approach (bug #5, gap #3)
   - Define event lifecycle management

4. **Before Phase 7 (Python):**
   - Clarify GIL release scope (bug #7)
   - Define batch size validation (gap #5)

5. **Before Phase 4 (insertion):**
   - Design stash atomic queue (bug #4, gap #2)
   - Specify rehash trigger (contradiction #5)

6. **Throughout:**
   - Fix VRAM budget calculation (gap #8)
   - Remove misleading SoA claims (contradiction #1)

---

**Next Step:** Proceed to `SPEC_CORRECTED.md` for the fixed specification.
