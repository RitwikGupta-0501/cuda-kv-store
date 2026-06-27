# WarpKV v2 — Corrected Implementation Plan

**Version:** 2.0.1 (Audit-Corrected)  
**Status:** Production-ready specification; ready for implementation  
**Based on:** SPEC_CORRECTED.md (all 20 bugs fixed)  

---

## Executive Summary

This plan implements WarpKV based on the **corrected specification**. Key differences from the broken plan:

✅ **Lanes 0–7 per bucket** (not 16) — eliminates out-of-bounds access  
✅ **Launch grid formula: `grid = BATCH_SIZE`** (not `(BATCH_SIZE+31)/32`) — processes all keys  
✅ **Separate graphs per slot** (not monolithic) — proper triple-buffering  
✅ **Mapped pinned stash queue** — GPU→CPU handoff for failed insertions  
✅ **Python keys copied before GIL release** — prevents corruption  
✅ **All gaps closed** — rehash triggers, stash signaling, event management specified  

**Effort:** 715 hours (18 weeks 1-FTE, 9 weeks 2-FTE) — same as broken plan, but now correct.

---

## Phase 0: Foundation & Validation (Week 1)

**Deliverable:** Hardware baseline, build system, test infrastructure ready.

**Key Tasks:**
1. Verify MX130 hardware: `nvidia-smi`, PCIe bandwidth measurement (~6 GB/s target)
2. CMakeLists.txt with gtest, CUDA CC 5.0, C++17
3. Test framework: `tests/unit/`, `tests/integration/`, `benchmarks/`
4. Document hardware profile in `docs/hardware_baseline.md`

**Risk Mitigation:** Baseline PCIe BW on actual hardware (may differ from spec).

**Estimated hours:** 40–60h

---

## Phase 1: XXHash3 Kernel (Week 1–2)

**Deliverable:** Hash function, validated no modulo, 8-bit fingerprint analysis.

**Key Corrections from Original Plan:** None (original spec was correct here).

**Key Tests:**
- ✓ Known key→hash mappings
- ✓ Avalanche property (chi-square)
- ✓ `b1 != b2` across 10M keys (XOR decorrelation verified)
- ✓ Fingerprint false positive rate: 3.1% (8-bit fingerprint on 8-slot bucket)

**Estimated hours:** 20–30h

---

## Phase 2: Bucket Cuckoo Data Structure (Week 2–3)

**Deliverable:** Bucket-AoS layout (cache-line aligned), arena allocator, VRAM budget calculation.

**Key Corrections:**
1. **Bucket layout:** 128 bytes, verified no out-of-bounds padding
2. **Memory layout:** Bucket-AoS (NOT global SoA) — clarified terminology
3. **VRAM budget:** 750 MB × 2 tables (EBR) + buffers, realistic on 2 GB MX130
4. **Power-of-two sizing:** Round down to bucket count, store mask

**Key Tests:**
- ✓ Bucket struct exactly 128 bytes (verified with `sizeof()`)
- ✓ Occupancy mask bit operations
- ✓ Arena allocation: correct bucket count (power of 2)
- ✓ No buffer overruns
- ✓ VRAM allocation stays within 1.6 GB budget

**Estimated hours:** 50–80h

---

## Phase 3: Single-Warp Lookup Kernel (Week 3–4) ⚠️ CRITICAL

**Deliverable:** Zero-divergence lookup, lanes 0–7 per bucket, no out-of-bounds.

**Key Corrections (CRITICAL):**
1. **Lane-to-bucket mapping (CORRECTED):**
   ```cuda
   const int slot = lane & 7;  // CORRECTED: 0-7, not 0-15
   const uint32_t bucket_idx = (lane < 8) ? b1 : ((lane < 16) ? b2 : UINT32_MAX);
   
   // Skip for idle lanes
   bool hit = false;
   if (bucket_idx != UINT32_MAX) {
       const Bucket* bkt = &table->buckets[bucket_idx];
       hit = (bkt->occupancy_mask >> slot & 1u) &&
             (bkt->fingerprint[slot] == fp) &&
             (bkt->keys[slot] == key);  // SAFE: slot 0-7, keys[0-7] valid
   }
   ```

2. **Launch grid formula (CORRECTED):**
   ```cuda
   dim3 block(32);      // 1 warp per block
   dim3 grid(BATCH_SIZE); // BATCH_SIZE blocks (CORRECTED, not (BATCH_SIZE+31)/32)
   ```
   Example: BATCH_SIZE=4096 → 4096 blocks → 131K threads → 4096 warps → 4096 keys ✓

3. **Result broadcast:** Winner guaranteed in lanes 0–15 (only those can match)

**Key Tests:**
- ✓ No out-of-bounds reads (verify with AddressSanitizer)
- ✓ Single key lookups: b1 hit, b2 hit
- ✓ Key not found → NOT_FOUND
- ✓ Fingerprint collision: key comparison rejects
- ✓ Full 16-lane batch (lanes 0-15 all find their keys, no corruption)
- ✓ Idle lanes (16-31) don't corrupt ballot results
- ✓ Worst-case 2 cache-line reads verified with profiler

**Profiling Targets:**
- L2 hit rate: ≥95% at batch size 4096
- Cache lines read: ≤2 per lookup
- Warp divergence: 0% on lookup path

**Risk Mitigation:** This is the highest-risk phase. Extensive unit tests validate:
- Lane indexing never goes out of bounds
- `__ballot_sync` aggregates correctly across 32 lanes
- `__shfl_sync` broadcasts from correct lane
- Memory ordering is correct

**Estimated hours:** 60–100h

---

## Phase 4: Warp-Cooperative Insert & Stash Handoff (Week 4–5)

**Deliverable:** CAS-based insertion, stash queue (CORRECTED), rehash trigger.

**Key Corrections (CRITICAL):**
1. **Stash queue (CORRECTED) — Mapped pinned memory:**
   ```cpp
   struct StashQueue {
       std::atomic<uint32_t> head;         // GPU writes
       std::atomic<uint32_t> tail;         // Host reads
       std::atomic<uint32_t> needs_rehash; // GPU sets if head >= 64
       struct { uint32_t key, value; } entries[128];
   };
   
   // Host polls after each batch:
   uint32_t new_head = stash->head.load();
   while (stash->tail < new_head && stash->tail < 128) {
       // Process stash_queue->entries[stash->tail++]
   }
   ```

2. **GPU stash append (CORRECTED):**
   ```cuda
   if (lane == 0) {
       uint32_t idx = atomicAdd(&stash->head, 1);
       if (idx < 128) {
           stash->entries[idx] = {cur_key, cur_val};
           return true;
       } else {
           atomicOr(&stash->needs_rehash, 1u);
           return false;  // Stash full, rehash will handle
       }
   }
   ```

3. **Rehash trigger (CORRECTED):**
   - Trigger: `head >= 64` (50% stash)
   - Not: when first entry added (too aggressive)
   - Not: when full (too late, may drop entries)

4. **Rehash signal (CORRECTED) — Condition variable:**
   ```cpp
   std::condition_variable rehash_cv;
   std::atomic<bool> needs_rehash;
   
   // GPU sets flag
   atomicOr(&stash->needs_rehash, 1u);
   
   // Rehash thread waits
   rehash_cv.wait(lock, []{ return stash->needs_rehash.load(); });
   ```

**Key Tests:**
- ✓ Single insert + lookup retrieves value
- ✓ Eviction chain: 5+ evictions, all values retrievable
- ✓ Load factor 0.5: insert succeeds at boundary
- ✓ Load factor 0.5 + 1: key goes to stash, `head` incremented
- ✓ Stash append: GPU appends atomically, host reads without race
- ✓ Stash overflow (head = 64): `needs_rehash` flag set
- ✓ Multiple concurrent inserts: no corruption of stash entries

**Estimated hours:** 70–120h

---

## Phase 5: Epoch-Based Reclamation & Rehash (Week 5–6)

**Deliverable:** Double-buffered EBR, safe pointer swap, reader drain.

**Key Corrections:** Memory ordering clarified (acquire/release semantics documented).

**Key Tests:**
- ✓ Double-buffered tables: two arenas allocated at startup
- ✓ Reader count increment/decrement: correct ordering
- ✓ Pointer swap: old pointer stable for in-flight readers
- ✓ Rehash drains: waits until readers == 0 before clearing
- ✓ Concurrent reads during rehash: no crashes, no stale pointers
- ✓ Stress test: 10 reader threads, 1 rehash thread, 1000 cycles

**Estimated hours:** 50–80h

---

## Phase 6: CUDA Graphs & Triple-Buffered Pipeline (Week 6–7)

**Deliverable:** Three separate graphs, slot rotation, event lifecycle.

**Key Corrections (CRITICAL):**
1. **Three separate graphs (CORRECTED, not monolithic):**
   ```cpp
   // Build 3 graphs, one per slot
   for (int slot = 0; slot < 3; ++slot) {
       cudaStreamBeginCapture(stream_h2d, ...);
       // Capture full 3-stage pipeline for slot
       cudaMemcpyAsync(d_keys[slot], ...);
       // ... compute ...
       cudaMemcpyAsync(h_vals_out[slot], ...);
       cudaStreamEndCapture(..., &graphs[slot]);
       cudaGraphInstantiate(&graphExecs[slot], ...);
   }
   ```

2. **Slot rotation (CORRECTED):**
   ```cpp
   int slot = (current_slot++ % 3);  // 0 → 1 → 2 → 0 → 1 → ...
   cudaEventSynchronize(ev_d2h[slot]);  // Wait for prev batch on this slot
   // Now safe to reuse buffers for this slot
   cudaGraphLaunch(graphExecs[slot], stream_h2d);
   ```

3. **Event lifecycle (CORRECTED):**
   - Events created once at startup
   - `cudaEventRecord()` called once per replay (overwrites previous timestamp)
   - Never reset between replays
   - Safe to reuse across infinite graph replays

**Key Tests:**
- ✓ Three graphs instantiate correctly
- ✓ Slot rotation: 0 → 1 → 2 → 0 with no aliasing
- ✓ Event synchronization: correct slot events awaited
- ✓ Graph replay 1000 times: results correct, events reused safely
- ✓ Pipeline profiling: H→D, Compute, D→H times measured independently

**Estimated hours:** 50–80h

---

## Phase 7: Python Interface & GIL Bypass (Week 7–8)

**Deliverable:** pybind11 binding, key copy before GIL release, batch validation.

**Key Corrections (CRITICAL):**
1. **Copy keys BEFORE GIL release (CORRECTED):**
   ```cpp
   // Copy and validate BEFORE releasing GIL
   std::vector<uint32_t> keys_vec(n);
   for (int i = 0; i < n; ++i) {
       try {
           keys_vec[i] = keys[i].cast<uint32_t>();
       } catch (...) {
           throw py::type_error("Key must be uint32_t");
       }
   }
   
   // NOW release GIL
   {
       py::gil_scoped_release release;
       memcpy(h_keys_in[slot], keys_vec.data(), ...);
       submit_batch(slot);
   }
   ```

2. **Batch size validation (CORRECTED):**
   ```cpp
   if (n == 0) return py::list();  // Empty batch OK
   if (n > BATCH_SIZE) {
       throw py::value_error("Batch size " + std::to_string(n) + 
                             " exceeds limit " + std::to_string(BATCH_SIZE));
   }
   ```

3. **Python module:**
   ```cpp
   PYBIND11_MODULE(warpkv, m) {
       py::class_<WarpKVEngine>(m, "WarpKVEngine")
           .def(py::init<>())
           .def("lookup", &warpkv_batch_lookup)
           .def("insert", &warpkv_batch_insert);
   }
   ```

**Key Tests:**
- ✓ Module imports
- ✓ Engine initializes
- ✓ Batch lookup: 512 keys → correct results
- ✓ Empty batch: returns empty list
- ✓ Oversized batch: raises ValueError
- ✓ Invalid key type: raises TypeError
- ✓ GIL released: concurrent Python threads don't block CUDA

**Estimated hours:** 30–50h

---

## Phase 8: Comprehensive Testing (Week 8–9)

**Deliverable:** 80+ test cases, 100% pass rate, zero sanitizer errors.

**Test Categories:**
1. **Unit tests (50+):** Hash, bucket layout, kernels, EBR, graphs, Python
2. **Integration tests (20+):** End-to-end insert/lookup, pipeline, rehash, Python
3. **Stress tests (3):** 100K cycles, concurrent reads + rehash, stash overflow
4. **Property tests (5):** insert/lookup commute, load factor bounds, no duplicates

**Sanitizers:**
- ✓ AddressSanitizer: zero leaks/errors/out-of-bounds
- ✓ ThreadSanitizer: zero race conditions
- ✓ MemorySanitizer: zero uninitialized reads

**Spec Compliance Checklist:**
- ✓ Lane mapping: 0-7 per bucket (no 0-15)
- ✓ Launch grid: BATCH_SIZE blocks (no formula error)
- ✓ Stash queue: atomic, mapped pinned (no silent drops)
- ✓ EBR: reader count drained before clear (no use-after-free)
- ✓ Graphs: separate per slot (no slot aliasing)
- ✓ GIL: keys copied before release (no corruption)

**Estimated hours:** 60–100h

---

## Phase 9: Benchmark Suite & Competitive Analysis (Week 9–10)

**Deliverable:** Throughput, latency, L2 hit rate; comparison vs. Redis/FASTER.

**Benchmarks:**
1. **Throughput:** Batch sizes [128, 256, 512, 1024, 2048, 4096, 8192] → keys/sec
2. **Latency:** Single batch latency, P50/P99/P99.9 histogram
3. **L2 hit rate:** Measured with Nsight, ≥95% expected at ≤4096 keys
4. **Competitive:** Redis 7.x local, FASTER (if available)

**Expected Results:**
- Break-even: ~512 entries vs. Redis
- Speedup at 4096 entries: 4–6× vs. Redis (spec claim, now with correct implementation)
- L2 hit rate: ≥95%

**Estimated hours:** 80–120h

---

## Phase 10: Documentation & Production Readiness (Week 10–11)

**Deliverable:** Doxygen API docs, user guide, architecture guide, release package.

**Documentation:**
- `docs/API.md` — Doxygen-generated (C++ + Python)
- `docs/USER_GUIDE.md` — Installation, build, examples
- `docs/ARCHITECTURE.md` — Deep dive, design decisions
- `docs/PERFORMANCE_TUNING.md` — Batch size selection, load factor impact
- `CHANGELOG.md` — v2.0 → v2.0.1 corrections

**CI/CD:**
- GitHub Actions: `cmake && make && make test` on each commit
- Code coverage: ≥90% on kernels, ≥85% overall
- Automated benchmarks (if hardware available)

**Release Package:**
- CMakeLists.txt install targets
- Python wheel (`pip install warpkv`)
- Versioning: `v2.0.1` tag, release notes

**Estimated hours:** 40–60h

---

## Critical Path & Timeline

**Dependency chain:**
1. Phase 0 (Foundation)
2. → Phase 1 (XXHash3)
3. → Phase 2 (Bucket layout)
4. → **Phase 3 (Lookup) ⚠️ CRITICAL**
5. → Phase 4 (Insert + stash)
6. → Phase 5 (EBR)
7. → Phase 6 (Graphs)
8. → Phase 7 (Python)
9. → Phase 9 (Benchmarks)

**Parallel tracks:**
- Phase 8 (Testing) can start after Phase 4
- Phase 10 (Docs) can start after Phase 7

**Total effort:** 715 hours
- **1 FTE:** 18 weeks (~4.5 months)
- **2 FTE:** 9 weeks (~2.25 months with parallelization)
- **Buffer:** +20% for unknowns = 21 weeks (1 FTE) / 11 weeks (2 FTE)

---

## Risk Mitigation

| Risk | Mitigation | Checked |
|------|-----------|---------|
| Phase 3: Lane out-of-bounds | Extensive unit tests, AddressSanitizer | Phase 3 unit tests |
| Phase 3: Launch grid error | Verify grid = BATCH_SIZE formula | Phase 3 integration tests |
| Phase 4: Stash queue corruption | Thread safety with atomics, stress test | Phase 8 stress tests |
| Phase 5: EBR race condition | Memory ordering verified, reader count stress | Phase 8 stress tests |
| Phase 6: Graph slot aliasing | Triple-buffered slot rotation verified | Phase 6 unit tests |
| Phase 7: GIL deadlock | Keys copied before release, concurrent test | Phase 8 integration tests |
| Phase 9: Benchmarks vary | Warm-up runs, N≥5 trials, error bars | Phase 9 benchmark harness |

---

## Success Criteria

✅ **Spec Compliance:**
- All 20 corrections from audit implemented correctly
- Zero out-of-bounds access
- All keys processed (no silent drops)
- Correct lane mapping (0-7 per bucket)
- Proper stash queue handoff
- Safe EBR pointer swap

✅ **Testing:**
- 80+ test cases, 100% pass rate
- Zero AddressSanitizer/ThreadSanitizer errors
- ≥90% code coverage on kernels

✅ **Performance:**
- Throughput: ≥512 keys/µs at 4096-entry batch (target)
- L2 hit rate: ≥95% at ≤4096 entries
- Latency: ~20–25µs roundtrip (matches spec Section III)
- Benchmark comparison: published with error bars

✅ **Production Ready:**
- CI/CD pipeline passing
- Documentation complete
- Release package available
- Zero known issues

---

## Checkpoints

**After Phase 2 (Week 3):**
- Bucket layout correct, arena allocator working
- Decision: proceed to Phase 3 or adjust

**After Phase 4 (Week 5):**
- Kernels complete, stash queue working
- Decision: proceed to Phase 5 or iterate

**After Phase 6 (Week 7):**
- Pipeline complete, CUDA Graphs verified
- Decision: proceed to Python or optimize

**Final (Week 11):**
- All phases complete, spec compliance 100%
- Benchmark report published
- Production deployment ready

---

**Version:** 2.0.1 (Corrected Implementation Plan)  
**Status:** Ready to implement; all bugs fixed, all gaps closed  
**Next:** Begin Phase 0 — Foundation & Validation

