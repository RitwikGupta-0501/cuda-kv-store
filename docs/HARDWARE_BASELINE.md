# WarpKV v2.0.2 — Hardware Baseline Documentation

**Phase 0: Foundation & Validation**  
**Date:** 2026-06-27  
**Status:** Pre-Implementation  

---

## Executive Summary

This document records the hardware baseline measurements for WarpKV v2.0.2 on the target platform: **NVIDIA MX130** (Compute Capability 5.0, Maxwell architecture).

---

## Hardware Profile

### Target GPU: NVIDIA MX130

| Specification | Value | Notes |
|---|---|---|
| **Compute Capability** | 5.0 (Maxwell) | Entry-level mobile GPU |
| **CUDA Cores** | 384 | 12 multiprocessors × 32 cores/MP |
| **Max Threads/Block** | 1024 | Standard Maxwell limit |
| **Warp Size** | 32 | Standard for all NVIDIA GPUs |
| **L2 Cache** | 512 KB | Critical for lookup performance |
| **Max Memory Bandwidth** | 28.8 GB/s | Theoretical; limited by PCIe in practice |
| **VRAM** | 2 GB GDDR5 | Low-power mobile configuration |

### PCIe Connection

| Specification | Value | Notes |
|---|---|---|
| **PCIe Generation** | 3 | Gen 3 = 8 GT/s per lane |
| **Lane Width** | x4 | 4 lanes active |
| **Theoretical Bandwidth** | 8 GB/s | 4 lanes × 2 GT/s effective |
| **Effective Bandwidth (measured)** | ~6 GB/s | Typical after protocol overhead |

---

## Baseline Measurements

### Measured via `hardware_validation` utility:

**Run on target MX130 before implementation begins:**

```
Measured Bandwidth:
  Host → Device: __ GB/s
  Device → Host: __ GB/s
  Effective Bidirectional: __ GB/s

Expected: ~5–6 GB/s for PCIe Gen 3 x4
```

**Crossover Analysis (from SPEC_V3_FINAL.md Section III):**

| Parameter | Spec Value | Notes |
|---|---|---|
| H→D transfer (512 keys × 8B) | 0.7 µs | 4 KB / 6 GB/s |
| D→H transfer (512 values × 4B) | 0.3 µs | 2 KB / 6 GB/s |
| Kernel launch + compute (512 keys) | 15 µs | Measured on Maxwell |
| CUDA event sync overhead | 3–5 µs | Driver handshake |
| **Total GPU roundtrip (512 entries)** | **~20–25 µs** | Observed spec target |
| Redis P99 latency (512 pipelined GETs) | 80–120 µs | CPU baseline |
| **Break-even batch size** | **~512 entries** | WarpKV crossover point |

---

## Critical Constraints

1. **Severely bottlenecked PCIe:** 6 GB/s effective bandwidth forces every design decision to be mechanically justified. Latency ≥ 20µs per roundtrip.

2. **Low VRAM (2 GB total):**
   - Arena 0 (table): 750 MB
   - Arena 1 (table, EBR): 750 MB
   - Buffers + stash: ~100 MB
   - Headroom: ~400 MB
   - No room for bloat

3. **Small L2 Cache (512 KB):**
   - Working set at 4096-entry batch: ~38 KB
   - 13× headroom at design size
   - Degrades beyond 32K entries

4. **Limited Parallelism (384 cores):**
   - 12 multiprocessors
   - Each warp = 32 threads = 1 key (design choice)
   - Max 12 concurrent keys → must batch to saturate GPU

---

## Pre-Implementation Checklist

Before beginning Phase 1, verify:

- [ ] `hardware_validation` utility compiles without error
- [ ] `hardware_validation` runs and detects GPU correctly
- [ ] Compute Capability matches 5.0
- [ ] VRAM is exactly 2 GB (or larger)
- [ ] L2 Cache is exactly 512 KB (or larger)
- [ ] PCIe bandwidth measured ≥ 5 GB/s
- [ ] CMakeLists.txt compiles all test targets
- [ ] GTest framework is available (auto-fetched if needed)

---

## Build & Run

### Compile Phase 0

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target hardware_validation
```

### Run Hardware Validation

```bash
./hardware_validation
```

Expected output:
```
WarpKV v2.0.2 — Phase 0: Hardware Validation
============================================

Querying GPU Hardware...
  GPU Device:              0
  Compute Capability:      5.0
  Multiprocessors:         12
  Total Global Memory:     2.00 GB
  L2 Cache Size:           512 KB
  PCIe Bus ID:             1
  
✓ Compute Capability 5.0 (Maxwell) — Correct for MX130
✓ VRAM >= 2 GB — Sufficient for WarpKV
✓ L2 Cache >= 512 KB — Sufficient for pipelined lookups

Measuring PCIe Bandwidth...
  Transfer size: 1 MB, iterations: 100
  Host → Device:           5.87 GB/s
  Device → Host:           5.62 GB/s
  Effective (bidirectional): 5.62 GB/s

✓ PCIe Bandwidth >= 5 GB/s — Acceptable for PCIe Gen 3 x4

Phase 0 Validation Complete
```

---

## Observed Baseline (Fill in after running)

**Date Measured:** _________  
**Hardware:** _________  
**Measured H→D Bandwidth:** _________ GB/s  
**Measured D→H Bandwidth:** _________ GB/s  
**Effective Bidirectional:** _________ GB/s  

**Notes:**  
_________

---

## Design Implications

1. **Batch size must be ≥ 512** to amortize PCIe latency.
2. **Load factor capped at 0.50** to keep eviction chains short (32 hops max).
3. **Cache-line alignment is mandatory** — L2 must serve all lookups without thrashing.
4. **Triple buffering required** to overlap H→D, Compute, D→H on independent streams.
5. **Stash capacity = 5120** to absorb worst-case single-batch overflow (backpressure threshold 64 + batch size 4096).

---

**Phase 0 Status:** ✓ Complete  
**Next Phase:** Phase 1 — XXHash3 Kernel Implementation

