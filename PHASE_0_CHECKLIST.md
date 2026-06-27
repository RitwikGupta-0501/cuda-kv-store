# Phase 0: Foundation & Validation — Completion Checklist

**Status:** IN PROGRESS  
**Started:** 2026-06-27  
**Target Duration:** 1 week (40–60 hours)

---

## Deliverables

### ✓ 1. CMakeLists.txt — Production Build Configuration
- [x] CUDA architecture set to CC 5.0 (Maxwell)
- [x] C++ standard set to C++17
- [x] Google Test framework integrated
- [x] Test targets defined (phases 1–3)
- [x] Benchmark targets defined
- [x] Build documentation added

**Files:**
- `CMakeLists.txt` — Updated for Phase 0+

### ✓ 2. Hardware Validation Utility
- [x] `hardware_validation.cpp` implemented
- [x] Checks Compute Capability (should be 5.0)
- [x] Measures total VRAM (should be 2 GB)
- [x] Checks L2 cache size (should be 512 KB)
- [x] Measures PCIe bandwidth (expected ~6 GB/s)
- [x] Validates against spec requirements

**Files:**
- `src/core/hardware_validation.cpp` — Created
- `docs/HARDWARE_BASELINE.md` — Created (baseline measurements to fill in)

### ✓ 3. Test Infrastructure
- [x] GTest framework configured
- [x] `tests/unit/` directory created
- [x] `tests/integration/` directory created
- [x] Placeholder test files for phases 1–3
  - `tests/unit/test_xxhash3.cpp`
  - `tests/unit/test_bucket_layout.cpp`
  - `tests/unit/test_arena_allocator.cpp`
  - `tests/unit/test_warp_lookup.cpp`
  - `tests/integration/test_lookup_correctness.cpp`

**Files:**
- All test skeleton files created with `TODO` placeholders

### ✓ 4. Build System Organization
- [x] Source code organized: `src/core/`, `src/gpu/`, `include/`
- [x] Tests organized: `tests/unit/`, `tests/integration/`
- [x] Benchmarks organized: `benchmarks/`
- [x] Documentation: `docs/` (including HARDWARE_BASELINE.md)

---

## Pre-Compilation Verification

Before compiling, verify:

- [x] Directory structure correct
- [x] All source files created
- [x] All test skeleton files in place
- [x] CMakeLists.txt references correct files

---

## Compilation & Verification

### Step 1: Configure Build

```bash
cd /mnt/data/Repository/gpu-kvstore
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Expected output:
```
-- WarpKV v2.0.2 Build Configuration
-- CUDA Architecture: CC 5.0 (Maxwell)
-- C++ Standard: C++17
-- Build Type: Release
-- Unit Tests: ENABLED
```

### Step 2: Build Hardware Validation Target

```bash
cmake --build . --target hardware_validation
```

Expected: Compiles without error

### Step 3: Run Hardware Validation

```bash
./hardware_validation
```

Expected: Detects GPU, measures bandwidth, validates specs

### Step 4: Build Test Targets (Phase 1+)

```bash
cmake --build . --target test_xxhash3
```

Expected: Compiles test skeleton (tests will be TODO placeholders)

### Step 5: Verify All Tests Enumerate

```bash
ctest --verbose
```

Expected: Lists all test cases (some will fail due to TODO)

---

## Success Criteria

Phase 0 is **complete** when:

- [ ] `hardware_validation` compiles and runs successfully
- [ ] GPU is correctly detected with CC 5.0
- [ ] PCIe bandwidth is measured ≥ 5 GB/s
- [ ] CMakeLists.txt compiles all test targets without error
- [ ] GTest framework is available (auto-fetched if needed)
- [ ] Test skeltons can be listed via `ctest`
- [ ] Hardware baseline measurements recorded in `HARDWARE_BASELINE.md`

---

## Hardware Baseline (Fill in after running)

**Date:** _________  
**GPU Detected:** _________  
**Compute Capability:** _________  
**VRAM:** _________ GB  
**L2 Cache:** _________ KB  
**Measured PCIe Bandwidth:** _________ GB/s  

✓ / ✗ Compute Capability is 5.0  
✓ / ✗ VRAM is 2 GB  
✓ / ✗ L2 Cache is 512 KB  
✓ / ✗ PCIe Bandwidth ≥ 5 GB/s  

---

## Next Phase: Phase 1 (XXHash3 Kernel)

Once Phase 0 is verified:

1. Implement XXHash3 hash function (Section V of SPEC_V3_FINAL.md)
2. Create unit tests for hash properties (avalanche, correlation, etc.)
3. Create benchmark for hash throughput
4. Validate against spec requirements

---

**Phase 0 Status:** FOUNDATION SET UP  
**Next Milestone:** Phase 1 — XXHash3 Implementation

