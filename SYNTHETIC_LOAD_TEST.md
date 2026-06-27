# Synthetic Load Test — Validation Before Phase 6

## Overview

Before implementing Phase 6 (CUDA Graphs batching pipeline), we validate that the cuckoo hashing engine works correctly under realistic load conditions.

Two test variants:
1. **CPU Simulation** (`synthetic_load_test`): Fast analytical validation
2. **GPU Execution** (`gpu_synthetic_load_test`): Real kernel testing

## Running the Tests Locally

```bash
cd build
cmake --build . --target synthetic_load_test
./synthetic_load_test

cmake --build . --target gpu_synthetic_load_test
./gpu_synthetic_load_test
```

## Running on Colab

```bash
# Clone and build
! git clone https://github.com/YOUR_REPO/gpu-kvstore.git
% cd gpu-kvstore
! bash scripts/run_tests_colab.sh --build-only

# Run synthetic load tests
! ./build/synthetic_load_test
! ./build/gpu_synthetic_load_test
```

## Test Scenarios

### 1. CPU Simulation (`synthetic_load_test`)

**Purpose**: Fast analytical validation of eviction chain behavior

**Workload**:
- Insert 100,000 keys in 4096-key batches
- Track insertion success/stash rates
- Simulate rehashing when stash reaches 50% capacity
- Lookup 100K inserted keys (expect ~99% hit rate)
- Lookup 50K non-existent keys (expect 0% false positives)

**Expected Results**:
```
Insertions:
  Total:        100,000
  Successful:    95,000 (95%)  -- Keys inserted into buckets via eviction chains
  Stashed:        5,000 (5%)   -- Keys that couldn't fit after 32 hops
  Failed:             0 (0%)   -- NO FAILURES

Lookups:
  Total:        150,000
  Successful:   149,500 (99.7%) -- All inserted keys found
  Failed:             500 (0.3%) -- Negligible false misses

Memory & Rehashing:
  Rehash count:        1         -- Triggered when stash >= 2560
  Avg eviction hops:   1.8       -- Low hop count = efficient evictions
  Stash utilization:   0.5%      -- Stash nearly empty after drain
  Final load factor:   30%       -- After 2x expansion
```

**Time**: <1 second

### 2. GPU Execution (`gpu_synthetic_load_test`)

**Purpose**: Real kernel validation with actual atomicCAS and eviction chains

**Workload**:
- Allocates 12.8MB per table (fits in Colab free tier)
- Inserts 100K unique keys
- Executes real insertion kernels
- Measures insert/stash distribution
- Validates NO DATA LOSS

**Expected Results**:
```
Insertion Results:
  Successful:     ~95,000 (95%)
  Stashed:         ~5,000 (5%)
  Load factor:       ~33%

Rehashing:
  Triggered:        Yes (load > 50%)
  Count:              1

Data Integrity:
  ✓ NO DATA LOSS DETECTED
  ✓ All insertions successful or stashed
  ✓ Eviction chains working correctly
```

**Time**: 2-5 seconds

## What These Tests Validate

✓ **Cuckoo Eviction Chains**
- Victims are selected pseudo-randomly (not infinite loops)
- Keys successfully evicted to alternative buckets
- Loop terminates after MAX_EVICTION_HOPS (32)

✓ **Stash Behavior**
- Keys that can't be evicted go to stash
- Stash doesn't overflow (capacity = 5120)
- Stash is properly drained during rehashing

✓ **Rehashing**
- Triggered at correct threshold (50% stash fullness)
- Old table entries successfully moved to new table
- Stash properly drained into new table
- No data loss during rehashing

✓ **Data Integrity**
- All inserted keys are retrievable
- No silent data loss
- Load factor stays within bounds

✓ **Atomic Safety**
- atomicCAS on key field prevents conflicts
- Multiple warps can evict safely
- No race conditions detected

## Interpreting Results

### Success Indicators

✓ All tests complete without CUDA errors
✓ Zero failed insertions
✓ Hit rate > 98% for inserted keys
✓ False positive rate ~0% for missing keys
✓ Load factor < 50% after rehashing
✓ NO DATA LOSS message printed

### Warning Signs

⚠ High stash utilization (>50%) — eviction chains struggling
⚠ Failed insertions — eviction logic issue
⚠ Low hit rate (<95%) — data not being stored correctly
⚠ CUDA out of memory — table too large for device

## Before Proceeding to Phase 6

Confirm:
1. ✅ Eviction chain unit tests pass (test_eviction_chains)
2. ✅ CPU simulation completes without errors
3. ✅ GPU synthetic load test shows zero data loss
4. ✅ Load factors stay reasonable (<50% after rehash)
5. ✅ Stash remains mostly empty (<10% utilization)

If any condition fails, investigate eviction chain implementation before moving to Phase 6.

## Performance Notes

**Expected Timings** (on Colab T4 GPU):
- CPU simulation: <1 second
- GPU test: 2-5 seconds

**Memory Usage**:
- CPU simulation: ~10MB (simulated only)
- GPU test: ~30MB total (tables + buffers)

**Scalability**:
- Can increase NUM_KEYS in source to 1M+ for stress testing
- Memory grows linearly with key count
- Rehash frequency increases slightly with larger datasets

## Next Steps

After synthetic load tests pass:
1. Proceed to Phase 6: Batching pipeline with CUDA Graphs
2. Implement triple-buffered host/device pipeline
3. Add CUDA Graph capture for deterministic scheduling
4. Implement host-side backpressure mechanism

Phase 6 will orchestrate insertion/lookup/rehash operations using the validated engine from Phases 1-5.
