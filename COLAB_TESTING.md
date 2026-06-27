# WarpKV v2.0.2 — Google Colab Testing Guide

## Quick Start (Copy-Paste to Colab Cell)

```bash
# Clone repository
! git clone https://github.com/YOUR_REPO/gpu-kvstore.git
% cd gpu-kvstore

# Run all tests
! bash scripts/run_tests_colab.sh

# Run specific phase
! bash scripts/run_tests_colab.sh --phase 4
```

## Full Colab Notebook Setup

```python
# Cell 1: Install dependencies and setup
!apt-get update -qq
!apt-get install -y -qq cmake > /dev/null 2>&1
!nvcc --version

# Cell 2: Clone and build
!git clone https://github.com/YOUR_REPO/gpu-kvstore.git
%cd gpu-kvstore
!bash scripts/run_tests_colab.sh --build-only

# Cell 3: Run tests
!bash scripts/run_tests_colab.sh
```

## Test Phases

Run specific phases without building everything:

```bash
# Phase 1: Hash function (XXHash3)
! bash scripts/run_tests_colab.sh --phase 1

# Phase 2: Bucket layout & arena allocator
! bash scripts/run_tests_colab.sh --phase 2

# Phase 3: Warp-cooperative lookup
! bash scripts/run_tests_colab.sh --phase 3

# Phase 4: Cuckoo insertion with eviction chains
! bash scripts/run_tests_colab.sh --phase 4

# Phase 5: Rehashing with stash drain
! bash scripts/run_tests_colab.sh --phase 5

# All phases
! bash scripts/run_tests_colab.sh --phase all
```

## Troubleshooting on Colab

### CUDA Out of Memory
If you see CUDA allocation errors:
```python
import torch
torch.cuda.empty_cache()
```

Then re-run tests. Some tests allocate large buffers (750MB per table).

### Build Fails
Check CUDA availability:
```bash
! nvidia-smi
! nvcc --version
```

If `nvcc` not found, CUDA toolkit may not be available on that Colab session. Try a new session.

### Tests Timeout
If tests hang (unlikely for unit tests), increase timeout in Colab cell settings.
Most tests should complete in <10 seconds.

## Understanding Test Output

```
============================================================================
WarpKV v2.0.2 — GPU Test Harness
============================================================================
Repository: /content/gpu-kvstore
Build Directory: /content/gpu-kvstore/build
Phase: all

[1/4] Checking dependencies...
  ✓ cmake 3.22.4
  ✓ CUDA 11.8

[2/4] Building WarpKV...
  ✓ Build successful

[3/4] Running tests...

  ✓ PASS  test_xxhash3 (7 tests)
  ✓ PASS  test_bucket_layout (9 tests)
  ✓ PASS  test_arena_allocator (6 tests)
  ✓ PASS  test_warp_lookup (9 tests)
  ✓ PASS  test_cuckoo_insert (10 tests)
  ✓ PASS  test_eviction_chains (10 tests)
  ✓ PASS  test_rehash_kernel (10 tests)

[4/4] Test Summary
============================================================================
Total Tests Run: 7
  ✓ Passed: 7
  ✗ Failed: 0

All tests passed! ✓
```

## Continuous Integration (Optional)

To automate testing on each commit:

### GitHub Actions (`.github/workflows/colab-test.yml`)

```yaml
name: Colab GPU Tests
on: [push, pull_request]

jobs:
  test-gpu:
    runs-on: [ubuntu-latest, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Run tests on GPU
        run: bash scripts/run_tests_colab.sh --phase all
```

## Performance Notes

Typical test execution times on Colab T4 GPU:
- Phase 1-3: <1 second each
- Phase 4-5: 1-2 seconds each (includes large arena allocation)
- Full suite: ~10 seconds

If times are significantly longer, GPU may be throttled or memory fragmented.
Restart Colab kernel to clear state.

## Next Steps

After tests pass:
1. Proceed to Phase 6 (Batching pipeline with CUDA Graphs)
2. Create synthetic load test for concurrent insert/lookup
3. Benchmark eviction chain overhead
4. Profile memory usage under different load factors

See `DEVELOPMENT.md` for architecture details.
