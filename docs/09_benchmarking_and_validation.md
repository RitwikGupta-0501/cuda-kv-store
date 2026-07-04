# WarpKV Benchmarking & Validation Methodology

## 1. Overview
WarpKV utilizes a comprehensive suite of unit tests and synthetic benchmarks to validate both the correctness and the raw throughput of the GPU database. The benchmarking suite is designed to push the engine to its limits, simulating heavy concurrent loads and adversarial data distributions.

## 2. Unit Testing (`tests/unit/`)
The project uses **Google Test (gtest)** for correctness validation. Because GPU code is notoriously difficult to debug, the tests isolate specific kernels and engine components:

- **`test_bucket_layout.cpp`**: Validates the 128-byte alignment, ensuring that padding and memory footprints match the theoretical coalesced read layout.
- **`test_xxhash3.cpp`**: Validates the GPU port of `xxhash3` against known hash outputs to guarantee deterministic behavior across CPU and GPU.
- **`test_cuckoo_insert.cu` & `test_eviction_chains.cu`**: These test the core logic of the `warp_insert_kernel`. They construct artificial hash tables on the CPU, copy them to the device, and verify that insertions correctly resolve collisions by cascading evictions. They specifically test the failure condition where an eviction chain exceeds `MAX_EVICTION_HOPS` and is correctly relegated to the `StashQueue`.
- **`test_rehash_kernel.cu`**: Validates the `execute_rehash` kernel, ensuring that data successfully migrates from an old table (and its stash) into a new table without data loss.

## 3. The End-to-End Engine Benchmark (`engine_benchmark.cu`)
This is the primary performance validator. It spins up the full `WarpKVEngine` (complete with multiple stream slots, ring buffers, and the background rehash thread) and executes three phases:

1. **Insertion Phase**: Generates 500,000 unique keys and submits them in batches of `BATCH_SIZE` (typically 4096). This phase intentionally triggers the background rehashing thread to ensure lock-free insertions succeed while a rehash occurs in the background. It calculates the raw insertion throughput (Keys / Second).
2. **Positive Lookup Phase**: Looks up all 500,000 inserted keys. It calculates the lookup throughput and verifies that exactly 100% of the keys are found, and that the returned values perfectly match.
3. **Negative Lookup Phase**: Generates 100,000 *new* keys that are guaranteed not to be in the dataset. It submits these for lookup and expects 0% to be found. This phase specifically stress-tests the `StashQueue` scanning speed, ensuring that negative lookups don't suffer from PCIe bottlenecks.

## 4. Synthetic Load Testing (`gpu_synthetic_load_test.cu`)
While `engine_benchmark.cu` measures the higher-level engine pipeline, the synthetic load test drops down to directly hammering the GPU kernels to measure theoretical peak hardware limits.

- It bypasses the engine's ring buffer and pinned memory pipelines.
- It pre-loads millions of keys directly into massive VRAM arrays.
- It launches the `warp_insert_batch` and `warp_lookup_batch` API functions in massive grids, saturating every SM (Streaming Multiprocessor) on the GPU simultaneously.
- This provides the "Speed of Light" metric—the absolute maximum throughput the GPU can achieve if the CPU submission overhead was exactly zero. The delta between the Synthetic Load Test and the Engine Benchmark helps identify CPU pipelining overheads.
