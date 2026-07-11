<div align="center">
  <h1>🚀 WarpKV</h1>
  <p><strong>An ultra high-performance, GPU-accelerated Key-Value store written in CUDA and C++17.</strong></p>
</div>

WarpKV is designed for extreme throughput workloads that require millions of operations per second. By offloading hash table lookups and insertions to the GPU using **Warp-Cooperative Cuckoo Hashing** and **CUDA Graphs**, it achieves sustained speeds of up to **96 Million lookups per second**.

## ✨ Key Architectural Features

- **Warp-Cooperative Execution:** Uses all 32 threads in a CUDA warp collaboratively to scan hash buckets in parallel, eliminating warp divergence and achieving 100% coalesced memory access.
- **Double-Buffered Epoch-Based Reclamation (EBR):** A fully lock-free, zero-downtime background rehashing system. When the GPU load factor crosses 50%, a background thread seamlessly allocates a new table, migrates data, and hot-swaps the pointers without blocking in-flight CPU operations.
- **Lock-Free Stash Queue:** A highly optimized emergency overflow queue for Cuckoo Hash collisions, preventing data loss and triggering automatic backpressure.
- **Asynchronous PCIe Pipelining:** Uses multiple pinned memory streams (`cudaMemcpyAsync`) and CUDA Graphs (`cudaGraphLaunch`) to saturate the PCIe bus and overlap CPU batching with GPU execution.
- **GPU-Optimized XXHash3:** Includes a custom, heavily vectorized implementation of `xxhash3` that runs natively on the GPU for sub-nanosecond fingerprinting.
- **Arena Memory Allocator:** Pre-allocates all necessary VRAM upfront, completely eliminating `cudaMalloc` overheads during execution.

## 📊 Benchmarks

WarpKV includes industry-standard benchmarking tools out of the box, including the **YCSB (Yahoo! Cloud Serving Benchmark) Workload C**, which simulates a cache-heavy, 100% read workload using a mathematically accurate Scrambled Zipfian distribution.

**YCSB Workload C (4 Million Buckets, 10 Million Keys)**
*Hardware: NVIDIA RTX 4090 / Maxwell architecture baseline*
- **Insert Throughput:** `75.06 Million keys/sec`
- **Lookup Throughput:** `96.85 Million keys/sec` (Cache-hit optimized)
- **Mismatches:** `0`

**Load Factor Degradation Curve**
A built-in stress test that disables automatic rehashing to demonstrate Cuckoo Hash breakdown points, proving the 50% rehash threshold.

| Load Factor % | Insert (M keys/s) | Lookup (M keys/s) | Missing Keys |
|---------------|-------------------|-------------------|--------------|
| 10%           | 70.52             | 92.14             | 0            |
| 30%           | 75.11             | 94.25             | 0            |
| 45%           | 24.95             | 87.29             | 0            |
| 50%           | 15.19             | 18.62             | 638 (Stash Overflow) |
| 90%           | 2.08              | 15.58             | ~5.8 Million |

*Note: WarpKV dynamically prevents data loss by automatically rehashing at exactly 50% capacity.*

## 🛠️ Build & Installation

### Requirements
- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Compiler:** GCC/G++ with C++17 support
- **CUDA:** Toolkit 11.0+ (Tested on 12.8)
- **CMake:** 3.18+

### Compilation

```bash
git clone https://github.com/RitwikGupta-0501/cuda-kv-store.git
cd cuda-kv-store
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## 🚀 Running Benchmarks

After building the project, you can run the benchmarks directly from the `build` directory:

```bash
# Run the YCSB Workload C Test
./ycsb_benchmark

# Run the Load Factor Breakdown Test
./load_factor_benchmark
```

## 📖 Technical Documentation

For a deep dive into the internal architecture, engine design, and mathematical proofs behind the implementation, please see our comprehensive design docs:

- [`01_engine_architecture.md`](docs/01_engine_architecture.md) - PCIe bottleneck fixes and async pipelining
- [`02_epoch_based_reclamation.md`](docs/02_epoch_based_reclamation.md) - Background lock-free rehashing
- [`03_cuckoo_hashing.md`](docs/03_cuckoo_hashing.md) - Mathematical bounds and eviction chains
- [`04_stream_buffer.md`](docs/04_stream_buffer.md) - Pinned host memory batching
- [`05_memory_coalescing.md`](docs/05_memory_coalescing.md) - 128-byte warp alignment optimizations
- [`06_arena_allocator.md`](docs/06_arena_allocator.md) - VRAM double-buffering
- [`07_xxhash3_gpu.md`](docs/07_xxhash3_gpu.md) - Vectorized GPU hashing
- [`08_python_bindings.md`](docs/08_python_bindings.md) - pybind11 integration

## 🐍 Python Bindings

WarpKV comes with native Python bindings via `pybind11` for use in Machine Learning or Data Science workflows. *(Requires `import warpkv` from the compiled `.so` module).*

```python
import warpkv
import numpy as np

# Initialize engine with 4 Million buckets
engine = warpkv.Engine(4194304)

# Create 1 Million sequential keys
keys = np.arange(1, 1000001, dtype=np.uint32)
values = keys * 2

# Insert batch
engine.insert_batch(keys, values)

# Lookup batch
results = engine.lookup_batch(keys)

print(f"Lookups completed successfully!")
```

## License
MIT License.
