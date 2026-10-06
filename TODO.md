# WarpKV Viability & Architecture Roadmap

This document outlines the necessary architectural changes to transition WarpKV from a systems prototype into a production-viable, GPU-native key-value store. 

## 🎯 Target Use Case: GPU-Native State
Do not attempt to build a generic CPU-facing KV store (like Redis). The PCIe bus latency kills viability. Instead, target workloads where **data and compute are already on the GPU**, such as:
1. **Sparse Embedding Caches (AI):** For PyTorch custom CUDA kernels that need to look up sparse user/token embeddings.
2. **GPU Dataframe Analytics:** For hash joins and aggregations in columnar engines like cuDF.

---

## 🛠️ Phase 1: Architectural Pivot (Eliminate PCIe Bottleneck)
- [ ] **Expose Device-Side APIs:** Implement `__device__` functions (e.g., `warp_lookup`, `warp_insert`) that can be called directly by *other* external CUDA kernels, completely bypassing the CPU.
- [ ] **Deprecate the Blocking Pipeline:** Remove the `cudaStreamSynchronize` calls currently hardcoded at the end of `submit_insert_batch` and `submit_lookup_batch`.
- [ ] **C ABI for Cross-Language Support:** Wrap device APIs in `extern "C" __device__` to disable C++ name mangling. This allows languages like Rust (Rust-CUDA) and Python (Numba, CuPy) to link directly to the PTX modules.
- [ ] **Zero-Dependency Header:** Distribute a clean `warpkv_device.cuh` header containing only core structs and device logic for AI researchers to safely `#include` in their custom PyTorch/Triton kernels.

## 💾 Phase 2: Data Types & Schema Redesign
- [ ] **Drop `uint32_t` Support:** Upgrade the core structures to support `uint64_t` or 128-bit keys and values. This is strictly required for cryptographic hashes, composite keys, or storing **pointers** to larger structures (like tensors) in VRAM.
- [ ] **Ban Variable-Length Strings/Blobs:** Explicitly restrict the system to fixed-width types to maintain coalesced memory access and prevent branch divergence.

## ⚡ Phase 3: Optimize Resource Waste
- [ ] **Fix Memory Padding (Bucket Struct):** The current 128-byte `Bucket` wastes 52 bytes (40%) on padding. Redesign the layout. For 64-bit keys/values, a 128-byte L2 cache line can hold exactly 7 keys, 7 values, 7 fingerprints, and an occupancy mask (total 120 bytes).
- [ ] **Fix Compute Utilization (Warp Idle Time):** The current lookup uses a 32-thread warp but leaves 16 threads completely idle when scanning the two buckets. 
    - *Option A:* Refactor the kernel to use Half-Warps (16 threads per key) to process two keys per warp.
    - *Option B:* Expand Cuckoo hashing to use 4 buckets instead of 2, utilizing all 32 threads for parallel scanning.

## 🔄 Phase 4: AI Library Integration & Memory Management
- [ ] **Agnostic Memory Management (Kill `cudaMalloc`):** ML frameworks (like PyTorch) use their own caching allocators (e.g., `c10`). Internal `cudaMalloc` calls will cause out-of-memory (OOM) crashes. Modify the engine to accept a pre-allocated raw memory buffer (e.g., a PyTorch 1D Byte Tensor) to back the buckets.
- [ ] **PyTorch C++ Extension:** Throw away the NumPy `pybind11` wrapper. Write a native PyTorch C++ extension (`#include <torch/extension.h>`) where the host API natively accepts and returns `torch::Tensor` objects.
- [ ] **Zero-Copy Batching:** Ensure the new host API operates entirely on the `data_ptr` of the provided PyTorch tensors, launching batches directly on GPU-resident data without any host-to-device PCIe copies.

## 🚀 Phase 5: CPU API Overhaul (If Host-Side Ingestion is Maintained)
- [ ] **Implement True Asynchronous Submissions:** The CPU API must return Futures/Promises or accept Callbacks, allowing the CPU to prepare the next batch while the GPU computes.
- [ ] **Dynamic Mega-Batches:** Remove the hardcoded `BATCH_SIZE` of 4096. To actually amortize PCIe transfer costs, batch sizes must scale to 100,000 - 1,000,000+ keys per submission.
