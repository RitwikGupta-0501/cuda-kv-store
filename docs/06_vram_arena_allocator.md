# WarpKV Architecture: VRAM Arena Allocator

## 1. Overview
Dynamic memory allocation (`cudaMalloc` / `cudaFree`) on the GPU is extremely expensive. It requires CPU-GPU synchronization, driver-level overhead, and can lead to severe memory fragmentation. High-performance GPU databases must avoid runtime allocations entirely. 
WarpKV solves this using a static **Arena Allocator** design.

## 2. Pre-Allocation Strategy
At startup, before the engine accepts any user queries, the `ArenaAllocator` singleton initializes.
1. It queries the total available VRAM using `cudaMemGetInfo`.
2. It calculates the maximum possible table size that fits into memory while leaving a ~10% safety buffer for OS/driver overhead.
3. Because WarpKV uses Epoch Based Reclamation (EBR) with double-buffering, it divides the available VRAM in half.
4. It issues exactly **two** massive `cudaMalloc` calls, allocating `arena0` and `arena1`. For example, on a 2GB GPU, it might allocate two 750 MB chunks.

## 3. Zero-Overhead Resizing
Traditional hash tables grow by allocating a larger array, migrating data, and freeing the old array. 
In WarpKV:
- We never re-allocate. The VRAM is fully allocated from the beginning.
- When the active table reaches its load factor (e.g., 50%), we simply switch the active pointer to the *other* pre-allocated arena.
- We clear the old arena using a blazing fast `cudaMemsetAsync` (which operates at hundreds of GB/s), making it ready for the *next* resize cycle.
This means "resizing" the table has zero memory allocation overhead.

## 4. Power-of-2 Sizing & Bitwise Modulo
When the `ArenaAllocator` maps the raw bytes of the arena to the `BucketTable` structure, it rounds down the number of possible buckets to the nearest power of 2.

```cpp
size_t raw_buckets = arena_size_per_table_ / sizeof(Bucket);
size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));
```

This enforces a power-of-2 table size. Why is this critical?
Because modulo operations (`hash % num_buckets`) are mathematically expensive on a GPU. By ensuring `num_buckets` is a power of 2, the engine can replace the modulo operation with a simple bitwise AND mask:

```cpp
// Extremely fast on GPU:
uint32_t bucket_idx = hash & table.bucket_mask; 
```

## 5. Singleton Lifecycle
The `ArenaAllocator` is designed as a C++ Singleton because the VRAM it holds effectively represents the entire capacity of the GPU. Tying it to the singleton ensures that the multi-gigabyte allocations are safely freed (`cudaFree`) exactly once when the application terminates, preventing driver-level memory leaks.
