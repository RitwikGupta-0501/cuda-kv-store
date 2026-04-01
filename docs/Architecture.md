# WarpKV: GPU-Accelerated Key-Value Store Architecture

## 1. Introduction & Design Philosophy

WarpKV is a high-performance, GPU-accelerated key-value data store designed to bypass the traditional CPU and block-storage bottlenecks associated with LSM-trees and B-trees.

By leveraging the massive parallel compute capabilities and High-Bandwidth Memory (HBM/GDDR) of modern GPUs, WarpKV achieves hundreds of millions of operations per second (MOPS). To avoid the latency penalties of the PCIe bus, the system utilizes aggressive query batching, asynchronous CUDA streams, and lock-free parallel data structures.

### 1.1 Core Constraints & Data Types

To maximize GPU performance and memory coalescing, the system adheres to the following constraints:

Key/Value Types: Fixed-size 64-bit unsigned integers (uint64_t). Variable-length strings are not supported in the core index to prevent memory fragmentation and pointer-chasing.

Hash Table Design: Open Addressing with Linear Probing. Linked lists (separate chaining) are strictly prohibited to prevent thread divergence.

Concurrency: Strictly lock-free. Synchronization is achieved using hardware-level 64-bit atomic Compare-And-Swap (atomicCAS) instructions.

## 2. Phase 1: The In-VRAM Architecture

In this initial architecture, the entire hash table resides in the GPU's Video RAM (VRAM). The CPU acts as a network endpoint and batch dispatcher, while the GPU acts as the database engine.

### 2.1 Memory Layout: Struct of Arrays (SoA)

To ensure that when a warp of 32 CUDA threads accesses the hash table, they read from contiguous memory blocks (Memory Coalescing), the hash table is laid out as two separate flat arrays in VRAM, rather than an array of objects.

```
// VRAM Memory Allocation
uint64_t* d_keys; // Device pointer to Keys array
uint64_t* d_values; // Device pointer to Values array
uint64_t CAPACITY = 100000000; // Power of 2 recommended for fast modulo masking

cudaMalloc(&d_keys, CAPACITY _ sizeof(uint64_t));
cudaMalloc(&d_values, CAPACITY _ sizeof(uint64_t));
```

    Note: The d_keys array is initialized with a specific EMPTY_SENTINEL value (e.g., 0xFFFFFFFFFFFFFFFF) to denote unused slots.

### 2.2 The Host Dispatcher (CPU Batching)

The CPU does not send individual GET or PUT requests to the GPU. Doing so would cripple the system with PCIe latency. Instead:

The CPU receives incoming requests and buffers them into a pinned host memory array (cudaHostAlloc).

Once the buffer reaches a threshold (e.g., 10,000 queries) or a microsecond timer expires, the batch is dispatched.

The CPU utilizes cudaMemcpyAsync to transfer the batch over the PCIe bus, overlapping the transfer with computation via CUDA Streams.

### 2.3 The Device Execution (CUDA Kernel)

When the batch arrives in VRAM, a CUDA kernel is launched with one thread per query.

The Lock-Free Insertion (PUT) Logic:

Hash: Thread hashes its assigned key to find the starting index i.

Probe & Swap: The thread enters a while(true) loop:

It performs an atomic hardware swap: old_key = atomicCAS(&d_keys[i], EMPTY_SENTINEL, my_key).

Success: If old_key == EMPTY_SENTINEL, the slot was empty, and the thread successfully claimed it. It writes d_values[i] = my_value and exits.

Update: If old_key == my_key, the key already exists. The thread updates d_values[i] = my_value and exits.

Collision: If old_key is something else, the slot is taken. The thread increments i = (i + 1) % CAPACITY (Linear Probing) and tries again.

The Retrieval (GET) Logic:

Thread hashes the key to find index i.

It reads d_keys[i].

If d_keys[i] == my_key, it reads d_values[i], writes it to the Result Array, and exits.

If d_keys[i] == EMPTY_SENTINEL, the key does not exist. It writes a NOT_FOUND sentinel to the Result Array and exits.

Otherwise, it probes forward i++.

### 2.4 Limitations of Phase 1

The maximum size of the data store is strictly bounded by the physical VRAM of the GPU (e.g., ~100M entries on a 12GB GPU). Once VRAM is full, the database can no longer accept new keys.

## 3. Scaling Up: Transitioning to the L4 Cache Architecture

To bypass the VRAM capacity limit without sacrificing GPU acceleration, the system scales into an Out-of-Core Heterogeneous Memory Architecture.

The CPU's system RAM becomes the primary data store (massive capacity, slower access), and the GPU's VRAM is repurposed as an "L4 Cache" (limited capacity, ultra-fast access) for the hottest data.

### 3.1 What Changes and What Stays the Same?

The Code: The lock-free CUDA kernel written in Phase 1 does not change.

The Pointers: The only change is where the pointers passed into the kernel reside.

The Additions: The C++ Host Dispatcher must be updated to handle "Cache Misses" and data eviction.

### 3.2 The Heterogeneous Memory Layout

The Cold Store (Host RAM): A massive hash table (e.g., 256GB) is allocated in the CPU's system memory using cudaHostAlloc (Pinned Memory). This allows the GPU to read it directly over PCIe via Direct Memory Access (DMA).

The Hot Cache (VRAM): The Phase 1 hash table remains in VRAM but is renamed d_cache_keys and d_cache_values.

### 3.3 The New Query Data Flow

When a batch of 10,000 GET queries is dispatched to the GPU:

Step 1: The VRAM Cache Check (Device)

The CUDA kernel executes against the d_cache residing in VRAM.

If a thread finds its key, it writes the value to the Result Array (Cache Hit).

If a thread hits an EMPTY_SENTINEL during probing, it writes a special CACHE_MISS flag to the Result Array.

The GPU sends the Result Array back to the CPU over the PCIe bus.

Step 2: The Host Fallback (CPU)

The CPU thread pool quickly scans the returned Result Array.

For every index marked CACHE_MISS, the CPU uses standard C++ to query the massive Cold Store hash table located in its own local Host RAM.

The CPU combines the VRAM results and Host RAM results and returns the final batch to the client.

### 3.4 Data Promotion and Eviction (The Cache Policy)

Because VRAM is limited, it must constantly cycle data to remain effective.

Promotion (Cold to Hot): When a CPU thread fetches a key from the Cold Store, it increments a frequency counter. If the counter crosses a threshold, the CPU asynchronously sends a PUT request to the GPU to insert that key into the VRAM cache.

Eviction (Hot to Cold): A background CUDA kernel runs periodically to implement a Clock Sweep or LRU (Least Recently Used) approximation. It zeroes out "cold" keys in the VRAM cache (reverting them to EMPTY_SENTINEL), freeing up slots for new hot data.

## 4. Summary of System Scalability

By structuring WarpKV this way, the system achieves maximum theoretical throughput for datasets that fit in VRAM (Phase 1). As the dataset grows into the terabytes (Phase 2), the system gracefully degrades. Instead of failing out of memory, it automatically utilizes the PCIe bus to fetch colder records, while maintaining microsecond GPU latencies for the most frequently accessed data.
