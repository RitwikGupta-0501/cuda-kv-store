# WarpKV: The "SQLite for ML" & Dynamic Embedding Engine

## The Dual-Layer Product Vision
WarpKV is fundamentally a high-performance **GPU Caching Engine** (the "SQLite of ML"). However, to capture the massive ML training market, this engine powers a **Dynamic Embedding Layer** wrapper. 

We provide two distinct APIs, serving both inference and training:

1. **`warpkv.GPUCache` (The Engine):** A raw, no-grad key-vector store for fast nearest-neighbor retrieval, LLM KV-caching, and RAG architectures.
2. **`warpkv.nn.DynamicEmbedding` (The Killer App):** A PyTorch `nn.Module` wrapper with full autograd support, solving the unpredictable sparse ID problem for Recommender Systems (RecSys) and Graph Neural Networks (GNNs).

---

## Phase 1: The Core Engine (Memory & Data Structures)
We must decouple the index from the payload to preserve GPU memory bandwidth and completely rip out legacy research constraints.

### 1. The Index (64-bit PyTorch Reality)
- **Action:** Refactor the index to store `(Key=int64_t, Index=uint32_t)`. PyTorch strictly uses 64-bit integers (`torch.LongTensor`) for categorical IDs. A 32-bit key caps the system at 4 billion IDs, which breaks real-world pipelines. 
- **Benefit:** Moving 128-dim vectors during hash collisions is avoided. Collisions are resolved via lightning-fast 12-byte atomic swaps.

### 2. The Slab Pool (Vector Storage)
- **Action:** Embeddings live in a massive pre-allocated contiguous VRAM array (`[MAX_CAPACITY, EMBEDDING_DIM]`).
- **Action:** Use a lock-free bump allocator (`atomicAdd` on a global counter) to safely claim new rows in the pool.

### 3. Rip Out Static Limits
- **Action:** Explicitly delete the hardcoded `BATCH_SIZE = 4096` macro and static CUDA Graphs. The engine must accept arbitrary-length 1D PyTorch tensors, launching dynamically sized CUDA grids (`(count + block_size - 1) / block_size`) at runtime.

---

## Phase 2: PyTorch Deep Integration (The Autograd Wrapper)
The training wrapper must handle the mathematical and hardware realities of ML workflows.

### 1. The Forward Pass (Insert-or-Get)
- **Action:** The forward pass cannot just be a "Gather". It must be a massively parallel **Insert-or-Get**. If an ID is unseen, the kernel must concurrently claim a Slab Pool row, execute the hash insert, initialize the vector (e.g., Xavier/Zero init), and return the pointer without locking the GPU.

### 2. The Backward Pass (Zipfian Gradients & Optimizers)
- **Action (Hardware Serialization):** Real-world IDs follow a Zipfian/Power-law distribution. Thousands of threads hitting `atomicAdd` on the exact same popular ID will bottleneck the SM. Mitigate this by performing a **Segmented Sort and Reduce (via CUB)** on incoming gradients locally *before* doing global memory updates.
- **Action (Optimizer State):** Vanilla SGD is rarely used in production ML. Allocate parallel Slab Pools for Adam/Adagrad moments (`M` and `V`). The backward kernel must fuse the gradient application and optimizer step directly inside the Slab Pool update.

---

## Phase 3: Load Management & Resizing

### 1. Realistic Hash Thresholds
- **Action:** Pure 2-way Cuckoo hashing mathematically degrades at ~50% capacity. To achieve high VRAM utilization (e.g., 75%+), the core algorithm must be augmented (e.g., implementing an aggressively sized stash overflow queue) or shifted toward Open Addressing with Linear Probing / Robin Hood hashing, which behaves better on GPUs at high loads.

### 2. "Stop-the-World" Resizing
- **Action:** Drop the complex, lock-free Epoch-Based Reclamation (EBR). Training loops are fundamentally synchronous. When the table hits its safety capacity threshold, temporarily pause the training step, allocate a 2x larger Index and Slab Pool, launch a high-speed migration kernel, free the old memory, and resume PyTorch execution.
