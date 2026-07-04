# WarpKV Architecture: Stream Pipelining & Ring Buffers

## 1. Overview
High-performance GPU systems are bottlenecked not just by memory latency, but by the CPU-GPU control path. If the CPU waits for the GPU to finish a batch of inserts before preparing the next batch, both processors sit idle during the other's execution phase. WarpKV utilizes **CUDA Streams** and a **Ring Buffer** architecture to pipeline these operations, keeping both the CPU and GPU fully saturated.

## 2. CUDA Streams and Concurrency
A CUDA Stream is a sequence of operations that execute in issue-order on the GPU. Operations in *different* streams can execute concurrently. 
WarpKV provisions a fixed number of stream "slots" (defined by `NUM_SLOTS`, typically 3). Each slot acts as an independent execution pipeline containing:
- Its own `cudaStream_t` instances (for Memcpy H2D, Compute, and Memcpy D2H).
- Its own pre-allocated pinned host memory buffers (`h_keys_in`, `h_values_in`, `h_values_out`).
- Its own pre-allocated device memory buffers (`d_keys_in`, `d_values_in`, `d_values_out`).
- Pre-captured CUDA Graphs containing the exact kernel launch parameters.

## 3. The Ring Buffer Submission Model
When the CPU submits a new batch of operations (e.g., `submit_insert_batch`), it does not just pick a random stream. It operates on the slots as a circular ring buffer:
1. An atomic counter `current_slot` is incremented and modulo'd by `NUM_SLOTS` to deterministically select the next slot in the ring.
2. A mutex (`slot_mutex[slot]`) ensures thread-safety if multiple CPU threads are concurrently submitting batches to the engine.
3. **Pre-Synchronization**: Before copying the new batch into the slot's host memory buffer, the CPU calls `cudaStreamSynchronize` on that specific slot. 
   - *Why?* If the CPU is submitting batches faster than the GPU can process them, it will eventually lap the ring buffer and reach a slot that is still processing an older batch. The synchronization ensures the CPU naturally throttles and waits for the slot to become available without overwriting in-flight data.

## 4. CUDA Graphs (Low Overhead Launch)
Traditional kernel launches (`<<<grid, block>>>`) have a non-trivial CPU overhead (~5-10 microseconds per launch). In a high-throughput pipeline, this CPU overhead can become the bottleneck.
WarpKV records the memory copies and kernel executions for both Inserts and Lookups into **CUDA Graphs** during the engine's `build_graphs()` initialization. 
When submitting a batch, the engine simply calls `cudaGraphLaunch`. This reduces the CPU submission overhead to <2 microseconds, allowing the CPU to instantly enqueue work and move on to preparing the next batch in the ring buffer.

## 5. Synchronous vs Asynchronous API Contracts
While the underlying stream architecture is designed to support fully asynchronous overlapping (e.g., using `cudaLaunchHostFunc` C-style callbacks to notify the CPU when a batch is done), WarpKV's current public API enforces synchronous semantics for individual batches. 

- After a batch is launched via a CUDA Graph, the engine explicitly calls `cudaStreamSynchronize` before returning to the caller.
- This guarantees **Data Integrity**. It ensures that if a caller submits a lookup immediately after an insert, the lookup is guaranteed to see the inserted data. True asynchronous pipelining requires the caller to manage data dependencies and poll for completion, which violates the immediate-return contract expected by most KV-store users.
- The ring-buffer architecture remains in place to support future fully-async API variants (e.g., `submit_lookup_async(keys, callback)`).
