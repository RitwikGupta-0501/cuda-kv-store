#pragma once

#include "../gpu/bucket_cuckoo.h"
#include "../include/warpkv/warpkv_allocator.h"
#include "../gpu/cuckoo_insert.h"
#include "../gpu/warp_lookup.h"
#include <cuda_runtime.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <future>
#include <vector>

namespace warpkv {

struct EpochTable {
    BucketTable* arenas[2] = {nullptr, nullptr};
    std::atomic<uint64_t> epoch{0};
    std::atomic<int32_t> readers[2];

    EpochTable() {
        readers[0] = 0;
        readers[1] = 0;
    }
};

struct PipelineStreams {
    cudaStream_t h2d;
    cudaStream_t compute;
    cudaStream_t d2h;
};

// ============================================================================
// LookupFutureResult — value type returned by async lookup futures.
// ============================================================================
struct LookupFutureResult {
    std::vector<ValueT> values; ///< Output values, parallel to input keys.
};

class WarpKVEngine {
private:
    static constexpr uint32_t NUM_SLOTS = 3;

    // Pinned host memory
    KeyT*         h_keys_in[NUM_SLOTS]          = {nullptr};
    ValueT*       h_values_in[NUM_SLOTS]        = {nullptr};
    ValueT*       h_values_out[NUM_SLOTS]       = {nullptr};
    InsertStatus* h_insert_statuses[NUM_SLOTS]  = {nullptr};
    uint32_t*     h_lookup_found[NUM_SLOTS]     = {nullptr};

    // Device memory
    KeyT*         d_keys_in[NUM_SLOTS]          = {nullptr};
    ValueT*       d_values_in[NUM_SLOTS]        = {nullptr};
    ValueT*       d_values_out[NUM_SLOTS]       = {nullptr};
    InsertStatus* d_insert_statuses[NUM_SLOTS]  = {nullptr};
    uint32_t*     d_lookup_found[NUM_SLOTS]     = {nullptr};

    // Streams & events
    PipelineStreams streams[NUM_SLOTS]           = {};
    cudaEvent_t    ev_h2d[NUM_SLOTS]            = {nullptr};
    cudaEvent_t    ev_compute[NUM_SLOTS]        = {nullptr};
    // ev_d2h marks completion of D→H copy within the captured graph.
    cudaEvent_t    ev_d2h[NUM_SLOTS]            = {nullptr};

    // CUDA Graphs
    cudaGraphExec_t  lookup_graphs[NUM_SLOTS]           = {nullptr};
    cudaGraphExec_t  insert_graphs[NUM_SLOTS]           = {nullptr};
    cudaGraphExec_t  delete_graphs[NUM_SLOTS]           = {nullptr};
    cudaGraphNode_t  lookup_nodes[NUM_SLOTS]            = {nullptr};
    cudaGraphNode_t  insert_nodes[NUM_SLOTS]            = {nullptr};
    cudaGraphNode_t  delete_nodes[NUM_SLOTS]            = {nullptr};
    cudaGraph_t      template_insert_graphs[NUM_SLOTS]  = {nullptr};
    cudaGraph_t      template_lookup_graphs[NUM_SLOTS]  = {nullptr};
    cudaGraph_t      template_delete_graphs[NUM_SLOTS]  = {nullptr};
    uint64_t         active_epoch[NUM_SLOTS]            = {0, 0, 0};

    // Concurrency control
    std::atomic<uint32_t> current_slot{0};
    std::mutex            slot_mutex[NUM_SLOTS];

    // Table and EBR
    EpochTable            epoch_table;
    bool                  auto_rehash_disabled = false;
    std::atomic<bool>     is_rehashing{false};
    std::thread           rehash_thread;
    std::mutex            rehash_mutex;
    std::condition_variable rehash_cv;
    std::atomic<bool>     stop_rehash_thread{false};
    std::atomic<uint32_t> active_inserts{0};
    cudaStream_t          rehash_stream = nullptr;

    StashQueue*  d_stash_queue        = nullptr;
    uint32_t*    h_needs_rehash_flag  = nullptr;
    uint32_t*    d_needs_rehash_flag  = nullptr;

public:
    WarpKVAllocator* allocator_ = nullptr;
    std::unique_ptr<WarpKVAllocator> default_allocator_;
    WarpKVEngine();
    ~WarpKVEngine();

    // Non-copyable, non-movable.
    WarpKVEngine(const WarpKVEngine&)            = delete;
    WarpKVEngine& operator=(const WarpKVEngine&) = delete;

    void init(uint32_t num_buckets, WarpKVAllocator* allocator = nullptr);
    void build_graphs();

    // =========================================================================
    // Asynchronous submit API
    // =========================================================================
    //
    // These functions launch the CUDA Graph and return immediately without
    // waiting for GPU completion. The returned future becomes ready once the
    // D→H copy is finished and host output buffers are safe to read.
    //
    // IMPORTANT: The host input buffers (keys, values) must remain valid until
    // the returned future is ready — the H→D copy may not have started yet
    // when submit returns. Call future.wait() or future.get() before
    // invalidating input data.
    //
    // Key 0 (EMPTY_KEY) is reserved and silently ignored on insert/delete.
    // =========================================================================

    /// Insert a batch of (key, value) pairs asynchronously.
    /// @param keys    Host pointer to KeyT array.
    /// @param values  Host pointer to ValueT array.
    /// @param count   Number of key-value pairs (must be <= BATCH_SIZE).
    /// @return Future that completes when the GPU pipeline finishes.
    std::future<void> submit_insert_batch(
        const KeyT* keys,
        const ValueT* values,
        uint32_t      count);

    /// Look up a batch of keys asynchronously.
    /// @param keys   Host pointer to KeyT array.
    /// @param count  Number of keys (must be <= BATCH_SIZE).
    /// @return Future containing a LookupFutureResult with the output values.
    std::future<LookupFutureResult> submit_lookup_batch(
        const KeyT* keys,
        uint32_t    count);

    /// Look up a batch of keys asynchronously writing results into user-supplied buffer.
    /// @param keys        Host pointer to KeyT array.
    /// @param values_out  Host pointer to ValueT output buffer (must be valid until future completes).
    /// @param count       Number of keys (must be <= BATCH_SIZE).
    /// @return Future that completes when D->H copy to values_out finishes.
    std::future<void> submit_lookup_batch(
        const KeyT* keys,
        ValueT*     values_out,
        uint32_t    count);

    /// Delete a batch of keys asynchronously.
    /// @param keys   Host pointer to KeyT array.
    /// @param count  Number of keys (must be <= BATCH_SIZE).
    /// @return Future that completes when the GPU pipeline finishes.
    std::future<void> submit_delete_batch(
        const KeyT* keys,
        uint32_t    count);

    // =========================================================================
    // Synchronous convenience wrappers (blocking)
    // =========================================================================
    //
    // These call submit_*_batch and immediately call .get() on the returned
    // future. They exist for backward compatibility and testing. Python
    // bindings use these so that GIL release + blocking happens correctly.
    // =========================================================================

    /// Blocking insert — equivalent to submit_insert_batch(...).get().
    void submit_insert_batch_sync(
        const KeyT* keys,
        const ValueT* values,
        uint32_t      count);

    /// Blocking lookup — writes results directly into values_out.
    void submit_lookup_batch_sync(
        const KeyT* keys,
        ValueT*     values_out,
        uint32_t    count);

    /// Blocking delete — equivalent to submit_delete_batch(...).get().
    void submit_delete_batch_sync(
        const KeyT* keys,
        uint32_t    count);

    /// Wait for all in-flight pipelines to drain.
    void sync_all();

    // Config

    // =========================================================================
    // Zero-Copy Device API (PyTorch Integration)
    // =========================================================================
    std::future<void> submit_insert_batch_device(
        const KeyT* d_keys,
        const ValueT* d_values,
        uint32_t count,
        cudaStream_t stream);

    std::future<void> submit_lookup_batch_device(
        const KeyT* d_keys,
        ValueT* d_values_out,
        uint32_t count,
        cudaStream_t stream);

    std::future<void> submit_delete_batch_device(
        const KeyT* d_keys,
        uint32_t count,
        cudaStream_t stream);

    void disable_automatic_rehash(bool disable) { auto_rehash_disabled = disable; }

private:
    BucketTable* acquire_table(uint64_t& out_epoch);
    void         release_table(uint64_t epoch);
    void         rehash_worker();
    void         update_graph_nodes(int slot, BucketTable* current_tbl);
    void         apply_backpressure();
};

} // namespace warpkv
