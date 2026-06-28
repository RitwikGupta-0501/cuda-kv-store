#pragma once

#include "../gpu/bucket_cuckoo.h"
#include "../gpu/cuckoo_insert.h"
#include "../gpu/warp_lookup.h"
#include <cuda_runtime.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>

namespace warpkv {

struct PipelineStreams {
    cudaStream_t h2d;
    cudaStream_t compute;
    cudaStream_t d2h;
};

class WarpKVEngine {
private:
    static constexpr uint32_t NUM_SLOTS = 3;
    
    // Pinned host memory
    uint32_t* h_keys_in[NUM_SLOTS] = {nullptr};
    uint32_t* h_values_in[NUM_SLOTS] = {nullptr};
    uint32_t* h_values_out[NUM_SLOTS] = {nullptr};
    InsertStatus* h_insert_statuses[NUM_SLOTS] = {nullptr};
    uint32_t* h_lookup_found[NUM_SLOTS] = {nullptr};
    
    // Device memory
    uint32_t* d_keys_in[NUM_SLOTS] = {nullptr};
    uint32_t* d_values_in[NUM_SLOTS] = {nullptr};
    uint32_t* d_values_out[NUM_SLOTS] = {nullptr};
    InsertStatus* d_insert_statuses[NUM_SLOTS] = {nullptr};
    uint32_t* d_lookup_found[NUM_SLOTS] = {nullptr};
    
    // Streams & events
    PipelineStreams streams[NUM_SLOTS] = {};
    cudaEvent_t ev_h2d[NUM_SLOTS] = {nullptr};
    cudaEvent_t ev_compute[NUM_SLOTS] = {nullptr};
    cudaEvent_t ev_d2h[NUM_SLOTS] = {nullptr};
    
    // Graphs
    cudaGraphExec_t lookup_graphs[NUM_SLOTS] = {nullptr};
    cudaGraphExec_t insert_graphs[NUM_SLOTS] = {nullptr};
    
    // Concurrency control
    std::atomic<uint32_t> current_slot{0};
    std::mutex slot_mutex[NUM_SLOTS]; 
    
    // Table and Backpressure
    // NOTE (Phase 7): current_table is captured by value into the static graphs.
    // When rehash happens, the graphs must be updated using cudaGraphExecKernelNodeSetParams
    // to point to the newly allocated buckets pointer.
    BucketTable current_table; // Simplified for Phase 6 (single table)
    StashQueue* h_stash_queue = nullptr;
    StashQueue* d_stash_queue = nullptr;
    
public:
    WarpKVEngine();
    ~WarpKVEngine();
    
    // Disable copy/move
    WarpKVEngine(const WarpKVEngine&) = delete;
    WarpKVEngine& operator=(const WarpKVEngine&) = delete;
    
    void init(uint32_t num_buckets);
    void build_graphs();
    
    // Note: 0xFFFFFFFFu (EMPTY_KEY) is a reserved forbidden key.
    // Inserting EMPTY_KEY will be silently ignored.
    void submit_insert_batch(const uint32_t* keys, const uint32_t* values, uint32_t count);
    
    // Note: Looking up EMPTY_KEY will always return NOT_FOUND.
    void submit_lookup_batch(const uint32_t* keys, uint32_t* values_out, uint32_t count);
    
private:
    void apply_backpressure();
};

} // namespace warpkv
