#pragma once

#include "../gpu/bucket_cuckoo.h"
#include "../gpu/cuckoo_insert.h"
#include "../gpu/warp_lookup.h"
#include <cuda_runtime.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <condition_variable>

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
    cudaGraphNode_t lookup_nodes[NUM_SLOTS] = {nullptr};
    cudaGraphNode_t insert_nodes[NUM_SLOTS] = {nullptr};
    cudaGraph_t template_insert_graphs[NUM_SLOTS] = {nullptr};
    cudaGraph_t template_lookup_graphs[NUM_SLOTS] = {nullptr};
    uint64_t active_epoch[NUM_SLOTS] = {0, 0, 0};
    
    // Concurrency control
    std::atomic<uint32_t> current_slot{0};
    std::mutex slot_mutex[NUM_SLOTS]; 
    
    // Table and EBR
    EpochTable epoch_table;
    std::atomic<bool> is_rehashing{false};
    std::thread rehash_thread;
    std::mutex rehash_mutex;
    std::condition_variable rehash_cv;
    std::atomic<bool> stop_rehash_thread{false};
    std::atomic<uint32_t> active_inserts{0};
    cudaStream_t rehash_stream = nullptr;
    
    StashQueue* d_stash_queue = nullptr;
    
    uint32_t* h_needs_rehash_flag = nullptr;
    uint32_t* d_needs_rehash_flag = nullptr;
    
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
    
    void sync_all();
    
    // Callbacks for pipelining
    void release_table_callback(uint64_t epoch);
    void decrement_active_inserts();
    
private:
    BucketTable* acquire_table(uint64_t& out_epoch);
    void release_table(uint64_t epoch);
    void rehash_worker();
    void update_graph_nodes(int slot, BucketTable* current_tbl);
    void apply_backpressure();
};

} // namespace warpkv
