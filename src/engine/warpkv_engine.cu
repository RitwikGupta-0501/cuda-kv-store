#include "warpkv_engine.h"
#include <stdexcept>
#include <iostream>
#include <string>

namespace warpkv {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while (0)

WarpKVEngine::WarpKVEngine() {}

WarpKVEngine::~WarpKVEngine() {
    if (current_table.buckets) {
        cudaFree(current_table.buckets);
    }
    if (h_stash_queue) {
        cudaFreeHost(h_stash_queue);
    }
    for (int i = 0; i < NUM_SLOTS; ++i) {
        if (streams[i].h2d) cudaStreamDestroy(streams[i].h2d);
        if (streams[i].compute) cudaStreamDestroy(streams[i].compute);
        if (streams[i].d2h) cudaStreamDestroy(streams[i].d2h);
        
        if (ev_h2d[i]) cudaEventDestroy(ev_h2d[i]);
        if (ev_compute[i]) cudaEventDestroy(ev_compute[i]);
        if (ev_d2h[i]) cudaEventDestroy(ev_d2h[i]);
        
        if (insert_graphs[i]) cudaGraphExecDestroy(insert_graphs[i]);
        if (lookup_graphs[i]) cudaGraphExecDestroy(lookup_graphs[i]);
        
        if (h_keys_in[i]) cudaFreeHost(h_keys_in[i]);
        if (h_values_in[i]) cudaFreeHost(h_values_in[i]);
        if (h_values_out[i]) cudaFreeHost(h_values_out[i]);
        if (h_insert_statuses[i]) cudaFreeHost(h_insert_statuses[i]);
        if (h_lookup_found[i]) cudaFreeHost(h_lookup_found[i]);
        
        if (d_keys_in[i]) cudaFree(d_keys_in[i]);
        if (d_values_in[i]) cudaFree(d_values_in[i]);
        if (d_values_out[i]) cudaFree(d_values_out[i]);
        if (d_insert_statuses[i]) cudaFree(d_insert_statuses[i]);
        if (d_lookup_found[i]) cudaFree(d_lookup_found[i]);
    }
}

void WarpKVEngine::init(uint32_t num_buckets) {
    // allocate table
    CUDA_CHECK(cudaMalloc(&current_table.buckets, num_buckets * sizeof(Bucket)));
    current_table.num_buckets = num_buckets;
    current_table.bucket_mask = num_buckets - 1;
    current_table.load_factor_limit = num_buckets / 2;
    
    // clear table
    CUDA_CHECK(cudaMemset(current_table.buckets, 0, num_buckets * sizeof(Bucket)));
    
    // allocate stash
    CUDA_CHECK(cudaHostAlloc(&h_stash_queue, sizeof(StashQueue), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_stash_queue, h_stash_queue, 0));
    std::memset(h_stash_queue, 0, sizeof(StashQueue));
    
    for (int i = 0; i < NUM_SLOTS; ++i) {
        // streams
        CUDA_CHECK(cudaStreamCreate(&streams[i].h2d));
        CUDA_CHECK(cudaStreamCreate(&streams[i].compute));
        CUDA_CHECK(cudaStreamCreate(&streams[i].d2h));
        
        // events
        CUDA_CHECK(cudaEventCreate(&ev_h2d[i]));
        CUDA_CHECK(cudaEventCreate(&ev_compute[i]));
        CUDA_CHECK(cudaEventCreate(&ev_d2h[i]));
        
        // host memory
        CUDA_CHECK(cudaHostAlloc(&h_keys_in[i], BATCH_SIZE * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_values_in[i], BATCH_SIZE * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_values_out[i], BATCH_SIZE * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_insert_statuses[i], BATCH_SIZE * sizeof(InsertStatus), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_lookup_found[i], BATCH_SIZE * sizeof(uint32_t), cudaHostAllocDefault));
        
        // device memory
        CUDA_CHECK(cudaMalloc(&d_keys_in[i], BATCH_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_in[i], BATCH_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_out[i], BATCH_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_insert_statuses[i], BATCH_SIZE * sizeof(InsertStatus)));
        CUDA_CHECK(cudaMalloc(&d_lookup_found[i], BATCH_SIZE * sizeof(uint32_t)));
    }
    
    build_graphs();
}

void WarpKVEngine::build_graphs() {
    dim3 block(256);
    dim3 grid(BATCH_SIZE / 8);
    
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        // ================= INSERT GRAPH =================
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));
        
        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot], h_keys_in[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_values_in[slot], h_values_in[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        
        warp_insert_kernel<<<grid, block, 0, streams[slot].compute>>>(
            current_table, d_stash_queue, d_keys_in[slot], d_values_in[slot], d_insert_statuses[slot], nullptr, BATCH_SIZE
        );
        
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_insert_statuses[slot], d_insert_statuses[slot], BATCH_SIZE * sizeof(InsertStatus), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0)); // JOIN
        
        cudaGraph_t insert_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &insert_graph));
        CUDA_CHECK(cudaGraphInstantiate(&insert_graphs[slot], insert_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(insert_graph));
        
        // ================= LOOKUP GRAPH =================
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));
        
        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot], h_keys_in[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        
        warp_lookup_kernel<<<grid, block, 0, streams[slot].compute>>>(
            current_table, d_keys_in[slot], d_values_out[slot], d_lookup_found[slot], BATCH_SIZE
        );
        
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_values_out[slot], d_values_out[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaMemcpyAsync(h_lookup_found[slot], d_lookup_found[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0)); // JOIN
        
        cudaGraph_t lookup_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &lookup_graph));
        CUDA_CHECK(cudaGraphInstantiate(&lookup_graphs[slot], lookup_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(lookup_graph));
    }
}

void WarpKVEngine::apply_backpressure() {
    if (__atomic_load_n(&h_stash_queue->needs_rehash, __ATOMIC_ACQUIRE) != 0) {
        throw std::runtime_error("Rehash triggered! Stash capacity exceeded.");
    }
    
    if (__atomic_load_n(&h_stash_queue->head, __ATOMIC_ACQUIRE) >= BACKPRESSURE_THRESHOLD) {
        throw std::runtime_error("Stash full, rehash not implemented in Phase 6");
    }
}

void WarpKVEngine::submit_insert_batch(const uint32_t* keys, const uint32_t* values, uint32_t count) {
    if (count == 0) return;
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }
    
    apply_backpressure();
    
    int slot = current_slot.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);
    
    apply_backpressure();
    
    std::memcpy(h_keys_in[slot], keys, count * sizeof(uint32_t));
    std::memcpy(h_values_in[slot], values, count * sizeof(uint32_t));
    
    if (count < BATCH_SIZE) {
        for (uint32_t i = count; i < BATCH_SIZE; ++i) {
            h_keys_in[slot][i] = EMPTY_KEY;
        }
    }
    
    CUDA_CHECK(cudaGraphLaunch(insert_graphs[slot], streams[slot].h2d));
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));
}

void WarpKVEngine::submit_lookup_batch(const uint32_t* keys, uint32_t* values_out, uint32_t count) {
    if (count == 0) return;
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }
    
    apply_backpressure();
    
    int slot = current_slot.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);
    
    apply_backpressure();
    
    std::memcpy(h_keys_in[slot], keys, count * sizeof(uint32_t));
    
    if (count < BATCH_SIZE) {
        for (uint32_t i = count; i < BATCH_SIZE; ++i) {
            h_keys_in[slot][i] = EMPTY_KEY;
        }
    }
    
    CUDA_CHECK(cudaGraphLaunch(lookup_graphs[slot], streams[slot].h2d));
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));
    
    std::memcpy(values_out, h_values_out[slot], count * sizeof(uint32_t));
}

} // namespace warpkv
