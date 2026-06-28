#include "warpkv_engine.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include "../gpu/rehash_kernel.h"

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
    {
        std::lock_guard<std::mutex> lock(rehash_mutex);
        stop_rehash_thread = true;
    }
    rehash_cv.notify_one();
    if (rehash_thread.joinable()) {
        rehash_thread.join();
    }
    
    if (epoch_table.arenas[0]) {
        cudaFree(epoch_table.arenas[0]->buckets);
        cudaFreeHost(epoch_table.arenas[0]);
    }
    if (epoch_table.arenas[1]) {
        cudaFree(epoch_table.arenas[1]->buckets);
        cudaFreeHost(epoch_table.arenas[1]);
    }
    if (rehash_stream) {
        cudaStreamDestroy(rehash_stream);
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
    // allocate double-buffered arenas (structs on host)
    CUDA_CHECK(cudaHostAlloc(&epoch_table.arenas[0], sizeof(BucketTable), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&epoch_table.arenas[1], sizeof(BucketTable), cudaHostAllocDefault));
    
    // Only allocate bucket memory for the first epoch
    CUDA_CHECK(cudaMalloc(&epoch_table.arenas[0]->buckets, num_buckets * sizeof(Bucket)));
    epoch_table.arenas[1]->buckets = nullptr;
    
    epoch_table.arenas[0]->num_buckets = num_buckets;
    epoch_table.arenas[0]->bucket_mask = num_buckets - 1;
    epoch_table.arenas[0]->load_factor_limit = num_buckets / 2;
    CUDA_CHECK(cudaMemset(epoch_table.arenas[0]->buckets, 0, num_buckets * sizeof(Bucket)));
    
    epoch_table.epoch.store(0, std::memory_order_seq_cst);
    epoch_table.readers[0].store(0, std::memory_order_seq_cst);
    epoch_table.readers[1].store(0, std::memory_order_seq_cst);
    
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
    
    CUDA_CHECK(cudaStreamCreate(&rehash_stream));
    rehash_thread = std::thread(&WarpKVEngine::rehash_worker, this);
    
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
            epoch_table.arenas[0][0], d_stash_queue, d_keys_in[slot], d_values_in[slot], d_insert_statuses[slot], nullptr, BATCH_SIZE
        );
        
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_insert_statuses[slot], d_insert_statuses[slot], BATCH_SIZE * sizeof(InsertStatus), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0)); // JOIN
        
        cudaGraph_t insert_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &insert_graph));
        
        size_t numNodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(insert_graph, nullptr, &numNodes));
        std::vector<cudaGraphNode_t> nodes(numNodes);
        CUDA_CHECK(cudaGraphGetNodes(insert_graph, nodes.data(), &numNodes));
        for (size_t i = 0; i < numNodes; ++i) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
            if (type == cudaGraphNodeTypeKernel) {
                insert_nodes[slot] = nodes[i];
                break;
            }
        }
        
        CUDA_CHECK(cudaGraphInstantiate(&insert_graphs[slot], insert_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(insert_graph));
        
        // ================= LOOKUP GRAPH =================
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));
        
        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot], h_keys_in[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        
        warp_lookup_kernel<<<grid, block, 0, streams[slot].compute>>>(
            epoch_table.arenas[0][0], d_keys_in[slot], d_values_out[slot], d_lookup_found[slot], BATCH_SIZE
        );
        
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_values_out[slot], d_values_out[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaMemcpyAsync(h_lookup_found[slot], d_lookup_found[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));
        
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0)); // JOIN
        
        cudaGraph_t lookup_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &lookup_graph));
        
        numNodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(lookup_graph, nullptr, &numNodes));
        nodes.resize(numNodes);
        CUDA_CHECK(cudaGraphGetNodes(lookup_graph, nodes.data(), &numNodes));
        for (size_t i = 0; i < numNodes; ++i) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
            if (type == cudaGraphNodeTypeKernel) {
                lookup_nodes[slot] = nodes[i];
                break;
            }
        }
        
        CUDA_CHECK(cudaGraphInstantiate(&lookup_graphs[slot], lookup_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(lookup_graph));
    }
}

void WarpKVEngine::apply_backpressure() {
    if (__atomic_load_n(&h_stash_queue->needs_rehash, __ATOMIC_ACQUIRE) != 0) {
        if (!is_rehashing.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(rehash_mutex);
            rehash_cv.notify_one();
        }
    }
}

BucketTable* WarpKVEngine::acquire_table(uint64_t& out_epoch) {
    uint64_t e;
    while (true) {
        e = epoch_table.epoch.load(std::memory_order_seq_cst);
        epoch_table.readers[e & 1].fetch_add(1, std::memory_order_seq_cst);
        if (e == epoch_table.epoch.load(std::memory_order_seq_cst)) {
            break;
        }
        epoch_table.readers[e & 1].fetch_sub(1, std::memory_order_seq_cst);
    }
    out_epoch = e;
    return epoch_table.arenas[e & 1];
}

void WarpKVEngine::release_table(uint64_t epoch) {
    epoch_table.readers[epoch & 1].fetch_sub(1, std::memory_order_seq_cst);
}

void WarpKVEngine::update_graph_nodes(int slot, BucketTable* current_tbl) {
    dim3 block(256);
    dim3 grid(BATCH_SIZE / 8);
    uint32_t batch_size = BATCH_SIZE;
    uint32_t* null_ptr = nullptr;
    
    void* lookup_args[] = { current_tbl, &d_keys_in[slot], &d_values_out[slot], &d_lookup_found[slot], &batch_size };
    cudaKernelNodeParams lookup_node_params = {0};
    lookup_node_params.func = (void*)warp_lookup_kernel;
    lookup_node_params.gridDim = grid;
    lookup_node_params.blockDim = block;
    lookup_node_params.sharedMemBytes = 0;
    lookup_node_params.kernelParams = lookup_args;
    lookup_node_params.extra = nullptr;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(lookup_graphs[slot], lookup_nodes[slot], &lookup_node_params));
    
    void* insert_args[] = { current_tbl, &d_stash_queue, &d_keys_in[slot], &d_values_in[slot], &d_insert_statuses[slot], &null_ptr, &batch_size };
    cudaKernelNodeParams insert_node_params = {0};
    insert_node_params.func = (void*)warp_insert_kernel;
    insert_node_params.gridDim = grid;
    insert_node_params.blockDim = block;
    insert_node_params.sharedMemBytes = 0;
    insert_node_params.kernelParams = insert_args;
    insert_node_params.extra = nullptr;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(insert_graphs[slot], insert_nodes[slot], &insert_node_params));
}

void WarpKVEngine::rehash_worker() {
    while (!stop_rehash_thread) {
        {
            std::unique_lock<std::mutex> lock(rehash_mutex);
            rehash_cv.wait(lock, [this]() { 
                return (__atomic_load_n(&h_stash_queue->needs_rehash, __ATOMIC_ACQUIRE) != 0) || stop_rehash_thread; 
            });
        }
        
        if (stop_rehash_thread) break;
        
        is_rehashing.store(true, std::memory_order_seq_cst);
        
        while (active_inserts.load(std::memory_order_seq_cst) > 0) {
            std::this_thread::yield();
        }
        
        uint64_t old_epoch = epoch_table.epoch.load(std::memory_order_seq_cst);
        BucketTable* old_tbl = epoch_table.arenas[old_epoch & 1];
        BucketTable* new_tbl = epoch_table.arenas[(old_epoch + 1) & 1];
        
        // Dynamically expand capacity by 2x
        uint32_t new_num = old_tbl->num_buckets * 2;
        new_tbl->num_buckets = new_num;
        new_tbl->bucket_mask = new_num - 1;
        new_tbl->load_factor_limit = new_num / 2;
        CUDA_CHECK(cudaMalloc(&new_tbl->buckets, new_num * sizeof(Bucket)));
        CUDA_CHECK(cudaMemsetAsync(new_tbl->buckets, 0, new_num * sizeof(Bucket), rehash_stream));
        
        RehashContext ctx;
        ctx.old_table = *old_tbl;
        ctx.new_table = *new_tbl;
        ctx.d_stash = d_stash_queue;
        
        RehashStats stats;
        execute_rehash(ctx, &stats, rehash_stream);
        
        epoch_table.epoch.store(old_epoch + 1, std::memory_order_seq_cst);
        
        while (epoch_table.readers[old_epoch & 1].load(std::memory_order_seq_cst) > 0) {
            std::this_thread::yield();
        }
        
        // Old table is fully drained, safe to free
        CUDA_CHECK(cudaFree(old_tbl->buckets));
        old_tbl->buckets = nullptr;
        
        __atomic_store_n(&h_stash_queue->needs_rehash, 0, __ATOMIC_RELEASE);
        
        is_rehashing.store(false, std::memory_order_release);
    }
}

void WarpKVEngine::submit_insert_batch(const uint32_t* keys, const uint32_t* values, uint32_t count) {
    if (count == 0) return;
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }
    
    while (true) {
        while (__atomic_load_n(&h_stash_queue->needs_rehash, __ATOMIC_ACQUIRE) != 0 || is_rehashing.load(std::memory_order_acquire)) {
            apply_backpressure();
            std::this_thread::yield();
        }
        
        active_inserts.fetch_add(1, std::memory_order_seq_cst);
        
        if (__atomic_load_n(&h_stash_queue->needs_rehash, __ATOMIC_ACQUIRE) != 0 || is_rehashing.load(std::memory_order_seq_cst)) {
            active_inserts.fetch_sub(1, std::memory_order_seq_cst);
            continue;
        }
        break;
    }
    
    int slot = current_slot.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);
    
    uint64_t epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }
    
    std::memcpy(h_keys_in[slot], keys, count * sizeof(uint32_t));
    std::memcpy(h_values_in[slot], values, count * sizeof(uint32_t));
    
    if (count < BATCH_SIZE) {
        for (uint32_t i = count; i < BATCH_SIZE; ++i) {
            h_keys_in[slot][i] = EMPTY_KEY;
        }
    }
    
    CUDA_CHECK(cudaGraphLaunch(insert_graphs[slot], streams[slot].h2d));
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));
    
    release_table(epoch);
    active_inserts.fetch_sub(1, std::memory_order_seq_cst);
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
    
    uint64_t epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }
    
    std::memcpy(h_keys_in[slot], keys, count * sizeof(uint32_t));
    
    if (count < BATCH_SIZE) {
        for (uint32_t i = count; i < BATCH_SIZE; ++i) {
            h_keys_in[slot][i] = EMPTY_KEY;
        }
    }
    
    CUDA_CHECK(cudaGraphLaunch(lookup_graphs[slot], streams[slot].h2d));
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));
    
    std::memcpy(values_out, h_values_out[slot], count * sizeof(uint32_t));
    
    release_table(epoch);
}

} // namespace warpkv
