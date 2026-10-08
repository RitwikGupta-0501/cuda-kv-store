#include "../include/warpkv/warpkv_allocator.h"
#include "warpkv_engine.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include "../gpu/rehash_kernel.h"
#include "../gpu/cuckoo_delete.h"

namespace warpkv {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(                                           \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " - " +                             \
                cudaGetErrorString(_err));                                      \
        }                                                                       \
    } while (0)

// ============================================================================
// Constructor / Destructor
// ============================================================================

WarpKVEngine::WarpKVEngine() {}

WarpKVEngine::~WarpKVEngine() {
    sync_all();
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
    if (rehash_stream)    cudaStreamDestroy(rehash_stream);
    if (d_stash_queue)    cudaFree(d_stash_queue);
    if (h_needs_rehash_flag) cudaFreeHost(h_needs_rehash_flag);

    for (int i = 0; i < NUM_SLOTS; ++i) {
        if (streams[i].h2d)    cudaStreamDestroy(streams[i].h2d);
        if (streams[i].compute) cudaStreamDestroy(streams[i].compute);
        if (streams[i].d2h)    cudaStreamDestroy(streams[i].d2h);

        if (ev_h2d[i])    cudaEventDestroy(ev_h2d[i]);
        if (ev_compute[i]) cudaEventDestroy(ev_compute[i]);
        if (ev_d2h[i])    cudaEventDestroy(ev_d2h[i]);

        if (insert_graphs[i])           cudaGraphExecDestroy(insert_graphs[i]);
        if (lookup_graphs[i])           cudaGraphExecDestroy(lookup_graphs[i]);
        if (delete_graphs[i])           cudaGraphExecDestroy(delete_graphs[i]);
        if (template_insert_graphs[i])  cudaGraphDestroy(template_insert_graphs[i]);
        if (template_lookup_graphs[i])  cudaGraphDestroy(template_lookup_graphs[i]);
        if (template_delete_graphs[i])  cudaGraphDestroy(template_delete_graphs[i]);

        if (h_keys_in[i])          cudaFreeHost(h_keys_in[i]);
        if (h_values_in[i])        cudaFreeHost(h_values_in[i]);
        if (h_values_out[i])       cudaFreeHost(h_values_out[i]);
        if (h_insert_statuses[i])  cudaFreeHost(h_insert_statuses[i]);
        if (h_lookup_found[i])     cudaFreeHost(h_lookup_found[i]);

        if (d_keys_in[i])          cudaFree(d_keys_in[i]);
        if (d_values_in[i])        cudaFree(d_values_in[i]);
        if (d_values_out[i])       cudaFree(d_values_out[i]);
        if (d_insert_statuses[i])  cudaFree(d_insert_statuses[i]);
        if (d_lookup_found[i])     cudaFree(d_lookup_found[i]);
    }
}

// ============================================================================
// Init
// ============================================================================

void WarpKVEngine::init(uint32_t num_buckets, WarpKVAllocator* allocator) {
    if (allocator) {
        allocator_ = allocator;
    } else {
        default_allocator_ = std::make_unique<DefaultCudaAllocator>();
        allocator_ = default_allocator_.get();
    }

    // Double-buffered arena descriptors on pinned host memory.
    CUDA_CHECK(cudaHostAlloc(&epoch_table.arenas[0], sizeof(BucketTable), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&epoch_table.arenas[1], sizeof(BucketTable), cudaHostAllocDefault));

    CUDA_CHECK(cudaMalloc(&epoch_table.arenas[0]->buckets, num_buckets * sizeof(Bucket)));
    epoch_table.arenas[1]->buckets = nullptr;

    epoch_table.arenas[0]->num_buckets       = num_buckets;
    epoch_table.arenas[0]->bucket_mask       = num_buckets - 1;
    epoch_table.arenas[0]->load_factor_limit = num_buckets / 2;
    CUDA_CHECK(cudaMemset(epoch_table.arenas[0]->buckets, 0, num_buckets * sizeof(Bucket)));

    epoch_table.epoch.store(0, std::memory_order_seq_cst);
    epoch_table.readers[0].store(0, std::memory_order_seq_cst);
    epoch_table.readers[1].store(0, std::memory_order_seq_cst);

    // Stash
    CUDA_CHECK(cudaMalloc(&d_stash_queue, sizeof(StashQueue)));
    CUDA_CHECK(cudaMemset(d_stash_queue, 0, sizeof(StashQueue)));

    // Mapped rehash flag (GPU sets, CPU reads without D→H copy)
    CUDA_CHECK(cudaHostAlloc(&h_needs_rehash_flag, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_needs_rehash_flag, h_needs_rehash_flag, 0));
    *h_needs_rehash_flag = 0;

    for (int i = 0; i < NUM_SLOTS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i].h2d));
        CUDA_CHECK(cudaStreamCreate(&streams[i].compute));
        CUDA_CHECK(cudaStreamCreate(&streams[i].d2h));

        // ev_d2h must not use cudaEventBlockingSync so that cudaEventQuery
        // (used by the future poller) doesn't block the CPU.
        CUDA_CHECK(cudaEventCreate(&ev_h2d[i]));
        CUDA_CHECK(cudaEventCreate(&ev_compute[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_d2h[i], cudaEventDisableTiming));

        CUDA_CHECK(cudaHostAlloc(&h_keys_in[i],         BATCH_SIZE * sizeof(KeyT),    cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_values_in[i],       BATCH_SIZE * sizeof(ValueT),    cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_values_out[i],      BATCH_SIZE * sizeof(ValueT),    cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_insert_statuses[i], BATCH_SIZE * sizeof(InsertStatus), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_lookup_found[i],    BATCH_SIZE * sizeof(uint32_t),    cudaHostAllocDefault));

        CUDA_CHECK(cudaMalloc(&d_keys_in[i],         BATCH_SIZE * sizeof(KeyT)));
        CUDA_CHECK(cudaMalloc(&d_values_in[i],       BATCH_SIZE * sizeof(ValueT)));
        CUDA_CHECK(cudaMalloc(&d_values_out[i],      BATCH_SIZE * sizeof(ValueT)));
        CUDA_CHECK(cudaMalloc(&d_insert_statuses[i], BATCH_SIZE * sizeof(InsertStatus)));
        CUDA_CHECK(cudaMalloc(&d_lookup_found[i],    BATCH_SIZE * sizeof(uint32_t)));
    }

    CUDA_CHECK(cudaStreamCreate(&rehash_stream));
    rehash_thread = std::thread(&WarpKVEngine::rehash_worker, this);

    slot_pool.init(NUM_SLOTS);
    build_graphs();
}

// ============================================================================
// build_graphs
// ============================================================================

void WarpKVEngine::build_graphs() {
    dim3 block(256);
    dim3 grid(BATCH_SIZE / 8);

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {

        // ---- INSERT GRAPH --------------------------------------------------
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));

        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot],   h_keys_in[slot],   BATCH_SIZE * sizeof(KeyT), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_values_in[slot], h_values_in[slot], BATCH_SIZE * sizeof(ValueT), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        warp_insert_kernel<<<grid, block, 0, streams[slot].compute>>>(
            epoch_table.arenas[0][0], d_stash_queue, d_needs_rehash_flag,
            d_keys_in[slot], d_values_in[slot], d_insert_statuses[slot], nullptr, BATCH_SIZE);
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_insert_statuses[slot], d_insert_statuses[slot], BATCH_SIZE * sizeof(InsertStatus), cudaMemcpyDeviceToHost, streams[slot].d2h));
        // ev_d2h signals that D→H copy is done — the future polls this event.
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));

        // Join: h2d stream waits for d2h to complete before next batch can reuse buffers.
        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0));

        cudaGraph_t insert_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &insert_graph));

        // Find and store the kernel node for later epoch patching.
        size_t num_nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(insert_graph, nullptr, &num_nodes));
        std::vector<cudaGraphNode_t> nodes(num_nodes);
        CUDA_CHECK(cudaGraphGetNodes(insert_graph, nodes.data(), &num_nodes));
        for (size_t n = 0; n < num_nodes; ++n) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[n], &type));
            if (type == cudaGraphNodeTypeKernel) {
                insert_nodes[slot] = nodes[n];
                break;
            }
        }
        CUDA_CHECK(cudaGraphInstantiate(&insert_graphs[slot], insert_graph, nullptr, nullptr, 0));
        template_insert_graphs[slot] = insert_graph;

        // ---- LOOKUP GRAPH --------------------------------------------------
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));

        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot], h_keys_in[slot], BATCH_SIZE * sizeof(KeyT), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        warp_lookup_kernel<<<grid, block, 0, streams[slot].compute>>>(
            epoch_table.arenas[0][0], d_stash_queue,
            d_keys_in[slot], d_values_out[slot], d_lookup_found[slot], BATCH_SIZE);
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_values_out[slot],    d_values_out[slot],   BATCH_SIZE * sizeof(ValueT), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaMemcpyAsync(h_lookup_found[slot],  d_lookup_found[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0));

        cudaGraph_t lookup_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &lookup_graph));

        num_nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(lookup_graph, nullptr, &num_nodes));
        nodes.resize(num_nodes);
        CUDA_CHECK(cudaGraphGetNodes(lookup_graph, nodes.data(), &num_nodes));
        for (size_t n = 0; n < num_nodes; ++n) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[n], &type));
            if (type == cudaGraphNodeTypeKernel) {
                lookup_nodes[slot] = nodes[n];
                break;
            }
        }
        CUDA_CHECK(cudaGraphInstantiate(&lookup_graphs[slot], lookup_graph, nullptr, nullptr, 0));
        template_lookup_graphs[slot] = lookup_graph;

        // ---- DELETE GRAPH --------------------------------------------------
        CUDA_CHECK(cudaStreamBeginCapture(streams[slot].h2d, cudaStreamCaptureModeGlobal));

        CUDA_CHECK(cudaMemcpyAsync(d_keys_in[slot], h_keys_in[slot], BATCH_SIZE * sizeof(KeyT), cudaMemcpyHostToDevice, streams[slot].h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot].h2d));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].compute, ev_h2d[slot], 0));
        // Reuse d_lookup_found buffer for delete flags.
        warp_delete_kernel<<<grid, block, 0, streams[slot].compute>>>(
            epoch_table.arenas[0][0], d_stash_queue,
            d_keys_in[slot], d_lookup_found[slot], BATCH_SIZE);
        CUDA_CHECK(cudaEventRecord(ev_compute[slot], streams[slot].compute));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].d2h, ev_compute[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(h_lookup_found[slot], d_lookup_found[slot], BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[slot].d2h));
        CUDA_CHECK(cudaEventRecord(ev_d2h[slot], streams[slot].d2h));

        CUDA_CHECK(cudaStreamWaitEvent(streams[slot].h2d, ev_d2h[slot], 0));

        cudaGraph_t delete_graph;
        CUDA_CHECK(cudaStreamEndCapture(streams[slot].h2d, &delete_graph));

        num_nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(delete_graph, nullptr, &num_nodes));
        nodes.resize(num_nodes);
        CUDA_CHECK(cudaGraphGetNodes(delete_graph, nodes.data(), &num_nodes));
        for (size_t n = 0; n < num_nodes; ++n) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[n], &type));
            if (type == cudaGraphNodeTypeKernel) {
                delete_nodes[slot] = nodes[n];
                break;
            }
        }
        CUDA_CHECK(cudaGraphInstantiate(&delete_graphs[slot], delete_graph, nullptr, nullptr, 0));
        template_delete_graphs[slot] = delete_graph;
    }
}

// ============================================================================
// EBR helpers
// ============================================================================

void WarpKVEngine::apply_backpressure() {
    if (auto_rehash_disabled) return;
    if (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0) {
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
        if (e == epoch_table.epoch.load(std::memory_order_seq_cst)) break;
        epoch_table.readers[e & 1].fetch_sub(1, std::memory_order_seq_cst);
    }
    out_epoch = e;
    return epoch_table.arenas[e & 1];
}

void WarpKVEngine::release_table(uint64_t epoch) {
    epoch_table.readers[epoch & 1].fetch_sub(1, std::memory_order_seq_cst);
}

void WarpKVEngine::update_graph_nodes(int slot, BucketTable* current_tbl) {
    dim3    block(256);
    dim3    grid(BATCH_SIZE / 8);
    uint32_t batch_size = BATCH_SIZE;
    uint32_t* null_ptr  = nullptr;

    void* lookup_args[] = {
        current_tbl,
        &d_stash_queue,
        &d_keys_in[slot],
        &d_values_out[slot],
        &d_lookup_found[slot],
        &batch_size
    };
    cudaKernelNodeParams lookup_params = {0};
    lookup_params.func            = (void*)warp_lookup_kernel;
    lookup_params.gridDim         = grid;
    lookup_params.blockDim        = block;
    lookup_params.sharedMemBytes  = 0;
    lookup_params.kernelParams    = lookup_args;
    lookup_params.extra           = nullptr;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(lookup_graphs[slot], lookup_nodes[slot], &lookup_params));

    void* insert_args[] = {
        current_tbl,
        &d_stash_queue,
        &d_needs_rehash_flag,
        &d_keys_in[slot],
        &d_values_in[slot],
        &d_insert_statuses[slot],
        &null_ptr,
        &batch_size
    };
    cudaKernelNodeParams insert_params = {0};
    insert_params.func            = (void*)warp_insert_kernel;
    insert_params.gridDim         = grid;
    insert_params.blockDim        = block;
    insert_params.sharedMemBytes  = 0;
    insert_params.kernelParams    = insert_args;
    insert_params.extra           = nullptr;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(insert_graphs[slot], insert_nodes[slot], &insert_params));

    void* delete_args[] = {
        current_tbl,
        &d_stash_queue,
        &d_keys_in[slot],
        &d_lookup_found[slot],
        &batch_size
    };
    cudaKernelNodeParams delete_params = {0};
    delete_params.func            = (void*)warp_delete_kernel;
    delete_params.gridDim         = grid;
    delete_params.blockDim        = block;
    delete_params.sharedMemBytes  = 0;
    delete_params.kernelParams    = delete_args;
    delete_params.extra           = nullptr;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(delete_graphs[slot], delete_nodes[slot], &delete_params));
}

// ============================================================================
// Rehash worker thread
// ============================================================================

void WarpKVEngine::rehash_worker() {
    while (!stop_rehash_thread) {
        {
            std::unique_lock<std::mutex> lock(rehash_mutex);
            rehash_cv.wait(lock, [this]() {
                return (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0)
                    || stop_rehash_thread;
            });
        }

        if (stop_rehash_thread) break;

        is_rehashing.store(true, std::memory_order_seq_cst);

        // Drain in-flight inserts before touching the table.
        while (active_inserts.load(std::memory_order_seq_cst) > 0) {
            std::this_thread::yield();
        }

        const uint64_t  old_epoch = epoch_table.epoch.load(std::memory_order_seq_cst);
        BucketTable*    old_tbl   = epoch_table.arenas[old_epoch & 1];
        BucketTable*    new_tbl   = epoch_table.arenas[(old_epoch + 1) & 1];

        const uint32_t new_num = old_tbl->num_buckets * 2;
        new_tbl->num_buckets       = new_num;
        new_tbl->bucket_mask       = new_num - 1;
        new_tbl->load_factor_limit = new_num / 2;
        CUDA_CHECK(cudaMalloc(&new_tbl->buckets, new_num * sizeof(Bucket)));
        CUDA_CHECK(cudaMemsetAsync(new_tbl->buckets, 0, new_num * sizeof(Bucket), rehash_stream));

        RehashContext ctx;
        ctx.old_table = *old_tbl;
        ctx.new_table = *new_tbl;
        ctx.d_stash   = d_stash_queue;

        RehashStats stats;
        execute_rehash(ctx, &stats, rehash_stream);

        epoch_table.epoch.store(old_epoch + 1, std::memory_order_seq_cst);

        // Wait for all readers on the old epoch to drain.
        while (epoch_table.readers[old_epoch & 1].load(std::memory_order_seq_cst) > 0) {
            std::this_thread::yield();
        }

        // Lock all slot mutexes before freeing old buckets and patching graphs.
        for (int i = 0; i < NUM_SLOTS; ++i) slot_mutex[i].lock();

        allocator_->deallocate(old_tbl->buckets);
        old_tbl->buckets = nullptr;

        for (int i = 0; i < NUM_SLOTS; ++i) {
            update_graph_nodes(i, new_tbl);
            active_epoch[i] = old_epoch + 1;
        }

        for (int i = 0; i < NUM_SLOTS; ++i) slot_mutex[i].unlock();

        __atomic_store_n(h_needs_rehash_flag, 0, __ATOMIC_RELEASE);
        is_rehashing.store(false, std::memory_order_release);
    }
}

// ============================================================================
// Async submit host callbacks and context types
// ============================================================================

struct VoidContext {
    WarpKVEngine* engine;
    int           slot;
    uint64_t      epoch;
    bool          is_insert;
    std::promise<void> promise;
};

struct LookupVectorContext {
    WarpKVEngine* engine;
    int           slot;
    uint64_t      epoch;
    const ValueT* slot_out;
    uint32_t      count;
    std::promise<LookupFutureResult> promise;
};

struct LookupBufferContext {
    WarpKVEngine* engine;
    int           slot;
    uint64_t      epoch;
    ValueT*       user_out;
    const ValueT* slot_out;
    uint32_t      count;
    std::promise<void> promise;
};

void CUDART_CB WarpKVEngine::host_void_callback(void* data) {
    auto* ctx = static_cast<VoidContext*>(data);
    try {
        ctx->engine->release_table(ctx->epoch);
        if (ctx->is_insert) {
            ctx->engine->active_inserts.fetch_sub(1, std::memory_order_seq_cst);
        }
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_value();
    } catch (...) {
        ctx->engine->release_table(ctx->epoch);
        if (ctx->is_insert) {
            ctx->engine->active_inserts.fetch_sub(1, std::memory_order_seq_cst);
        }
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_exception(std::current_exception());
    }
    delete ctx;
}

void CUDART_CB WarpKVEngine::host_lookup_vector_callback(void* data) {
    auto* ctx = static_cast<LookupVectorContext*>(data);
    try {
        LookupFutureResult result;
        result.values.assign(ctx->slot_out, ctx->slot_out + ctx->count);
        ctx->engine->release_table(ctx->epoch);
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_value(std::move(result));
    } catch (...) {
        ctx->engine->release_table(ctx->epoch);
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_exception(std::current_exception());
    }
    delete ctx;
}

void CUDART_CB WarpKVEngine::host_lookup_buffer_callback(void* data) {
    auto* ctx = static_cast<LookupBufferContext*>(data);
    try {
        if (ctx->user_out && ctx->count > 0) {
            std::memcpy(ctx->user_out, ctx->slot_out, ctx->count * sizeof(ValueT));
        }
        ctx->engine->release_table(ctx->epoch);
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_value();
    } catch (...) {
        ctx->engine->release_table(ctx->epoch);
        ctx->engine->slot_pool.release(ctx->slot);
        ctx->promise.set_exception(std::current_exception());
    }
    delete ctx;
}

// ============================================================================
// Async submit: insert
// ============================================================================

std::future<void> WarpKVEngine::submit_insert_batch(
    const KeyT* keys,
    const ValueT* values,
    uint32_t        count)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    // Spin until any in-flight rehash is complete before accepting new inserts.
    while (true) {
        if (auto_rehash_disabled) {
            active_inserts.fetch_add(1, std::memory_order_seq_cst);
            break;
        }
        while (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0 ||
               is_rehashing.load(std::memory_order_acquire)) {
            apply_backpressure();
            std::this_thread::yield();
        }
        active_inserts.fetch_add(1, std::memory_order_seq_cst);
        if (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0 ||
            is_rehashing.load(std::memory_order_seq_cst)) {
            active_inserts.fetch_sub(1, std::memory_order_seq_cst);
            continue;
        }
        break;
    }

    const int slot = slot_pool.acquire();
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);

    // Drain the previous batch on this slot before overwriting its pinned buffers.
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));

    uint64_t     epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }

    std::memcpy(h_keys_in[slot],   keys,   count * sizeof(KeyT));
    std::memcpy(h_values_in[slot], values, count * sizeof(ValueT));

    // Pad remainder with EMPTY_KEY so the fixed-size kernel ignores them.
    for (uint32_t i = count; i < BATCH_SIZE; ++i) {
        h_keys_in[slot][i] = EMPTY_KEY;
    }

    CUDA_CHECK(cudaGraphLaunch(insert_graphs[slot], streams[slot].h2d));

    auto* ctx = new VoidContext{this, slot, epoch, true, {}};
    auto fut = ctx->promise.get_future();
    CUDA_CHECK(cudaLaunchHostFunc(streams[slot].h2d, &WarpKVEngine::host_void_callback, ctx));

    return fut;
}

// ============================================================================
// Async submit: lookup
// ============================================================================

std::future<LookupFutureResult> WarpKVEngine::submit_lookup_batch(
    const KeyT* keys,
    uint32_t        count)
{
    if (count == 0) {
        std::promise<LookupFutureResult> p;
        p.set_value(LookupFutureResult{});
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    apply_backpressure();

    const int slot = slot_pool.acquire();
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);

    // Drain previous batch on this slot.
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));

    apply_backpressure();

    uint64_t     epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }

    std::memcpy(h_keys_in[slot], keys, count * sizeof(KeyT));
    for (uint32_t i = count; i < BATCH_SIZE; ++i) {
        h_keys_in[slot][i] = EMPTY_KEY;
    }

    CUDA_CHECK(cudaGraphLaunch(lookup_graphs[slot], streams[slot].h2d));

    auto* ctx = new LookupVectorContext{this, slot, epoch, h_values_out[slot], count, {}};
    auto fut = ctx->promise.get_future();
    CUDA_CHECK(cudaLaunchHostFunc(streams[slot].h2d, &WarpKVEngine::host_lookup_vector_callback, ctx));

    return fut;
}

std::future<void> WarpKVEngine::submit_lookup_batch(
    const KeyT* keys,
    ValueT*       values_out,
    uint32_t        count)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    apply_backpressure();

    const int slot = slot_pool.acquire();
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);

    // Drain previous batch on this slot.
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));

    apply_backpressure();

    uint64_t     epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }

    std::memcpy(h_keys_in[slot], keys, count * sizeof(KeyT));
    for (uint32_t i = count; i < BATCH_SIZE; ++i) {
        h_keys_in[slot][i] = EMPTY_KEY;
    }

    CUDA_CHECK(cudaGraphLaunch(lookup_graphs[slot], streams[slot].h2d));

    auto* ctx = new LookupBufferContext{this, slot, epoch, values_out, h_values_out[slot], count, {}};
    auto fut = ctx->promise.get_future();
    CUDA_CHECK(cudaLaunchHostFunc(streams[slot].h2d, &WarpKVEngine::host_lookup_buffer_callback, ctx));

    return fut;
}

// ============================================================================
// Async submit: delete
// ============================================================================

std::future<void> WarpKVEngine::submit_delete_batch(
    const KeyT* keys,
    uint32_t        count)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    apply_backpressure();

    const int slot = slot_pool.acquire();
    std::lock_guard<std::mutex> lock(slot_mutex[slot]);

    // Drain previous batch on this slot.
    CUDA_CHECK(cudaStreamSynchronize(streams[slot].h2d));

    apply_backpressure();

    uint64_t     epoch;
    BucketTable* current_tbl = acquire_table(epoch);
    if (active_epoch[slot] != epoch) {
        update_graph_nodes(slot, current_tbl);
        active_epoch[slot] = epoch;
    }

    std::memcpy(h_keys_in[slot], keys, count * sizeof(KeyT));
    for (uint32_t i = count; i < BATCH_SIZE; ++i) {
        h_keys_in[slot][i] = EMPTY_KEY;
    }

    CUDA_CHECK(cudaGraphLaunch(delete_graphs[slot], streams[slot].h2d));

    auto* ctx = new VoidContext{this, slot, epoch, false, {}};
    auto fut = ctx->promise.get_future();
    CUDA_CHECK(cudaLaunchHostFunc(streams[slot].h2d, &WarpKVEngine::host_void_callback, ctx));

    return fut;
}

// ============================================================================
// Blocking sync wrappers
// ============================================================================


std::future<void> WarpKVEngine::submit_insert_batch_device(
    const KeyT* d_keys,
    const ValueT* d_values,
    uint32_t count,
    cudaStream_t stream)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    while (true) {
        if (auto_rehash_disabled) {
            active_inserts.fetch_add(1, std::memory_order_seq_cst);
            break;
        }
        while (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0 ||
               is_rehashing.load(std::memory_order_acquire)) {
            apply_backpressure();
            std::this_thread::yield();
        }
        active_inserts.fetch_add(1, std::memory_order_seq_cst);
        if (__atomic_load_n(h_needs_rehash_flag, __ATOMIC_ACQUIRE) != 0 ||
            is_rehashing.load(std::memory_order_seq_cst)) {
            active_inserts.fetch_sub(1, std::memory_order_seq_cst);
            continue;
        }
        break;
    }

    uint64_t epoch;
    BucketTable* current_tbl = acquire_table(epoch);

    dim3 block(512);
    dim3 grid((count * 16 + block.x - 1) / block.x);

    warp_insert_kernel<<<grid, block, 0, stream>>>(
        *current_tbl,
        d_stash_queue,
        d_needs_rehash_flag,
        d_keys,
        d_values,
        nullptr, // No insert statuses output for zero-copy
        nullptr,
        count
    );

    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, stream));

    return std::async(std::launch::async, [this, epoch, ev]() {
        cudaError_t status;
        do {
            status = cudaEventQuery(ev);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) {
                // Ignore error in cleanup thread, let it die
                break;
            }
            std::this_thread::yield();
        } while(true);
        
        this->release_table(epoch);
        this->active_inserts.fetch_sub(1, std::memory_order_seq_cst);
        cudaEventDestroy(ev);
    });
}

std::future<void> WarpKVEngine::submit_lookup_batch_device(
    const KeyT* d_keys,
    ValueT* d_values_out,
    uint32_t count,
    cudaStream_t stream)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    uint64_t epoch;
    BucketTable* current_tbl = acquire_table(epoch);

    dim3 block(512);
    dim3 grid((count * 16 + block.x - 1) / block.x);

    warp_lookup_kernel<<<grid, block, 0, stream>>>(
        *current_tbl,
        d_stash_queue,
        d_keys,
        d_values_out,
        nullptr,
        count
    );

    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, stream));

    return std::async(std::launch::async, [this, epoch, ev]() {
        cudaError_t status;
        do {
            status = cudaEventQuery(ev);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) break;
            std::this_thread::yield();
        } while(true);
        
        this->release_table(epoch);
        cudaEventDestroy(ev);
    });
}

std::future<void> WarpKVEngine::submit_delete_batch_device(
    const KeyT* d_keys,
    uint32_t count,
    cudaStream_t stream)
{
    if (count == 0) {
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }
    if (count > BATCH_SIZE) {
        throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
    }

    uint64_t epoch;
    BucketTable* current_tbl = acquire_table(epoch);

    dim3 block(512);
    dim3 grid((count * 16 + block.x - 1) / block.x);

    warp_delete_kernel<<<grid, block, 0, stream>>>(
        *current_tbl,
        d_stash_queue,
        d_keys,
        nullptr,
        count
    );

    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, stream));

    return std::async(std::launch::async, [this, epoch, ev]() {
        cudaError_t status;
        do {
            status = cudaEventQuery(ev);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) break;
            std::this_thread::yield();
        } while(true);
        
        this->release_table(epoch);
        cudaEventDestroy(ev);
    });
}

void WarpKVEngine::submit_insert_batch_sync(
    const KeyT* keys,
    const ValueT* values,
    uint32_t        count)
{
    submit_insert_batch(keys, values, count).get();
}

void WarpKVEngine::submit_lookup_batch_sync(
    const KeyT* keys,
    ValueT*       values_out,
    uint32_t        count)
{
    submit_lookup_batch(keys, values_out, count).get();
}

void WarpKVEngine::submit_delete_batch_sync(
    const KeyT* keys,
    uint32_t        count)
{
    submit_delete_batch(keys, count).get();
}

void WarpKVEngine::sync_all() {
    slot_pool.wait_all_idle();
    for (int i = 0; i < NUM_SLOTS; ++i) {
        std::lock_guard<std::mutex> lock(slot_mutex[i]);
        CUDA_CHECK(cudaStreamSynchronize(streams[i].h2d));
    }
}

} // namespace warpkv
