import re

with open('src/engine/warpkv_engine.h', 'r') as f:
    h = f.read()

device_api = """
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
"""

if 'Zero-Copy Device API' not in h:
    h = h.replace('    void disable_automatic_rehash(bool disable) { auto_rehash_disabled = disable; }', 
                  device_api + '\n    void disable_automatic_rehash(bool disable) { auto_rehash_disabled = disable; }')

with open('src/engine/warpkv_engine.h', 'w') as f:
    f.write(h)

with open('src/engine/warpkv_engine.cu', 'r') as f:
    c = f.read()

impl = """
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
"""

if 'submit_insert_batch_device' not in c:
    c = c.replace('void WarpKVEngine::submit_insert_batch_sync(', impl + '\nvoid WarpKVEngine::submit_insert_batch_sync(')

with open('src/engine/warpkv_engine.cu', 'w') as f:
    f.write(c)

