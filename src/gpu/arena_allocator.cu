#include "bucket_cuckoo.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace warpkv {

// ============================================================================
// Arena Allocator — Static allocation at startup
// ============================================================================

class ArenaAllocator {
public:
    // Singleton instance
    static ArenaAllocator& instance() {
        static ArenaAllocator allocator;
        return allocator;
    }

    // Initialize the arena (called once at startup)
    // Allocates two tables (EBR double-buffering) and stash
    void init() {
        if (initialized_) {
            throw std::runtime_error("ArenaAllocator already initialized");
        }

        // Query VRAM
        size_t free_vram, total_vram;
        cudaError_t err = cudaMemGetInfo(&free_vram, &total_vram);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                                   cudaGetErrorString(err));
        }

        std::cout << "GPU Memory: " << (total_vram / (1024 * 1024 * 1024)) << " GB total, "
                  << (free_vram / (1024 * 1024 * 1024)) << " GB free" << std::endl;

        // Allocate table arenas (EBR: two full copies)
        // Each table: ~750 MB (tuned for 2GB MX130)
        arena_size_per_table_ = 750 * 1024 * 1024;

        if (arena_size_per_table_ * 2 > free_vram * 0.9) {
            throw std::runtime_error("Insufficient VRAM for double-buffered tables");
        }

        // Allocate arena 0
        err = cudaMalloc(&arena0_, arena_size_per_table_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc arena0 failed: ") +
                                   cudaGetErrorString(err));
        }

        // Allocate arena 1
        err = cudaMalloc(&arena1_, arena_size_per_table_);
        if (err != cudaSuccess) {
            cudaFree(arena0_);
            throw std::runtime_error(std::string("cudaMalloc arena1 failed: ") +
                                   cudaGetErrorString(err));
        }

        std::cout << "Allocated " << (arena_size_per_table_ / (1024 * 1024)) << " MB per table"
                  << std::endl;

        // Configure table 0
        configure_table(&table0_, (Bucket*)arena0_);

        // Configure table 1
        configure_table(&table1_, (Bucket*)arena1_);

        std::cout << "Table 0: " << table0_.num_buckets << " buckets (mask: 0x"
                  << std::hex << table0_.bucket_mask << std::dec << ")" << std::endl;
        std::cout << "Table 1: " << table1_.num_buckets << " buckets (mask: 0x"
                  << std::hex << table1_.bucket_mask << std::dec << ")" << std::endl;

        // Allocate stash queue in mapped pinned memory
        err = cudaMallocHost(&stash_queue_, sizeof(StashQueue));
        if (err != cudaSuccess) {
            cudaFree(arena0_);
            cudaFree(arena1_);
            throw std::runtime_error(std::string("cudaMallocHost stash failed: ") +
                                   cudaGetErrorString(err));
        }

        // Get device pointer to mapped memory
        err = cudaHostGetDevicePointer(&d_stash_queue_, stash_queue_, 0);
        if (err != cudaSuccess) {
            cudaFreeHost(stash_queue_);
            cudaFree(arena0_);
            cudaFree(arena1_);
            throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                   cudaGetErrorString(err));
        }

        // Initialize stash
        stash_queue_->head = 0;
        stash_queue_->tail = 0;
        stash_queue_->needs_rehash = 0;

        std::cout << "Stash allocated: " << STASH_CAPACITY << " entries (~"
                  << (sizeof(StashQueue) / 1024) << " KB)" << std::endl;

        initialized_ = true;
    }

    // Cleanup (destructor)
    ~ArenaAllocator() {
        if (arena0_) cudaFree(arena0_);
        if (arena1_) cudaFree(arena1_);
        if (stash_queue_) cudaFreeHost(stash_queue_);
    }

    // Get pointers to tables
    BucketTable* get_table0() { return &table0_; }
    BucketTable* get_table1() { return &table1_; }
    StashQueue* get_stash() { return stash_queue_; }
    StashQueue* get_device_stash() { return d_stash_queue_; }

private:
    ArenaAllocator() = default;

    // Configure a table within an arena
    void configure_table(BucketTable* table, Bucket* arena_base) {
        // Calculate bucket count (power of 2)
        size_t raw_buckets = arena_size_per_table_ / sizeof(Bucket);
        size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));

        if (num_buckets == 0) {
            throw std::runtime_error("Arena too small to allocate any buckets");
        }

        table->buckets = arena_base;
        table->num_buckets = num_buckets;
        table->bucket_mask = (uint32_t)(num_buckets - 1);
        table->load_factor_limit = num_buckets / 2;  // 50% load factor

        // Initialize all buckets by clearing memory on the device
        cudaMemset(arena_base, 0, num_buckets * sizeof(Bucket));
    }

    // Initialization flag
    bool initialized_ = false;

    // Arena allocations
    void* arena0_ = nullptr;
    void* arena1_ = nullptr;
    size_t arena_size_per_table_ = 0;

    // Table structures
    BucketTable table0_ = {};
    BucketTable table1_ = {};

    // Stash
    StashQueue* stash_queue_ = nullptr;
    StashQueue* d_stash_queue_ = nullptr;
};

// ============================================================================
// Public initialization function
// ============================================================================

void init_arena() {
    ArenaAllocator::instance().init();
}

BucketTable* get_table0() {
    return ArenaAllocator::instance().get_table0();
}

BucketTable* get_table1() {
    return ArenaAllocator::instance().get_table1();
}

StashQueue* get_stash() {
    return ArenaAllocator::instance().get_stash();
}

StashQueue* get_device_stash() {
    return ArenaAllocator::instance().get_device_stash();
}

}  // namespace warpkv
