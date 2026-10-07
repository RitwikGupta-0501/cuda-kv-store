#include <gtest/gtest.h>
#include "../../src/gpu/rehash_kernel.h"
#include "../../src/gpu/cuckoo_insert.h"
#include "../../src/gpu/warp_lookup.h"
#include "../../include/warpkv/warpkv_allocator.h"

using namespace warpkv;

class RehashKernelTest : public ::testing::Test {
protected:
    static constexpr uint32_t OLD_BUCKETS = 2048;
    static constexpr uint32_t NEW_BUCKETS = 4096;
    DefaultCudaAllocator allocator_;
    BucketTable old_table_storage_;
    BucketTable new_table_storage_;
    BucketTable* old_table_ = &old_table_storage_;
    BucketTable* new_table_ = &new_table_storage_;
    StashQueue* stash_ = nullptr;
    uint32_t* d_needs_rehash_flag_ = nullptr;

    void SetUp() override {
        old_table_storage_.num_buckets = OLD_BUCKETS;
        old_table_storage_.bucket_mask = OLD_BUCKETS - 1;
        old_table_storage_.load_factor_limit = OLD_BUCKETS / 2;
        old_table_storage_.buckets = static_cast<Bucket*>(
            allocator_.allocate(OLD_BUCKETS * sizeof(Bucket))
        );
        cudaMemset(old_table_storage_.buckets, 0, OLD_BUCKETS * sizeof(Bucket));

        new_table_storage_.num_buckets = NEW_BUCKETS;
        new_table_storage_.bucket_mask = NEW_BUCKETS - 1;
        new_table_storage_.load_factor_limit = NEW_BUCKETS / 2;
        new_table_storage_.buckets = static_cast<Bucket*>(
            allocator_.allocate(NEW_BUCKETS * sizeof(Bucket))
        );
        cudaMemset(new_table_storage_.buckets, 0, NEW_BUCKETS * sizeof(Bucket));

        stash_ = static_cast<StashQueue*>(
            allocator_.allocate(sizeof(StashQueue))
        );
        cudaMemset(stash_, 0, sizeof(StashQueue));

        cudaMalloc(&d_needs_rehash_flag_, sizeof(uint32_t));
        cudaMemset(d_needs_rehash_flag_, 0, sizeof(uint32_t));
    }

    void TearDown() override {
        if (old_table_storage_.buckets) {
            allocator_.deallocate(old_table_storage_.buckets);
            old_table_storage_.buckets = nullptr;
        }
        if (new_table_storage_.buckets) {
            allocator_.deallocate(new_table_storage_.buckets);
            new_table_storage_.buckets = nullptr;
        }
        if (stash_) {
            allocator_.deallocate(stash_);
            stash_ = nullptr;
        }
        if (d_needs_rehash_flag_) {
            cudaFree(d_needs_rehash_flag_);
            d_needs_rehash_flag_ = nullptr;
        }
    }
};

TEST_F(RehashKernelTest, RealRehashExecution) {
    // 1. Insert 1024 unique keys into old_table
    const uint32_t num_keys = 1024;
    std::vector<KeyT> keys(num_keys);
    std::vector<ValueT> values(num_keys);
    std::vector<InsertStatus> statuses(num_keys);
    
    for (uint32_t i = 0; i < num_keys; ++i) {
        keys[i] = 50000 + i;
        values[i] = 60000 + i;
    }

    InsertBatch batch;
    batch.h_keys = keys.data();
    batch.h_values = values.data();
    batch.h_statuses = statuses.data();
    batch.h_hops = nullptr;
    batch.num_keys = num_keys;

    warp_insert_batch_sync(*old_table_, stash_, d_needs_rehash_flag_, batch);

    // 2. Put a couple of entries manually in the stash to test stash drain
    StashQueue h_stash;
    cudaMemcpy(&h_stash, stash_, sizeof(StashQueue), cudaMemcpyDeviceToHost);
    
    KeyT stash_key1 = 999991;
    KeyT stash_key2 = 999992;
    h_stash.entries[0].key = stash_key1;
    h_stash.entries[0].value = 111111;
    h_stash.entries[1].key = stash_key2;
    h_stash.entries[1].value = 222222;
    h_stash.head = 2; // two items in stash
    
    cudaMemcpy(stash_, &h_stash, sizeof(StashQueue), cudaMemcpyHostToDevice);

    // 3. Execute Rehash Kernel
    RehashContext ctx;
    ctx.old_table = *old_table_;
    ctx.new_table = *new_table_;
    ctx.d_stash = stash_;

    RehashStats stats;
    execute_rehash(ctx, &stats, nullptr);

    EXPECT_EQ(stats.status, REHASH_COMPLETE) << "Rehash should complete successfully";
    
    // We inserted 1024 into buckets (maybe some hit stash, but likely 0).
    // Plus the 2 we forced into stash. The total copied + drained should be 1026.
    EXPECT_EQ(stats.entries_copied + stats.entries_stashed, num_keys + 2) << "All valid entries should move to new table";

    // 4. Verify data in new_table_
    std::vector<KeyT> verify_keys = keys;
    verify_keys.push_back(stash_key1);
    verify_keys.push_back(stash_key2);
    
    std::vector<uint32_t> found_flags(verify_keys.size());
    std::vector<ValueT> found_values(verify_keys.size());
    
    LookupBatch l_batch;
    l_batch.h_keys = verify_keys.data();
    l_batch.h_values = found_values.data();
    l_batch.h_found = found_flags.data();
    l_batch.num_keys = verify_keys.size();

    warp_lookup_batch_sync(*new_table_, nullptr, l_batch);

    uint32_t total_found = 0;
    for (size_t i = 0; i < verify_keys.size(); ++i) {
        if (found_flags[i]) total_found++;
    }
    
    EXPECT_EQ(total_found, verify_keys.size()) << "All keys must be found in the NEW table after rehash";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
