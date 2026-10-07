#include <gtest/gtest.h>
#include "../../src/gpu/warp_lookup.h"
#include "../../src/gpu/bucket_cuckoo.h"
#include "../../src/gpu/cuckoo_insert.h" // For InsertStatus if needed
#include "../../include/warpkv/warpkv_allocator.h"
#include <cstring>

using namespace warpkv;

class WarpLookupTest : public ::testing::Test {
protected:
    static constexpr uint32_t NUM_BUCKETS = 256;
    DefaultCudaAllocator allocator_;
    BucketTable table_storage_;
    BucketTable* table_ = &table_storage_;

    void SetUp() override {
        table_storage_.num_buckets = NUM_BUCKETS;
        table_storage_.bucket_mask = NUM_BUCKETS - 1;
        table_storage_.load_factor_limit = NUM_BUCKETS / 2;
        table_storage_.buckets = static_cast<Bucket*>(
            allocator_.allocate(NUM_BUCKETS * sizeof(Bucket))
        );
        cudaMemset(table_storage_.buckets, 0, NUM_BUCKETS * sizeof(Bucket));
    }

    void TearDown() override {
        if (table_storage_.buckets) {
            allocator_.deallocate(table_storage_.buckets);
            table_storage_.buckets = nullptr;
        }
    }
};

TEST_F(WarpLookupTest, SingleKeyB1Hit) {
    KeyT key = 12345;
    ValueT value = 67890;
    
    HashPair hash = compute_hash_pair(key, table_->num_buckets - 1);
    
    // Manually construct bucket on host
    Bucket h_bucket;
    bucket_init(&h_bucket);
    h_bucket.keys[0] = key;
    h_bucket.values[0] = value;
    h_bucket.fingerprint[0] = hash.fingerprint;
    bucket_set_occupied(&h_bucket, 0);

    // Copy to device at b1
    cudaMemcpy(&table_->buckets[hash.b1], &h_bucket, sizeof(Bucket), cudaMemcpyHostToDevice);

    // Perform real lookup
    KeyT keys_in[1] = {key};
    ValueT values_out[1] = {0};
    uint32_t found_out[1] = {0};

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch_sync(*table_, nullptr, batch);

    EXPECT_EQ(found_out[0], 1) << "Key should be found";
    EXPECT_EQ(values_out[0], value) << "Value should match";
}

TEST_F(WarpLookupTest, SingleKeyB2Hit) {
    KeyT key = 98765;
    ValueT value = 43210;
    
    HashPair hash = compute_hash_pair(key, table_->num_buckets - 1);
    
    Bucket h_bucket;
    bucket_init(&h_bucket);
    h_bucket.keys[3] = key; // Put in slot 3
    h_bucket.values[3] = value;
    h_bucket.fingerprint[3] = hash.fingerprint;
    bucket_set_occupied(&h_bucket, 3);

    // Copy to device at b2 (so it misses b1 and hits b2)
    cudaMemcpy(&table_->buckets[hash.b2], &h_bucket, sizeof(Bucket), cudaMemcpyHostToDevice);

    KeyT keys_in[1] = {key};
    ValueT values_out[1] = {0};
    uint32_t found_out[1] = {0};

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch_sync(*table_, nullptr, batch);

    EXPECT_EQ(found_out[0], 1) << "Key should be found in b2";
    EXPECT_EQ(values_out[0], value) << "Value should match";
}

TEST_F(WarpLookupTest, KeyNotFound) {
    KeyT key = 11111;
    
    // Do not insert anything. Table was cleared in SetUp.
    KeyT keys_in[1] = {key};
    ValueT values_out[1] = {999};
    uint32_t found_out[1] = {1}; // Initialize to 1 to ensure kernel sets it to 0

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch_sync(*table_, nullptr, batch);

    EXPECT_EQ(found_out[0], 0) << "Key should NOT be found";
}

TEST_F(WarpLookupTest, FingerprintFalsePositive) {
    KeyT key = 22222;
    KeyT different_key = 33333; // Hashes to same bucket but different key
    
    HashPair hash_diff = compute_hash_pair(different_key, table_->num_buckets - 1);
    HashPair hash_target = compute_hash_pair(key, table_->num_buckets - 1);
    
    Bucket h_bucket;
    bucket_init(&h_bucket);
    h_bucket.keys[0] = different_key; // Wrong key
    h_bucket.values[0] = 55555;
    h_bucket.fingerprint[0] = hash_target.fingerprint; // Forcing a fingerprint collision!
    bucket_set_occupied(&h_bucket, 0);

    cudaMemcpy(&table_->buckets[hash_target.b1], &h_bucket, sizeof(Bucket), cudaMemcpyHostToDevice);

    KeyT keys_in[1] = {key};
    ValueT values_out[1] = {0};
    uint32_t found_out[1] = {1}; 

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch_sync(*table_, nullptr, batch);

    // Since the key is different, even if fingerprint matched, it should double-check the key and return NOT FOUND
    EXPECT_EQ(found_out[0], 0) << "Key should NOT be found despite fingerprint collision";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
