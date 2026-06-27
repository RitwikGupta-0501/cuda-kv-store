#include <gtest/gtest.h>
#include "../../src/gpu/warp_lookup.h"
#include "../../src/gpu/xxhash3.h"
#include "../../src/gpu/bucket_cuckoo.h"
#include "../../src/gpu/cuckoo_insert.h" // For InsertStatus if needed
#include <cstring>

namespace warpkv {
    void init_arena();
    BucketTable* get_table0();
}

using namespace warpkv;

class WarpLookupTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            init_arena();
        } catch (...) {}
        table_ = get_table0();
        // Clear table
        cudaMemset(table_->buckets, 0, table_->num_buckets * sizeof(Bucket));
    }

    BucketTable* table_;
};

TEST_F(WarpLookupTest, SingleKeyB1Hit) {
    uint32_t key = 12345;
    uint32_t value = 67890;
    
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
    uint32_t keys_in[1] = {key};
    uint32_t values_out[1] = {0};
    uint32_t found_out[1] = {0};

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch(*table_, batch);

    EXPECT_EQ(found_out[0], 1) << "Key should be found";
    EXPECT_EQ(values_out[0], value) << "Value should match";
}

TEST_F(WarpLookupTest, SingleKeyB2Hit) {
    uint32_t key = 98765;
    uint32_t value = 43210;
    
    HashPair hash = compute_hash_pair(key, table_->num_buckets - 1);
    
    Bucket h_bucket;
    bucket_init(&h_bucket);
    h_bucket.keys[3] = key; // Put in slot 3
    h_bucket.values[3] = value;
    h_bucket.fingerprint[3] = hash.fingerprint;
    bucket_set_occupied(&h_bucket, 3);

    // Copy to device at b2 (so it misses b1 and hits b2)
    cudaMemcpy(&table_->buckets[hash.b2], &h_bucket, sizeof(Bucket), cudaMemcpyHostToDevice);

    uint32_t keys_in[1] = {key};
    uint32_t values_out[1] = {0};
    uint32_t found_out[1] = {0};

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch(*table_, batch);

    EXPECT_EQ(found_out[0], 1) << "Key should be found in b2";
    EXPECT_EQ(values_out[0], value) << "Value should match";
}

TEST_F(WarpLookupTest, KeyNotFound) {
    uint32_t key = 11111;
    
    // Do not insert anything. Table was cleared in SetUp.
    uint32_t keys_in[1] = {key};
    uint32_t values_out[1] = {999};
    uint32_t found_out[1] = {1}; // Initialize to 1 to ensure kernel sets it to 0

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch(*table_, batch);

    EXPECT_EQ(found_out[0], 0) << "Key should NOT be found";
}

TEST_F(WarpLookupTest, FingerprintFalsePositive) {
    uint32_t key = 22222;
    uint32_t different_key = 33333; // Hashes to same bucket but different key
    
    HashPair hash_diff = compute_hash_pair(different_key, table_->num_buckets - 1);
    HashPair hash_target = compute_hash_pair(key, table_->num_buckets - 1);
    
    Bucket h_bucket;
    bucket_init(&h_bucket);
    h_bucket.keys[0] = different_key; // Wrong key
    h_bucket.values[0] = 55555;
    h_bucket.fingerprint[0] = hash_target.fingerprint; // Forcing a fingerprint collision!
    bucket_set_occupied(&h_bucket, 0);

    cudaMemcpy(&table_->buckets[hash_target.b1], &h_bucket, sizeof(Bucket), cudaMemcpyHostToDevice);

    uint32_t keys_in[1] = {key};
    uint32_t values_out[1] = {0};
    uint32_t found_out[1] = {1}; 

    LookupBatch batch;
    batch.h_keys = keys_in;
    batch.h_values = values_out;
    batch.h_found = found_out;
    batch.num_keys = 1;

    warp_lookup_batch(d_table_, batch);

    // Since the key is different, even if fingerprint matched, it should double-check the key and return NOT FOUND
    EXPECT_EQ(found_out[0], 0) << "Key should NOT be found despite fingerprint collision";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
