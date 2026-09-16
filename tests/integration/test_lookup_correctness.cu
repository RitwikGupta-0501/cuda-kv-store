#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "../../src/engine/warpkv_engine.h"
#include "../../src/gpu/bucket_cuckoo.h"

using namespace warpkv;

class LookupCorrectnessTest : public ::testing::Test {
protected:
    WarpKVEngine engine;

    void SetUp() override {
        // Small number of buckets to trigger contention & rehashing naturally
        engine.init(65536); 
    }

    void TearDown() override {
    }
};

TEST_F(LookupCorrectnessTest, InsertLookupDeleteFullCycle) {
    const uint32_t NUM_KEYS = 10000;
    std::vector<uint32_t> keys(NUM_KEYS);
    std::vector<uint32_t> values(NUM_KEYS);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, 0xFFFFFFFE); // Avoid 0

    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        keys[i] = dist(rng);
        values[i] = dist(rng);
    }

    // 1. Insert Batch
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, NUM_KEYS - offset);
        engine.submit_insert_batch(&keys[offset], &values[offset], current_batch);
    }
    engine.sync_all();

    // 2. Lookup Batch and Verify
    uint32_t found_count = 0;
    uint32_t mismatches = 0;
    
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, NUM_KEYS - offset);
        std::vector<uint32_t> out_values(current_batch);
        
        engine.submit_lookup_batch(&keys[offset], out_values.data(), current_batch);
        engine.sync_all();
        
        for (uint32_t i = 0; i < current_batch; ++i) {
            if (out_values[i] != NOT_FOUND) {
                found_count++;
                if (out_values[i] != values[offset + i]) mismatches++;
            }
        }
    }
    
    EXPECT_EQ(found_count, NUM_KEYS);
    EXPECT_EQ(mismatches, 0);

    // 3. Delete Half the Keys
    uint32_t delete_count = NUM_KEYS / 2;
    for (uint32_t offset = 0; offset < delete_count; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, delete_count - offset);
        engine.submit_delete_batch(&keys[offset], current_batch);
    }
    engine.sync_all();
    
    // 4. Verify Deletion
    found_count = 0;
    for (uint32_t offset = 0; offset < delete_count; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, delete_count - offset);
        std::vector<uint32_t> out_values(current_batch);
        engine.submit_lookup_batch(&keys[offset], out_values.data(), current_batch);
        engine.sync_all();
        for (uint32_t i = 0; i < current_batch; ++i) {
            if (out_values[i] != NOT_FOUND) found_count++;
        }
    }
    EXPECT_EQ(found_count, 0); // Deleted keys should be gone
    
    // 5. Verify Remaining Keys
    found_count = 0;
    for (uint32_t offset = delete_count; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, NUM_KEYS - offset);
        std::vector<uint32_t> out_values(current_batch);
        engine.submit_lookup_batch(&keys[offset], out_values.data(), current_batch);
        engine.sync_all();
        for (uint32_t i = 0; i < current_batch; ++i) {
            if (out_values[i] != NOT_FOUND) found_count++;
        }
    }
    EXPECT_EQ(found_count, NUM_KEYS - delete_count); // Second half should still exist
}

TEST_F(LookupCorrectnessTest, LookupNonExistentKeys) {
    const uint32_t NUM_KEYS = 5000;
    std::vector<uint32_t> missing_keys(NUM_KEYS);
    
    std::mt19937 rng(1337);
    std::uniform_int_distribution<uint32_t> dist(1, 0xFFFFFFFE);
    
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        missing_keys[i] = dist(rng);
    }

    uint32_t false_positives = 0;
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch = std::min((uint32_t)BATCH_SIZE, NUM_KEYS - offset);
        std::vector<uint32_t> out_values(current_batch);
        
        engine.submit_lookup_batch(&missing_keys[offset], out_values.data(), current_batch);
        engine.sync_all();
        
        for (uint32_t i = 0; i < current_batch; ++i) {
            if (out_values[i] != NOT_FOUND) false_positives++;
        }
    }
    
    EXPECT_EQ(false_positives, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
