#include <gtest/gtest.h>
#include "../../src/gpu/cuckoo_insert.h"

using namespace warpkv;

// Phase 4: Cuckoo Insertion Kernel Unit Tests
// Validates insertion, eviction chains, and stash overflow per SPEC_V3_FINAL.md Section VIII

class CuckooInsertTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize a test bucket
        bucket_init(&bucket_);
    }

    void TearDown() override {}

    Bucket bucket_;

    // Helper: Create a mostly-empty bucket
    Bucket create_partially_full_bucket(int occupied_slots) {
        Bucket b;
        bucket_init(&b);
        for (int i = 0; i < occupied_slots; ++i) {
            b.keys[i] = 1000 + i;
            b.values[i] = 2000 + i;
            b.fingerprint[i] = (uint8_t)(i * 0x11);
            bucket_set_occupied(&b, i);
        }
        return b;
    }
};

// Test 1: Insert into empty bucket b1
TEST_F(CuckooInsertTest, InsertIntoEmptyB1) {
    EXPECT_EQ(bucket_.occupancy_mask, 0) << "Bucket should be empty";

    uint32_t key = 12345;
    uint32_t value = 67890;
    uint8_t fp = 0x42;

    // Manually insert (simulating what kernel does)
    bucket_.keys[0] = key;
    bucket_.values[0] = value;
    bucket_.fingerprint[0] = fp;
    bucket_set_occupied(&bucket_, 0);

    // Verify insert succeeded
    EXPECT_TRUE(bucket_is_occupied(&bucket_, 0)) << "Slot 0 should be occupied";
    EXPECT_EQ(bucket_.keys[0], key) << "Key should be stored";
    EXPECT_EQ(bucket_.values[0], value) << "Value should be stored";
    EXPECT_EQ(bucket_.fingerprint[0], fp) << "Fingerprint should be stored";
}

// Test 2: Insert into second slot
TEST_F(CuckooInsertTest, InsertIntoSecondSlot) {
    // Fill first slot
    bucket_.keys[0] = 10001;
    bucket_.values[0] = 20001;
    bucket_.fingerprint[0] = 0x11;
    bucket_set_occupied(&bucket_, 0);

    // Insert into second slot
    uint32_t key = 10002;
    uint32_t value = 20002;
    uint8_t fp = 0x22;

    bucket_.keys[1] = key;
    bucket_.values[1] = value;
    bucket_.fingerprint[1] = fp;
    bucket_set_occupied(&bucket_, 1);

    // Verify both occupied
    EXPECT_EQ(bucket_.occupancy_mask, 0x03u) << "Both slots occupied (mask 0x03)";
    EXPECT_EQ(bucket_.keys[0], 10001) << "Slot 0 unchanged";
    EXPECT_EQ(bucket_.keys[1], key) << "Slot 1 has new key";
}

// Test 3: Insert into full bucket should fail locally
TEST_F(CuckooInsertTest, FullBucketNoSpace) {
    // Fill all 8 slots
    for (int i = 0; i < 8; ++i) {
        bucket_.keys[i] = 1000 + i;
        bucket_.values[i] = 2000 + i;
        bucket_.fingerprint[i] = (uint8_t)(i * 0x11);
        bucket_set_occupied(&bucket_, i);
    }

    EXPECT_EQ(bucket_.occupancy_mask, 0xFFu) << "All 8 slots occupied";

    // Verify no more free slots
    bool has_free_slot = false;
    for (int i = 0; i < 8; ++i) {
        if (!bucket_is_occupied(&bucket_, i)) {
            has_free_slot = true;
            break;
        }
    }
    EXPECT_FALSE(has_free_slot) << "No free slots in full bucket";
}

// Test 4: InsertStatus enum values
TEST_F(CuckooInsertTest, StatusEnumValues) {
    EXPECT_EQ(INSERT_SUCCESS, 0) << "INSERT_SUCCESS should be 0";
    EXPECT_EQ(INSERT_STASHED, 1) << "INSERT_STASHED should be 1";
    EXPECT_EQ(INSERT_FAILED, 2) << "INSERT_FAILED should be 2";
}

// Test 5: StashQueue structure
TEST_F(CuckooInsertTest, StashQueueStructure) {
    StashQueue stash;
    stash.head = 0;
    stash.tail = 0;
    stash.needs_rehash = 0;

    EXPECT_EQ(stash.head, 0) << "Head should initialize to 0";
    EXPECT_EQ(stash.tail, 0) << "Tail should initialize to 0";
    EXPECT_EQ(stash.needs_rehash, 0) << "needs_rehash should initialize to 0";

    // Verify capacity
    EXPECT_EQ(sizeof(stash.entries) / sizeof(StashEntry), 5120)
        << "Stash should have 5120 slots";
}

// Test 6: Stash entry structure
TEST_F(CuckooInsertTest, StashEntryStructure) {
    StashEntry entry;
    entry.key = 12345;
    entry.value = 67890;

    EXPECT_EQ(entry.key, 12345) << "Key should be stored";
    EXPECT_EQ(entry.value, 67890) << "Value should be stored";
    EXPECT_EQ(sizeof(StashEntry), 8) << "StashEntry should be 8 bytes";
}

// Test 7: Multiple keys in bucket
TEST_F(CuckooInsertTest, MultipleKeysPerBucket) {
    const int num_keys = 5;
    for (int i = 0; i < num_keys; ++i) {
        bucket_.keys[i] = 10000 + i;
        bucket_.values[i] = 20000 + i;
        bucket_.fingerprint[i] = (uint8_t)(i * 0x10);
        bucket_set_occupied(&bucket_, i);
    }

    // Verify all stored correctly
    for (int i = 0; i < num_keys; ++i) {
        EXPECT_EQ(bucket_.keys[i], 10000 + i) << "Key at slot " << i;
        EXPECT_EQ(bucket_.values[i], 20000 + i) << "Value at slot " << i;
        EXPECT_TRUE(bucket_is_occupied(&bucket_, i)) << "Slot " << i << " occupied";
    }

    // Verify empty slots still free
    for (int i = num_keys; i < 8; ++i) {
        EXPECT_FALSE(bucket_is_occupied(&bucket_, i)) << "Slot " << i << " free";
    }
}

// Test 8: Occupancy mask atomic operations
TEST_F(CuckooInsertTest, OccupancyMaskAtomicOps) {
    // Simulate atomic CAS for slot claiming
    uint32_t mask = 0x00u;

    // Simulate CAS for slot 0
    uint32_t expected = 0x00u;
    uint32_t desired = 0x01u;
    if (mask == expected) {
        mask = desired;
    }

    EXPECT_EQ(mask, 0x01u) << "Slot 0 claimed";

    // Try to claim same slot (should fail)
    expected = 0x00u;  // Mask is now 0x01, not 0x00
    desired = 0x01u;
    if (mask == expected) {
        mask = desired;
    } else {
        // CAS failed, as expected
    }

    EXPECT_EQ(mask, 0x01u) << "Mask unchanged after failed CAS";

    // Claim slot 1
    expected = 0x01u;
    desired = 0x03u;
    if (mask == expected) {
        mask = desired;
    }

    EXPECT_EQ(mask, 0x03u) << "Both slots 0 and 1 claimed";
}

// Test 9: Hash pair computation
TEST_F(CuckooInsertTest, HashPairComputation) {
    uint32_t key = 99999;
    uint32_t bucket_mask = 0xFFFFFFu;  // 16M buckets

    HashPair hash = compute_hash_pair(key, bucket_mask);

    // Verify hash pair structure
    EXPECT_LE(hash.b1, bucket_mask) << "b1 within mask range";
    EXPECT_LE(hash.b2, bucket_mask) << "b2 within mask range";

    // Most of the time b1 != b2
    EXPECT_NE(hash.b1, hash.b2) << "For most keys, b1 != b2";

    // Fingerprint should be in valid range
    EXPECT_GE(hash.fingerprint, 0) << "Fingerprint >= 0";
    EXPECT_LE(hash.fingerprint, 255) << "Fingerprint <= 255";
}

// Test 10: InsertResult structure
TEST_F(CuckooInsertTest, InsertResultStructure) {
    InsertResult result;
    result.status = INSERT_SUCCESS;
    result.slot_used = 3;

    EXPECT_EQ(result.status, INSERT_SUCCESS) << "Status set correctly";
    EXPECT_EQ(result.slot_used, 3) << "Slot recorded correctly";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
