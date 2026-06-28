#include <gtest/gtest.h>
#include "../src/gpu/bucket_cuckoo.h"
#include <cstddef>
#include <cstring>

using namespace warpkv;

// Phase 2: Bucket Layout Unit Tests (Complete)
// Validates cache-line alignment and bit operations per SPEC_V3_FINAL.md Section VI

class BucketLayoutTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test 1: Bucket struct is exactly 128 bytes
TEST_F(BucketLayoutTest, BucketSize) {
    EXPECT_EQ(sizeof(Bucket), 128)
        << "Bucket must be exactly 128 bytes (one L2 cache line)";

    // Verify internal field sizes
    EXPECT_EQ(sizeof(Bucket::keys), 32) << "keys[8] should be 32 bytes";
    EXPECT_EQ(sizeof(Bucket::values), 32) << "values[8] should be 32 bytes";
    EXPECT_EQ(sizeof(Bucket::fingerprint), 8) << "fingerprint[8] should be 8 bytes";
    EXPECT_EQ(sizeof(Bucket::occupancy_mask), 4) << "occupancy_mask should be 4 bytes";
    EXPECT_EQ(sizeof(Bucket::padding), 52) << "padding should be 52 bytes";
}

// Test 2: Bucket layout (field offsets)
TEST_F(BucketLayoutTest, BucketFieldOffsets) {


    // Verify field offsets (for memory layout verification)
    EXPECT_EQ(offsetof(Bucket, keys), 0) << "keys array should be at offset 0";
    EXPECT_EQ(offsetof(Bucket, values), 32) << "values array should be at offset 32";
    EXPECT_EQ(offsetof(Bucket, fingerprint), 64) << "fingerprint should start at offset 64";
    EXPECT_EQ(offsetof(Bucket, occupancy_mask), 72) << "occupancy_mask should start at offset 72";
    EXPECT_EQ(offsetof(Bucket, padding), 76) << "padding should start at offset 76";
}

// Test 3: Occupancy mask bit operations
TEST_F(BucketLayoutTest, OccupancyMaskBitOps) {
    Bucket b;
    bucket_init(&b);

    // Initially empty
    EXPECT_EQ(b.occupancy_mask, 0) << "Newly initialized bucket should be empty";

    // Test set_occupied for each slot
    for (int slot = 0; slot < 8; ++slot) {
        bucket_init(&b);
        EXPECT_FALSE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should not be occupied initially";

        bucket_set_occupied(&b, slot);

        EXPECT_TRUE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should be occupied after set";

        // Verify correct bit is set
        uint32_t expected_mask = (1u << slot);
        EXPECT_EQ(b.occupancy_mask, expected_mask)
            << "Only slot " << slot << " should be occupied";
    }

    // Test clear_occupied
    bucket_init(&b);
    for (int slot = 0; slot < 8; ++slot) {
        bucket_set_occupied(&b, slot);
    }
    for (int slot = 0; slot < 8; ++slot) {
        bucket_clear_occupied(&b, slot);

        EXPECT_FALSE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should not be occupied after clear";
    }

    EXPECT_EQ(b.occupancy_mask, 0) << "All slots should be empty after clearing";
}

// Test 4: All 8 slots can be set simultaneously
TEST_F(BucketLayoutTest, AllSlotsFull) {
    Bucket b;
    bucket_init(&b);

    // Set all 8 slots
    for (int slot = 0; slot < 8; ++slot) {
        bucket_set_occupied(&b, slot);
    }

    // Verify all are occupied
    for (int slot = 0; slot < 8; ++slot) {
        EXPECT_TRUE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should be occupied";
    }

    // Verify occupancy_mask is 0xFF (all 8 bits set)
    EXPECT_EQ(b.occupancy_mask, 0xFFu) << "All 8 slots should be marked occupied (0xFF)";
}

// Test 5: Fingerprint storage and retrieval
TEST_F(BucketLayoutTest, FingerprintStorage) {
    Bucket b;
    bucket_init(&b);

    // Store fingerprints in all slots
    for (int slot = 0; slot < 8; ++slot) {
        uint8_t fp = (uint8_t)((slot + 1) * 31);  // Arbitrary pattern
        b.fingerprint[slot] = fp;
    }

    // Verify fingerprints are stored and retrieved correctly
    for (int slot = 0; slot < 8; ++slot) {
        uint8_t expected_fp = (uint8_t)((slot + 1) * 31);
        EXPECT_EQ(b.fingerprint[slot], expected_fp)
            << "Fingerprint in slot " << slot << " doesn't match";
    }
}

// Test 6: Key/value storage
TEST_F(BucketLayoutTest, KeyValueStorage) {
    Bucket b;
    bucket_init(&b);

    // Store keys and values
    for (int slot = 0; slot < 8; ++slot) {
        b.keys[slot] = 1000 + slot;
        b.values[slot] = 2000 + slot;
        bucket_set_occupied(&b, slot);
    }

    // Verify keys and values
    for (int slot = 0; slot < 8; ++slot) {
        EXPECT_EQ(b.keys[slot], 1000 + slot) << "Key in slot " << slot;
        EXPECT_EQ(b.values[slot], 2000 + slot) << "Value in slot " << slot;
        EXPECT_TRUE(bucket_is_occupied(&b, slot)) << "Slot " << slot << " should be occupied";
    }
}

// Test 7: Bucket initialization clears data
TEST_F(BucketLayoutTest, BucketInitialization) {
    Bucket b;

    // Dirty the bucket with garbage
    std::memset(&b, 0xFF, sizeof(Bucket));

    // Initialize
    bucket_init(&b);

    // Verify it's empty
    EXPECT_EQ(b.occupancy_mask, 0) << "occupancy_mask should be cleared";

    // Verify keys and values are zero
    for (int slot = 0; slot < 8; ++slot) {
        EXPECT_EQ(b.keys[slot], 0) << "Key in slot " << slot << " should be zero";
        EXPECT_EQ(b.values[slot], 0) << "Value in slot " << slot << " should be zero";
        EXPECT_EQ(b.fingerprint[slot], 0) << "Fingerprint in slot " << slot << " should be zero";
    }
}

// Test 8: StashQueue structure size and capacity
TEST_F(BucketLayoutTest, StashQueueStructure) {
    EXPECT_EQ(STASH_CAPACITY, 32768) << "Stash capacity should be 32768";
    EXPECT_EQ(sizeof(StashQueue::entries) / sizeof(StashEntry), STASH_CAPACITY)
        << "StashQueue should hold exactly STASH_CAPACITY entries";

    // Verify stash fits the formula: BACKPRESSURE_THRESHOLD + NUM_SLOTS * BATCH_SIZE
    EXPECT_GE(STASH_CAPACITY, BACKPRESSURE_THRESHOLD + 3 * BATCH_SIZE)
        << "Stash must hold at least BACKPRESSURE_THRESHOLD + NUM_SLOTS * BATCH_SIZE";
}

// Test 9: Bucket constants are correct
TEST_F(BucketLayoutTest, Constants) {
    EXPECT_EQ(warpkv::MAX_EVICTION_HOPS, 128) << "Max eviction hops should be 128";
    EXPECT_EQ(BACKPRESSURE_THRESHOLD, 4096) << "Backpressure threshold should be 4096";
    EXPECT_EQ(BATCH_SIZE, 4096) << "Batch size should be 4096";
    EXPECT_EQ(STASH_CAPACITY, 32768) << "Stash capacity should be 32768";
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
