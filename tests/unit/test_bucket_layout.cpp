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

    // Verify internal field sizes (7 slots with 64-bit keys/values)
    EXPECT_EQ(sizeof(Bucket::keys), BUCKET_SLOTS * sizeof(KeyT))
        << "keys[7] should be 56 bytes";
    EXPECT_EQ(sizeof(Bucket::values), BUCKET_SLOTS * sizeof(ValueT))
        << "values[7] should be 56 bytes";
    EXPECT_EQ(sizeof(Bucket::occupancy_mask), sizeof(uint32_t))
        << "occupancy_mask should be 4 bytes";
    EXPECT_EQ(sizeof(Bucket::fingerprint), BUCKET_SLOTS * sizeof(uint8_t))
        << "fingerprint[7] should be 7 bytes";
    EXPECT_EQ(sizeof(Bucket::_pad), 5)
        << "_pad should be 5 bytes";
}

// Test 2: Bucket layout (field offsets)
TEST_F(BucketLayoutTest, BucketFieldOffsets) {
    // Verify field offsets (for memory layout verification)
    EXPECT_EQ(offsetof(Bucket, keys), 0)
        << "keys array should be at offset 0";
    EXPECT_EQ(offsetof(Bucket, values), 56)
        << "values array should be at offset 56";
    EXPECT_EQ(offsetof(Bucket, occupancy_mask), 112)
        << "occupancy_mask should start at offset 112";
    EXPECT_EQ(offsetof(Bucket, fingerprint), 116)
        << "fingerprint should start at offset 116";
    EXPECT_EQ(offsetof(Bucket, _pad), 123)
        << "_pad should start at offset 123";
}

// Test 3: Occupancy mask bit operations
TEST_F(BucketLayoutTest, OccupancyMaskBitOps) {
    Bucket b;
    bucket_init(&b);

    // Initially empty
    EXPECT_EQ(b.occupancy_mask, 0) << "Newly initialized bucket should be empty";

    // Test set_occupied for each slot
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
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
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        bucket_set_occupied(&b, slot);
    }
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        bucket_clear_occupied(&b, slot);

        EXPECT_FALSE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should not be occupied after clear";
    }

    EXPECT_EQ(b.occupancy_mask, 0) << "All slots should be empty after clearing";
}

// Test 4: All slots can be set simultaneously
TEST_F(BucketLayoutTest, AllSlotsFull) {
    Bucket b;
    bucket_init(&b);

    // Set all slots
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        bucket_set_occupied(&b, slot);
    }

    // Verify all are occupied
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        EXPECT_TRUE(bucket_is_occupied(&b, slot))
            << "Slot " << slot << " should be occupied";
    }

    // Verify occupancy_mask has all BUCKET_SLOTS bits set
    const uint32_t expected_full_mask = (1u << BUCKET_SLOTS) - 1;
    EXPECT_EQ(b.occupancy_mask, expected_full_mask)
        << "All slots should be marked occupied (" << expected_full_mask << ")";
}

// Test 5: Fingerprint storage and retrieval
TEST_F(BucketLayoutTest, FingerprintStorage) {
    Bucket b;
    bucket_init(&b);

    // Store fingerprints in all slots
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        uint8_t fp = (uint8_t)((slot + 1) * 31);  // Arbitrary pattern
        b.fingerprint[slot] = fp;
    }

    // Verify fingerprints are stored and retrieved correctly
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
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
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        b.keys[slot] = 1000ULL + slot;
        b.values[slot] = 2000ULL + slot;
        bucket_set_occupied(&b, slot);
    }

    // Verify keys and values
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        EXPECT_EQ(b.keys[slot], 1000ULL + slot) << "Key in slot " << slot;
        EXPECT_EQ(b.values[slot], 2000ULL + slot) << "Value in slot " << slot;
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
    for (uint32_t slot = 0; slot < BUCKET_SLOTS; ++slot) {
        EXPECT_EQ(b.keys[slot], 0ULL) << "Key in slot " << slot << " should be zero";
        EXPECT_EQ(b.values[slot], 0ULL) << "Value in slot " << slot << " should be zero";
        EXPECT_EQ(b.fingerprint[slot], 0) << "Fingerprint in slot " << slot << " should be zero";
    }
}

// Test 8: StashQueue structure size and capacity
TEST_F(BucketLayoutTest, StashQueueStructure) {
    EXPECT_EQ(STASH_CAPACITY, 32768) << "Stash capacity should be 32768";
    EXPECT_EQ(sizeof(StashQueue::entries) / sizeof(StashEntry), STASH_CAPACITY)
        << "StashQueue should hold exactly STASH_CAPACITY entries";

    // Verify StashQueue size fits 600 KB limit for 64-bit entries (32768 * 16 + 8 bytes = ~512 KB)
    EXPECT_LT(sizeof(StashQueue), 600000)
        << "StashQueue must be < 600 KB";
    EXPECT_EQ(offsetof(StashQueue, head), 0)
        << "head should be at offset 0";

    // Verify stash fits the formula: BACKPRESSURE_THRESHOLD + NUM_SLOTS * BATCH_SIZE
    EXPECT_GE(STASH_CAPACITY, BACKPRESSURE_THRESHOLD + 3 * BATCH_SIZE)
        << "Stash must hold at least BACKPRESSURE_THRESHOLD + NUM_SLOTS * BATCH_SIZE";
}

// Test 9: Bucket constants are correct
TEST_F(BucketLayoutTest, Constants) {
    EXPECT_EQ(BUCKET_SLOTS, 7) << "Bucket slots should be 7";
    EXPECT_EQ(EMPTY_KEY, 0x0000000000000000ULL) << "EMPTY_KEY should be 0ULL";
    EXPECT_EQ(LOCK_SENTINEL, 0xFFFFFFFFFFFFFFFFULL) << "LOCK_SENTINEL should be ~0ULL";
    EXPECT_EQ(NOT_FOUND, 0xFFFFFFFFFFFFFFFFULL) << "NOT_FOUND should be ~0ULL";
    EXPECT_EQ(warpkv::MAX_EVICTION_HOPS, 128) << "Max eviction hops should be 128";
    EXPECT_EQ(BACKPRESSURE_THRESHOLD, 4096) << "Backpressure threshold should be 4096";
    EXPECT_EQ(BATCH_SIZE, 4096) << "Batch size should be 4096";
    EXPECT_EQ(STASH_CAPACITY, 32768) << "Stash capacity should be 32768";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
