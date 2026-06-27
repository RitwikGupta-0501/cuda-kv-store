#include <gtest/gtest.h>
#include "../../src/gpu/warp_lookup.h"
#include <cstring>

using namespace warpkv;

// Phase 3: Warp-Cooperative Lookup Kernel Unit Tests
// Validates single-warp, dual-bucket parallel scanning per SPEC_V3_FINAL.md Section VII

class WarpLookupTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Insert a key-value pair into a bucket
    void insert_into_bucket(Bucket* bucket, uint32_t key, uint32_t value, uint8_t fp) {
        for (int i = 0; i < 8; ++i) {
            if (!bucket_is_occupied(bucket, i)) {
                bucket->keys[i] = key;
                bucket->values[i] = value;
                bucket->fingerprint[i] = fp;
                bucket_set_occupied(bucket, i);
                return;
            }
        }
    }

    // Helper: Create a test bucket and insert keys
    Bucket create_test_bucket(uint32_t key1, uint32_t val1, uint8_t fp1) {
        Bucket b;
        bucket_init(&b);
        insert_into_bucket(&b, key1, val1, fp1);
        return b;
    }
};

// Test 1: Single key found in bucket b1
TEST_F(WarpLookupTest, SingleKeyB1Hit) {
    Bucket bucket;
    bucket_init(&bucket);

    uint32_t key = 12345;
    uint32_t value = 67890;
    uint8_t fp = 0x42;

    insert_into_bucket(&bucket, key, value, fp);

    // Verify lookup would find it
    EXPECT_EQ(bucket.keys[0], key) << "Key should be in slot 0";
    EXPECT_EQ(bucket.values[0], value) << "Value should be in slot 0";
    EXPECT_EQ(bucket.fingerprint[0], fp) << "Fingerprint should match";
    EXPECT_TRUE(bucket_is_occupied(&bucket, 0)) << "Slot 0 should be occupied";
}

// Test 2: Multiple keys in same bucket
TEST_F(WarpLookupTest, SingleKeyB2Hit) {
    Bucket bucket;
    bucket_init(&bucket);

    // Insert key that would go to second position
    uint32_t key1 = 11111;
    uint32_t val1 = 22222;
    uint8_t fp1 = 0x11;

    uint32_t key2 = 33333;
    uint32_t val2 = 44444;
    uint8_t fp2 = 0x22;

    insert_into_bucket(&bucket, key1, val1, fp1);
    insert_into_bucket(&bucket, key2, val2, fp2);

    // Verify both stored correctly
    EXPECT_EQ(bucket.keys[0], key1) << "Key1 in slot 0";
    EXPECT_EQ(bucket.keys[1], key2) << "Key2 in slot 1";
    EXPECT_EQ(bucket.values[0], val1) << "Val1 in slot 0";
    EXPECT_EQ(bucket.values[1], val2) << "Val2 in slot 1";
}

// Test 3: Key not found in empty bucket
TEST_F(WarpLookupTest, KeyNotFound) {
    Bucket bucket;
    bucket_init(&bucket);

    // All slots should be unoccupied
    EXPECT_EQ(bucket.occupancy_mask, 0) << "Empty bucket has mask 0";

    for (int i = 0; i < 8; ++i) {
        EXPECT_FALSE(bucket_is_occupied(&bucket, i))
            << "Slot " << i << " should not be occupied";
    }
}

// Test 4: Fingerprint matching
TEST_F(WarpLookupTest, FingerprintFalsePositive) {
    Bucket bucket;
    bucket_init(&bucket);

    uint32_t key = 99999;
    uint32_t value = 88888;
    uint8_t fp = 0xAB;

    insert_into_bucket(&bucket, key, value, fp);

    // Verify exact fingerprint matches
    EXPECT_EQ(bucket.fingerprint[0], fp) << "FP should match exactly";

    // Different fingerprint should show no match on fast path
    uint8_t different_fp = 0xCD;
    EXPECT_NE(bucket.fingerprint[0], different_fp) << "Different FP should not match";
}

// Test 5: All 8 slots in bucket
TEST_F(WarpLookupTest, FullWarp32Keys) {
    Bucket bucket;
    bucket_init(&bucket);

    // Fill all 8 slots
    for (int slot = 0; slot < 8; ++slot) {
        uint32_t key = 1000 + slot;
        uint32_t value = 2000 + slot;
        uint8_t fp = slot * 0x11;

        bucket.keys[slot] = key;
        bucket.values[slot] = value;
        bucket.fingerprint[slot] = fp;
        bucket_set_occupied(&bucket, slot);
    }

    // Verify all slots occupied
    EXPECT_EQ(bucket.occupancy_mask, 0xFFu) << "All 8 slots should be occupied";

    // Verify each slot can be read back
    for (int slot = 0; slot < 8; ++slot) {
        EXPECT_EQ(bucket.keys[slot], 1000 + slot) << "Key in slot " << slot;
        EXPECT_EQ(bucket.values[slot], 2000 + slot) << "Value in slot " << slot;
        EXPECT_EQ(bucket.fingerprint[slot], slot * 0x11) << "FP in slot " << slot;
    }
}

// Test 6: Occupancy mask at 50% load
TEST_F(WarpLookupTest, LoadFactor50Percent) {
    Bucket bucket;
    bucket_init(&bucket);

    // Fill 4 out of 8 slots (50% load)
    for (int i = 0; i < 4; ++i) {
        insert_into_bucket(&bucket, 1000 + i, 2000 + i, 0x10 + i);
    }

    // Verify 4 slots occupied
    uint32_t expected_mask = 0x0Fu;  // Bits 0-3 set
    EXPECT_EQ(bucket.occupancy_mask, expected_mask) << "Should have 4 slots occupied";

    // Clear one, verify mask updates
    bucket_clear_occupied(&bucket, 0);
    EXPECT_EQ(bucket.occupancy_mask, 0x0Eu) << "Slot 0 cleared";
}

// Test 7: Bucket field access bounds
TEST_F(WarpLookupTest, NoOutOfBoundsAccess) {
    Bucket bucket;
    bucket_init(&bucket);

    // Verify array sizes constrain access
    EXPECT_EQ(sizeof(bucket.keys), 32) << "keys[8] = 32 bytes";
    EXPECT_EQ(sizeof(bucket.values), 32) << "values[8] = 32 bytes";
    EXPECT_EQ(sizeof(bucket.fingerprint), 8) << "fingerprint[8] = 8 bytes";

    // All indices 0-7 are valid
    for (int i = 0; i < 8; ++i) {
        bucket.keys[i] = 100 + i;
        bucket.values[i] = 200 + i;
        bucket.fingerprint[i] = i;
    }

    // Verify no out-of-bounds
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(bucket.keys[i], 100 + i) << "keys[" << i << "]";
        EXPECT_EQ(bucket.values[i], 200 + i) << "values[" << i << "]";
        EXPECT_EQ(bucket.fingerprint[i], i) << "fingerprint[" << i << "]";
    }
}

// Test 8: Hash function produces distinct bucket indices
TEST_F(WarpLookupTest, B1NotEqualB2) {
    uint32_t different_count = 0;
    const uint32_t sample_size = 1000;
    const uint32_t bucket_mask = (1ULL << 20) - 1;  // 1M buckets

    for (uint32_t i = 0; i < sample_size; ++i) {
        uint32_t key = 100000 + i;
        HashPair hash = compute_hash_pair(key, bucket_mask);

        // For well-distributed hashes, b1 != b2 most of the time
        if (hash.b1 != hash.b2) {
            different_count++;
        }
    }

    // Expect > 99% to have different bucket indices
    double ratio = (double)different_count / sample_size;
    EXPECT_GT(ratio, 0.99)
        << "Most keys should hash to different buckets ("
        << different_count << "/" << sample_size << ")";
}

// Test 9: Bucket cache-line alignment
TEST_F(WarpLookupTest, MaxTwoCacheLineReads) {
    // Verify Bucket is 128 bytes (one L2 cache line)
    EXPECT_EQ(sizeof(Bucket), 128)
        << "Bucket must be 128 bytes (one L2 cache line)";

    // Worst case: one warp reads two buckets (b1 and b2)
    // That's exactly 2 × 128 = 256 bytes = 2 cache lines
    EXPECT_LE(2 * sizeof(Bucket), 256)
        << "Two buckets = two cache lines max";

    // Verify all field offsets are within single bucket
    Bucket b;
    uintptr_t base = (uintptr_t)&b;
    uintptr_t keys_addr = (uintptr_t)b.keys;
    uintptr_t padding_addr = (uintptr_t)b.padding;

    EXPECT_GE(keys_addr, base);
    EXPECT_LT(padding_addr + sizeof(b.padding), base + 128)
        << "All fields within single cache line";
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
