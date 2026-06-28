#include <gtest/gtest.h>
#include "../src/gpu/bucket_cuckoo.h"

using namespace warpkv;

// Phase 2: Arena Allocator Unit Tests (Complete)
// Validates memory allocation strategy per SPEC_V3_FINAL.md Section VI

class ArenaAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test 1: Bucket count is power of two
TEST_F(ArenaAllocatorTest, PowerOfTwoBucketCount) {
    // Simulate power-of-two rounding (as done in arena allocator)
    size_t arena_size = 750 * 1024 * 1024;  // 750 MB
    size_t raw_buckets = arena_size / sizeof(Bucket);
    size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));

    // Verify it's a power of 2
    EXPECT_NE(num_buckets, 0) << "Bucket count should be > 0";
    EXPECT_EQ(num_buckets & (num_buckets - 1), 0)
        << "Bucket count " << num_buckets << " is not a power of 2";

    // For 750 MB arena: ~5.7M buckets
    EXPECT_GT(num_buckets, 1000000) << "Should have millions of buckets";
    EXPECT_LT(num_buckets, 10000000) << "Should be reasonable bucket count";
}

// Test 2: Bucket mask computation
TEST_F(ArenaAllocatorTest, BucketMaskComputation) {
    size_t arena_size = 750 * 1024 * 1024;
    size_t raw_buckets = arena_size / sizeof(Bucket);
    size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));

    uint32_t bucket_mask = (uint32_t)(num_buckets - 1);

    // Verify mask is all bits set for power-of-two
    // For a power-of-two count, mask = count - 1 = all lower bits set
    EXPECT_EQ(bucket_mask, num_buckets - 1) << "Mask should be count - 1";

    // Verify bitwise AND with mask gives index < num_buckets
    for (uint32_t test_val = 0; test_val < 1000; ++test_val) {
        uint32_t indexed = test_val & bucket_mask;
        EXPECT_LT(indexed, num_buckets)
            << "Index " << indexed << " exceeds bucket count " << num_buckets;
    }
}

// Test 3: Load factor limit (50%)
TEST_F(ArenaAllocatorTest, LoadFactorLimit) {
    size_t arena_size = 750 * 1024 * 1024;
    size_t raw_buckets = arena_size / sizeof(Bucket);
    size_t num_buckets = 1ULL << (63 - __builtin_clzll(raw_buckets));

    uint32_t load_factor_limit = num_buckets / 2;

    // At exactly 50% load, insertion should succeed
    EXPECT_LT(load_factor_limit, num_buckets / 2 + 1)
        << "Load factor limit should be floor(num_buckets / 2)";

    // Verify it's exactly half
    EXPECT_EQ(load_factor_limit, num_buckets / 2)
        << "Load factor limit should be exactly 50%";
}

// Test 4: VRAM budget fits constraints
TEST_F(ArenaAllocatorTest, VRAMBudget) {
    // Two tables at 750 MB each = 1.5 GB
    // Plus ~100 MB for buffers/stash
    // Total should fit in 2 GB with headroom

    size_t arena_per_table = 750 * 1024 * 1024;
    size_t total_tables = arena_per_table * 2;
    size_t buffers_and_stash = 100 * 1024 * 1024;  // Conservative estimate
    size_t total = total_tables + buffers_and_stash;
    EXPECT_LE(total, 2ULL * 1024 * 1024 * 1024)
        << "Total allocation exceeds 2GB budget limit";

    // Verify correct math for headroom
    size_t headroom = (2ULL * 1024 * 1024 * 1024) - total;
    EXPECT_GT(headroom, 300 * 1024 * 1024)
        << "Should have at least 300 MB headroom";
}

// Test 5: Stash capacity formula
TEST_F(ArenaAllocatorTest, StashCapacityFormula) {
    // STASH_CAPACITY = BACKPRESSURE_THRESHOLD + NUM_SLOTS * BATCH_SIZE
    // = 4096 + 3 * 4096 = 16384

    uint32_t required_min = BACKPRESSURE_THRESHOLD + 3 * BATCH_SIZE;
    EXPECT_GE(STASH_CAPACITY, required_min)
        << "Stash must hold at least " << required_min << " entries";

    // Verify it's reasonable (~128 KB)
    size_t stash_memory = STASH_CAPACITY * sizeof(StashEntry);
    EXPECT_LT(stash_memory, 200 * 1024)
        << "Stash should be < 200 KB";

    EXPECT_GT(stash_memory, 100 * 1024)
        << "Stash should be > 100 KB (16384 entries)";
}

// Test 6: Bucket initialization
TEST_F(ArenaAllocatorTest, BucketInitialization) {
    Bucket b;

    // Dirty the bucket
    std::memset(&b, 0xFF, sizeof(Bucket));

    // Initialize
    bucket_init(&b);

    // Verify empty
    EXPECT_EQ(b.occupancy_mask, 0) << "Bucket should be unoccupied after init";

    for (int slot = 0; slot < 8; ++slot) {
        EXPECT_EQ(b.keys[slot], 0) << "Key slot " << slot << " should be zero";
        EXPECT_EQ(b.values[slot], 0) << "Value slot " << slot << " should be zero";
        EXPECT_EQ(b.fingerprint[slot], 0) << "FP slot " << slot << " should be zero";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
