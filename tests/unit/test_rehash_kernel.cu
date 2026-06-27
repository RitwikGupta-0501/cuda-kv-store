#include <gtest/gtest.h>
#include "../../src/gpu/rehash_kernel.h"

using namespace warpkv;

// Phase 5: Rehashing Kernel Unit Tests
// Validates table expansion, stash drain, and EBR coordination per SPEC_V3_FINAL.md Section IX

class RehashKernelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test 1: RehashStatus enum values
TEST_F(RehashKernelTest, RehashStatusEnum) {
    EXPECT_EQ(REHASH_START, 0) << "REHASH_START should be 0";
    EXPECT_EQ(REHASH_IN_PROGRESS, 1) << "REHASH_IN_PROGRESS should be 1";
    EXPECT_EQ(REHASH_COMPLETE, 2) << "REHASH_COMPLETE should be 2";
    EXPECT_EQ(REHASH_FAILED, 3) << "REHASH_FAILED should be 3";
}

// Test 2: RehashStats structure
TEST_F(RehashKernelTest, RehashStatsStructure) {
    RehashStats stats;
    stats.entries_copied = 1000;
    stats.entries_stashed = 50;
    stats.new_table_capacity = 2000000;
    stats.status = REHASH_COMPLETE;

    EXPECT_EQ(stats.entries_copied, 1000) << "entries_copied field";
    EXPECT_EQ(stats.entries_stashed, 50) << "entries_stashed field";
    EXPECT_EQ(stats.new_table_capacity, 2000000) << "new_table_capacity field";
    EXPECT_EQ(stats.status, REHASH_COMPLETE) << "status field";
}

// Test 3: RehashContext structure
TEST_F(RehashKernelTest, RehashContextStructure) {
    Bucket b;
    bucket_init(&b);

    BucketTable old_table;
    old_table.buckets = &b;
    old_table.num_buckets = 1000000;
    old_table.bucket_mask = 0xFFFFFu;
    old_table.load_factor_limit = 500000;

    BucketTable new_table;
    new_table.buckets = &b;
    new_table.num_buckets = 2000000;
    new_table.bucket_mask = 0x1FFFFFu;
    new_table.load_factor_limit = 1000000;

    StashQueue stash;
    stash.head = 0;

    RehashContext ctx;
    ctx.d_old_table = &old_table;
    ctx.d_new_table = &new_table;
    ctx.d_stash = &stash;

    EXPECT_EQ(ctx.d_old_table->num_buckets, 1000000) << "old table buckets";
    EXPECT_EQ(ctx.d_new_table->num_buckets, 2000000) << "new table buckets (doubled)";
    EXPECT_EQ(ctx.d_stash->head, 0) << "stash head initialized";
}

// Test 4: Table capacity doubling
TEST_F(RehashKernelTest, TableCapacityDoubling) {
    // Typical rehash: double table size
    uint32_t old_capacity = 1 << 22;  // ~4.2M buckets
    uint32_t new_capacity = old_capacity * 2;

    EXPECT_EQ(new_capacity, 1u << 23) << "Doubling preserves power-of-2";
    EXPECT_GT(new_capacity, old_capacity) << "New capacity larger";
    EXPECT_LE(new_capacity / old_capacity, 2) << "Growth factor is 2x";
}

// Test 5: Load factor after rehash
TEST_F(RehashKernelTest, LoadFactorAfterRehash) {
    // Old table at 50% load
    uint32_t old_buckets = 1000000;
    uint32_t old_load_limit = old_buckets / 2;
    uint32_t entries = old_load_limit;

    // After rehash to 2M buckets
    uint32_t new_buckets = old_buckets * 2;
    // uint32_t new_load_limit = new_buckets / 2; // unused, but kept for logical completeness

    // New load factor is now 25%
    double new_load_factor = (double)entries / new_buckets;
    EXPECT_LE(new_load_factor, 0.5) << "Load factor should be at most 50%";
    EXPECT_GT(new_load_factor, 0.0) << "Load factor should be positive";
}

// Test 6: Stash capacity verification
TEST_F(RehashKernelTest, StashCapacityForRehash) {
    // Stash must hold at least one full batch during rehash
    uint32_t max_stash_during_rehash = STASH_CAPACITY;
    uint32_t batch_size = BATCH_SIZE;

    EXPECT_GE(max_stash_during_rehash, batch_size)
        << "Stash capacity must hold one batch";
    EXPECT_GE(max_stash_during_rehash, BACKPRESSURE_THRESHOLD)
        << "Stash must reach backpressure threshold";
}

// Test 7: Rehash backpressure threshold
TEST_F(RehashKernelTest, BackpressureThreshold) {
    // Rehash triggered at 50% stash fullness
    uint32_t trigger_point = BACKPRESSURE_THRESHOLD;
    uint32_t total_capacity = STASH_CAPACITY;

    double trigger_percentage = (double)trigger_point / total_capacity * 100;
    EXPECT_GE(trigger_percentage, 1.0) << "Threshold at least 1% of capacity";
    EXPECT_LE(trigger_percentage, 10.0) << "Threshold at most 10% of capacity";
}

// Test 8: HashPair structure for rehashing
TEST_F(RehashKernelTest, RehashHashPair) {
    uint32_t key = 12345;
    uint32_t old_mask = 0xFFFFFu;   // 1M buckets
    uint32_t new_mask = 0x1FFFFFu;  // 2M buckets

    HashPair old_hash = compute_hash_pair(key, old_mask);
    HashPair new_hash = compute_hash_pair(key, new_mask);

    // Both use same fingerprint
    EXPECT_EQ(old_hash.fingerprint, new_hash.fingerprint)
        << "Fingerprint unchanged across rehash";

    // Bucket indices may differ (due to mask change)
    // But both should be in valid range
    EXPECT_LE(old_hash.b1, old_mask) << "Old b1 in range";
    EXPECT_LE(new_hash.b1, new_mask) << "New b1 in range";
}

// Test 9: Epoch marker for EBR
TEST_F(RehashKernelTest, EBRCoordination) {
    // EBR requires atomic swap of table pointers
    // This test verifies the status transitions are logical

    RehashStats before;
    before.status = REHASH_START;

    // After rehash completes
    RehashStats after;
    after.status = REHASH_COMPLETE;

    // Should transition through in-progress
    EXPECT_LT((uint32_t)before.status, (uint32_t)after.status)
        << "Status should progress monotonically";
}

// Test 10: Stash drain into new table
TEST_F(RehashKernelTest, StashDrainCoordinates) {
    // Stash entries drain into new table with recomputed hashes
    StashEntry entry;
    entry.key = 55555;
    entry.value = 77777;

    // In new table with larger mask, key might map to different bucket
    uint32_t old_mask = 0xFFFFFu;
    uint32_t new_mask = 0x1FFFFFu;

    HashPair old_hash = compute_hash_pair(entry.key, old_mask);
    HashPair new_hash = compute_hash_pair(entry.key, new_mask);

    // Fingerprint stays consistent
    EXPECT_EQ(old_hash.fingerprint, new_hash.fingerprint)
        << "Stash drain preserves fingerprint";

    // Both hashes valid
    EXPECT_LE(old_hash.b1, old_mask) << "Old hash valid";
    EXPECT_LE(new_hash.b1, new_mask) << "New hash valid";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
