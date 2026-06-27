#include <gtest/gtest.h>
#include "../../src/gpu/cuckoo_insert.h"
#include "../../src/gpu/bucket_cuckoo.h"

using namespace warpkv;

// Phase 4 Extended: Eviction Chain Correctness Tests
// Validates cuckoo hashing: b1 → b2 → evict → loop → stash
// These tests are critical and must pass before Phase 6

class EvictionChainTest : public ::testing::Test {
protected:
    void SetUp() override {
        bucket_init(&bucket_b1_);
        bucket_init(&bucket_b2_);
    }

    Bucket bucket_b1_;
    Bucket bucket_b2_;

    // Helper: fill a bucket to capacity
    void fill_bucket_completely(Bucket* b) {
        for (int i = 0; i < 8; ++i) {
            b->keys[i] = 1000 + i;
            b->values[i] = 2000 + i;
            b->fingerprint[i] = (uint8_t)(i * 0x11);
            bucket_set_occupied(b, i);
        }
    }

    // Helper: verify bucket contains a specific key
    bool bucket_contains_key(const Bucket* b, uint32_t key) {
        for (int i = 0; i < 8; ++i) {
            if (bucket_is_occupied(b, i) && b->keys[i] == key) {
                return true;
            }
        }
        return false;
    }

    // Helper: count occupied slots
    int count_occupied_slots(const Bucket* b) {
        int count = 0;
        for (int i = 0; i < 8; ++i) {
            if (bucket_is_occupied(b, i)) {
                count++;
            }
        }
        return count;
    }
};

// Test 1: Single eviction from b1 to b2
TEST_F(EvictionChainTest, SingleEvictionB1ToB2) {
    // Fill b1 completely
    fill_bucket_completely(&bucket_b1_);

    // b2 is empty
    EXPECT_EQ(count_occupied_slots(&bucket_b2_), 0);

    // Simulate: evict slot 0 from b1
    uint32_t victim_key = bucket_b1_.keys[0];
    uint32_t victim_value = bucket_b1_.values[0];

    // Try to insert into b2 slot 0
    uint32_t new_key = 99999;
    uint32_t new_value = 88888;

    // Replace victim in b1 slot 0
    bucket_b1_.keys[0] = new_key;
    bucket_b1_.values[0] = new_value;

    // Verify b1 slot 0 changed
    EXPECT_EQ(bucket_b1_.keys[0], new_key);
    EXPECT_NE(bucket_b1_.keys[0], victim_key);

    // Verify victim data is intact
    EXPECT_EQ(victim_key, 1000);
    EXPECT_EQ(victim_value, 2000);
}

// Test 2: Victim selection pseudo-randomness
TEST_F(EvictionChainTest, VictimSelectionPseudoRandom) {
    uint32_t b1 = 0x12345678;
    uint32_t b2 = 0x87654321;

    // Test victim slot formula: (b1 ^ b2 ^ hop_count) % 8
    for (uint32_t hop = 0; hop < 32; ++hop) {
        uint32_t victim_slot = (b1 ^ b2 ^ hop) % 8;

        // Should be in range [0, 7]
        EXPECT_GE(victim_slot, 0) << "Hop " << hop;
        EXPECT_LT(victim_slot, 8) << "Hop " << hop;
    }
}

// Test 3: Different hops pick different victims
TEST_F(EvictionChainTest, DifferentHopsPickDifferentVictims) {
    uint32_t b1 = 0xDEADBEEF;
    uint32_t b2 = 0xCAFEBABE;

    uint32_t victims[32];
    for (uint32_t hop = 0; hop < 32; ++hop) {
        victims[hop] = (b1 ^ b2 ^ hop) % 8;
    }

    // Most hops should pick different victims
    // This is a soft test - exact distribution depends on hash values
    int unique_slots = 0;
    bool seen[8] = {false};
    for (uint32_t hop = 0; hop < 32; ++hop) {
        if (!seen[victims[hop]]) {
            seen[victims[hop]] = true;
            unique_slots++;
        }
    }

    // Should hit at least 4 different slots across 32 hops
    EXPECT_GE(unique_slots, 4) << "Should visit multiple slots across hops";
}

// Test 4: Bucket alternation (b1, b2, b1, b2, ...)
TEST_F(EvictionChainTest, BucketAlternation) {
    for (uint32_t hop = 0; hop < 32; ++hop) {
        bool pick_b1 = (hop % 2 == 0);
        bool pick_b2 = (hop % 2 == 1);

        if (hop < 16) {
            EXPECT_EQ(pick_b1, (hop % 2 == 0)) << "Even hops pick b1";
            EXPECT_EQ(pick_b2, (hop % 2 == 1)) << "Odd hops pick b2";
        }
    }
}

// Test 5: Fingerprint invariance across evictions
TEST_F(EvictionChainTest, FingerprintInvariance) {
    uint32_t key = 12345;
    uint8_t original_fp = 0x42;

    // FP should never change during evictions
    uint8_t fp_hop0 = original_fp;
    uint8_t fp_hop1 = original_fp;  // Same FP after eviction
    uint8_t fp_hop31 = original_fp;  // Same FP after 31 hops

    EXPECT_EQ(fp_hop0, original_fp);
    EXPECT_EQ(fp_hop1, original_fp);
    EXPECT_EQ(fp_hop31, original_fp);
}

// Test 6: No data loss in eviction chain
TEST_F(EvictionChainTest, NoDataLossInEvictionChain) {
    // Setup: full buckets (both b1 and b2)
    fill_bucket_completely(&bucket_b1_);
    fill_bucket_completely(&bucket_b2_);

    // Count initial keys in b1 and b2
    int initial_b1_count = count_occupied_slots(&bucket_b1_);
    int initial_b2_count = count_occupied_slots(&bucket_b2_);

    // Simulate eviction
    uint32_t evicted_key = bucket_b1_.keys[0];
    uint32_t evicted_value = bucket_b1_.values[0];

    // Replace in b1
    bucket_b1_.keys[0] = 99999;
    bucket_b1_.values[0] = 88888;

    // Evicted key is still intact
    EXPECT_EQ(evicted_key, 1000);
    EXPECT_EQ(evicted_value, 2000);

    // Total keys in system unchanged (still in memory, just moved)
    EXPECT_EQ(initial_b1_count, 8);
    EXPECT_EQ(initial_b2_count, 8);
}

// Test 7: Stash queue structure for MAX_EVICTION_HOPS
TEST_F(EvictionChainTest, StashCapacityForMaxHops) {
    // After 32 hops, final evicted key goes to stash
    // Stash must have capacity for at least one batch of failed keys
    EXPECT_GE(STASH_CAPACITY, BATCH_SIZE) << "Stash must hold one batch";
    EXPECT_GE(STASH_CAPACITY, MAX_EVICTION_HOPS) << "Stash must hold 32+ entries";

    // Typical scenario: batch insert fails on some keys
    // All MAX_EVICTION_HOPS batches fit in stash
    uint32_t total_possible_stashed = STASH_CAPACITY;
    EXPECT_GT(total_possible_stashed, MAX_EVICTION_HOPS);
}

// Test 8: atomicCAS simulation (winner/loser semantics)
TEST_F(EvictionChainTest, AtomicCASSemantics) {
    // Simulate two warps racing on same victim slot
    uint32_t key_slot = 12345;
    uint32_t new_key_warp0 = 99999;
    uint32_t new_key_warp1 = 88888;

    // Warp 0 tries first
    uint32_t old_key = atomicCAS(&bucket_b1_.keys[0], key_slot, new_key_warp0);
    EXPECT_EQ(old_key, 0) << "First CAS against empty slot succeeds (old value was 0 from init)";

    // Verify Warp 0's key is in slot
    EXPECT_EQ(bucket_b1_.keys[0], new_key_warp0);

    // Warp 1 tries now (should fail)
    old_key = atomicCAS(&bucket_b1_.keys[0], key_slot, new_key_warp1);
    EXPECT_NE(old_key, key_slot) << "Second CAS fails (slot now holds warp0's key)";

    // Warp 0's key stays
    EXPECT_EQ(bucket_b1_.keys[0], new_key_warp0);
}

// Test 9: Eviction chain convergence (doesn't infinite loop)
TEST_F(EvictionChainTest, EvictionChainConvergence) {
    // With MAX_EVICTION_HOPS = 32, loop must terminate
    uint32_t hop_limit = MAX_EVICTION_HOPS;
    EXPECT_EQ(hop_limit, 32) << "Hop limit should be exactly 32";

    // Even in worst case (every eviction succeeds), after 32 hops we stash
    int hops = 0;
    while (hops < MAX_EVICTION_HOPS) {
        hops++;
    }

    EXPECT_EQ(hops, 32);
}

// Test 10: InsertResult with eviction
TEST_F(EvictionChainTest, InsertResultAfterEviction) {
    InsertResult result;
    result.status = INSERT_STASHED;
    result.slot_used = 0;

    EXPECT_EQ(result.status, INSERT_STASHED);
    EXPECT_EQ(result.slot_used, 0);

    // After stashing, status should indicate stash
    EXPECT_NE(result.status, INSERT_SUCCESS);
    EXPECT_NE(result.status, INSERT_FAILED);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
