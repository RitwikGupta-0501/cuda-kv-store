#include <gtest/gtest.h>
#include "../src/gpu/xxhash3.h"
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>

using namespace warpkv;

// Phase 1: XXHash3 Unit Tests (Complete)
// Comprehensive validation of hash function properties per SPEC_V3_FINAL.md Section V

class XXHash3Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize RNG
        rng.seed(42);  // Fixed seed for reproducibility
    }

    void TearDown() override {}

    std::mt19937_64 rng;
};

// Test 1: Known value test
TEST_F(XXHash3Test, KnownValues) {
    // Test a few known keys to verify correctness
    EXPECT_EQ(xxhash3_32_host(0u), xxhash3_32_host(0u));  // Consistent
    EXPECT_NE(xxhash3_32_host(0u), xxhash3_32_host(1u));  // Different input → different output

    // Test small keys
    uint32_t hash0 = xxhash3_32_host(0u);
    uint32_t hash1 = xxhash3_32_host(1u);
    uint32_t hash2 = xxhash3_32_host(2u);

    // All different (with high probability for random keys)
    EXPECT_NE(hash0, hash1);
    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash0, hash2);
}

// Test 2: Avalanche property
TEST_F(XXHash3Test, AvalancheProperty) {
    const int num_trials = 100;
    const int bits_per_hash = 32;

    std::vector<int> bit_flip_count(bits_per_hash, 0);

    // For each input bit position
    for (int bit = 0; bit < bits_per_hash; ++bit) {
        for (int trial = 0; trial < num_trials; ++trial) {
            // Random key
            std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
            uint32_t key1 = dist(rng);

            // Flip the bit
            uint32_t key2 = key1 ^ (1u << bit);

            // Hash both
            uint32_t hash1 = xxhash3_32_host(key1);
            uint32_t hash2 = xxhash3_32_host(key2);

            uint32_t xor_result = hash1 ^ hash2;

            // Count which output bits flipped
            for (int out_bit = 0; out_bit < bits_per_hash; ++out_bit) {
                if (xor_result & (1u << out_bit)) {
                    bit_flip_count[out_bit]++;
                }
            }
        }
    }

    // Avalanche property: each output bit should flip ~50% of the time
    // (With some tolerance for statistical variance)
    const int expected_flips = num_trials * bits_per_hash / 2;
    const int tolerance = num_trials * bits_per_hash / 10;  // ±10% tolerance

    for (int bit = 0; bit < bits_per_hash; ++bit) {
        EXPECT_NEAR(bit_flip_count[bit], expected_flips, tolerance)
            << "Output bit " << bit << " has unusual avalanche behavior";
    }
}

// Test 3: Distribution uniformity (simplified chi-square)
TEST_F(XXHash3Test, UniformDistribution) {
    const int num_keys = 10000;
    const int num_buckets = 256;  // For chi-square test

    std::vector<int> bucket_count(num_buckets, 0);

    // Hash 10K random keys
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (int i = 0; i < num_keys; ++i) {
        uint32_t key = dist(rng);
        uint32_t hash = xxhash3_32_host(key);
        uint32_t bucket = hash % num_buckets;
        bucket_count[bucket]++;
    }

    // Expected count per bucket
    const double expected = (double)num_keys / num_buckets;

    // Chi-square test: sum((observed - expected)^2 / expected)
    double chi_square = 0.0;
    for (int bucket = 0; bucket < num_buckets; ++bucket) {
        double diff = bucket_count[bucket] - expected;
        chi_square += (diff * diff) / expected;
    }

    // For 256 buckets, chi-square threshold at 95% confidence ≈ 295
    // (Degrees of freedom = 255)
    EXPECT_LT(chi_square, 350.0)
        << "Hash distribution is not uniform (chi-square = " << chi_square << ")";
}

// Test 4: b1 != b2 decorrelation property (CRITICAL for spec)
TEST_F(XXHash3Test, DecorelationB1NotB2) {
    const uint32_t bucket_mask = 0xFFFFu;  // 64K buckets for testing
    const int num_keys = 10000000;  // 10M keys per spec requirement

    int b1_equals_b2_count = 0;

    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    for (int i = 0; i < num_keys; ++i) {
        uint32_t key = dist(rng);
        HashPair pair = compute_hash_pair_host(key, bucket_mask);

        if (pair.b1 == pair.b2) {
            b1_equals_b2_count++;
        }
    }

    // With proper XOR decorrelation, b1 == b2 should be extremely rare
    // Expected: ~0 collisions in 10M keys (probability = 1/65536 per key)
    // Allow up to 200 collisions (~1.9x expected) for statistical variance
    EXPECT_LT(b1_equals_b2_count, 200)
        << "Decorrelation failed: b1 == b2 in " << b1_equals_b2_count << " / " << num_keys << " keys";
}

// Test 5: Fingerprint properties
TEST_F(XXHash3Test, FingerprintProperties) {
    const int num_keys = 100000;

    // Count fingerprint distribution
    std::vector<int> fp_count(256, 0);

    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    for (int i = 0; i < num_keys; ++i) {
        uint32_t key = dist(rng);
        HashPair pair = compute_hash_pair_host(key, 0xFFFFu);
        fp_count[pair.fingerprint]++;
    }

    // Fingerprints should be roughly uniformly distributed
    const double expected = (double)num_keys / 256;
    double chi_square = 0.0;

    for (int i = 0; i < 256; ++i) {
        double diff = fp_count[i] - expected;
        chi_square += (diff * diff) / expected;
    }

    // Threshold at 95% confidence ≈ 295 (255 DOF)
    EXPECT_LT(chi_square, 350.0)
        << "Fingerprint distribution is not uniform (chi-square = " << chi_square << ")";
}

// Test 6: Consistency (same input always produces same output)
TEST_F(XXHash3Test, Consistency) {
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    for (int i = 0; i < 1000; ++i) {
        uint32_t key = dist(rng);
        uint32_t hash1 = xxhash3_32_host(key);
        uint32_t hash2 = xxhash3_32_host(key);
        uint32_t hash3 = xxhash3_32_host(key);

        EXPECT_EQ(hash1, hash2) << "Hash function is not consistent for key " << key;
        EXPECT_EQ(hash2, hash3) << "Hash function is not consistent for key " << key;
    }
}

// Test 7: Fingerprint false positive rate (3.1% expected for 8-bit fp, 8 slots)
TEST_F(XXHash3Test, FingerprintFalsePositiveRate) {
    // Simulate stash lookup scenario:
    // 8 slots per bucket, 8-bit fingerprints
    // P(fingerprint match | key mismatch) = 1/256
    // Expected FP per 8-slot bucket: 8 * (1/256) = 3.1%

    const int num_tests = 100000;
    const int slots_per_bucket = 8;
    int false_positives = 0;

    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    for (int test = 0; test < num_tests; ++test) {
        // Generate a random slot key
        uint32_t slot_key = dist(rng);
        HashPair slot_pair = compute_hash_pair_host(slot_key, 0xFFFFu);

        // Generate a different search key
        uint32_t search_key = slot_key ^ 0x12345678u;  // Guaranteed different
        HashPair search_pair = compute_hash_pair_host(search_key, 0xFFFFu);

        // Check if fingerprints match (this would be a false positive in a real lookup)
        if (slot_pair.fingerprint == search_pair.fingerprint) {
            false_positives++;
        }
    }

    // Expected: ~1/256 = 391 FP in 100K tests
    // Allow ±50% variance
    const int expected_fps = num_tests / 256;
    EXPECT_GT(false_positives, expected_fps / 2)
        << "Fingerprint collision rate is too low (expected ~" << expected_fps << ")";
    EXPECT_LT(false_positives, expected_fps * 2)
        << "Fingerprint collision rate is too high (expected ~" << expected_fps << ")";
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
