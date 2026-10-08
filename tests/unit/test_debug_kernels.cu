#include <gtest/gtest.h>
#include <cstdio>
#include "../../src/gpu/bucket_cuckoo.h"
#include "../../include/warpkv/warpkv_allocator.h"

using namespace warpkv;

// Test: Verify BucketTable parameter passing and hash consistency
TEST(DebugKernels, BucketTableConsistency) {
    DefaultCudaAllocator allocator;
    constexpr uint32_t num_buckets = 256;

    BucketTable table;
    table.num_buckets = num_buckets;
    table.bucket_mask = num_buckets - 1;
    table.load_factor_limit = num_buckets / 2;
    table.buckets = static_cast<Bucket*>(allocator.allocate(num_buckets * sizeof(Bucket)));
    cudaMemset(table.buckets, 0, num_buckets * sizeof(Bucket));

    BucketTable* table_ptr = &table;

    // Print table info
    printf("\n=== BucketTable Consistency Debug ===\n");
    printf("table_ptr->num_buckets: %u\n", table_ptr->num_buckets);
    printf("table_ptr->bucket_mask: 0x%x\n", table_ptr->bucket_mask);
    printf("Expected mask (num_buckets-1): 0x%x\n", table_ptr->num_buckets - 1);
    printf("Masks match: %s\n", (table_ptr->bucket_mask == table_ptr->num_buckets - 1) ? "YES" : "NO");

    // Test hash consistency
    KeyT test_key = 98765ULL;

    // Compute hash on host with different masks
    HashPair hash_with_table_mask = compute_hash_pair(test_key, table_ptr->bucket_mask);
    HashPair hash_with_manual_mask = compute_hash_pair(test_key, table_ptr->num_buckets - 1);
    HashPair hash_with_zero_mask = compute_hash_pair(test_key, 0);

    printf("\nHash computation for key %llu:\n", (unsigned long long)test_key);
    printf("With table_mask:   b1=0x%x, b2=0x%x, fp=0x%02x\n",
           hash_with_table_mask.b1, hash_with_table_mask.b2, hash_with_table_mask.fingerprint);
    printf("With (num-1) mask: b1=0x%x, b2=0x%x, fp=0x%02x\n",
           hash_with_manual_mask.b1, hash_with_manual_mask.b2, hash_with_manual_mask.fingerprint);
    printf("With zero mask:    b1=0x%x, b2=0x%x, fp=0x%02x\n",
           hash_with_zero_mask.b1, hash_with_zero_mask.b2, hash_with_zero_mask.fingerprint);

    printf("\nHashes match: %s\n",
           (hash_with_table_mask.fingerprint == hash_with_manual_mask.fingerprint) ? "YES" : "NO");

    // The CRITICAL issue: check if buckets are within valid range
    uint32_t b2 = hash_with_table_mask.b2;
    printf("\nBucket index validation:\n");
    printf("b2 index: 0x%x (%u)\n", b2, b2);
    printf("Max valid index: 0x%x (%u)\n", table_ptr->bucket_mask, table_ptr->bucket_mask);
    printf("b2 in range: %s\n", (b2 <= table_ptr->bucket_mask) ? "YES" : "NO");

    allocator.deallocate(table.buckets);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
