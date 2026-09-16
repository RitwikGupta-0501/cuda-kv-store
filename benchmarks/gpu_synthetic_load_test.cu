#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <set>
#include <random>
#include "../src/engine/warpkv_engine.h"

using namespace warpkv;

#define NUM_KEYS 100000
#define BATCH_SIZE 4096

int main(int argc, char** argv) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  WarpKV GPU Synthetic Load Test — REAL Kernel Execution                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n\n");

    // ========== Phase 1: Allocate tables ==========
    printf("[Phase 1] Allocating tables (WarpKVEngine)...\n");
    WarpKVEngine engine;
    try {
        engine.init(4194304); // 4M buckets
    } catch (const std::exception& e) {
        printf("Engine Init Error: %s\n", e.what());
        return 1;
    }
    printf("  ✓ Tables and Engine allocated\n\n");

    // ========== Phase 2: Prepare test data ==========
    printf("[Phase 2] Preparing test data...\n");
    printf("  Total keys to insert: %u\n", NUM_KEYS);
    printf("  Batch size: %u\n\n", BATCH_SIZE);

    std::vector<uint32_t> keys(NUM_KEYS);
    std::vector<uint32_t> values(NUM_KEYS);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, 0xFFFFFFFE); // Avoid 0

    std::set<uint32_t> unique_keys;
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        uint32_t k;
        do {
            k = dist(rng);
        } while (unique_keys.count(k) > 0);
        
        unique_keys.insert(k);
        keys[i] = k;
        values[i] = dist(rng);
    }

    // ========== Phase 3: Insert batches ==========
    printf("[Phase 3] Inserting keys in batches...\n");

    uint32_t total_success = 0;

    time_t phase3_start = time(NULL);

    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch_size = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_KEYS - offset));
        
        engine.submit_insert_batch(&keys[offset], &values[offset], current_batch_size);
        total_success += current_batch_size;

        if (((offset / BATCH_SIZE) + 1) % 10 == 0) {
            printf("  [Batch %3u] %u keys inserted\n", 
                   (offset / BATCH_SIZE) + 1, total_success);
        }
    }

    time_t phase3_end = time(NULL);
    printf("\n  Insert phase complete in %.1fs\n\n", difftime(phase3_end, phase3_start));

    // ========== Phase 4: Verification (Positive Lookups) ==========
    printf("[Phase 4] Verifying inserted keys (100%% Hit Rate Expected)...\n");

    uint32_t found_count = 0;
    uint32_t value_mismatch = 0;

    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch_size = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_KEYS - offset));
        
        std::vector<uint32_t> out_values(current_batch_size);
        
        engine.submit_lookup_batch(&keys[offset], out_values.data(), current_batch_size);
        
        for (uint32_t i = 0; i < current_batch_size; ++i) {
            if (out_values[i] != NOT_FOUND) {
                found_count++;
                if (out_values[i] != values[offset + i]) value_mismatch++;
            }
        }
    }
    
    printf("  Keys found in buckets: %u / %u\n", found_count, total_success);
    printf("  Value mismatches: %u\n\n", value_mismatch);

    // ========== Phase 5: Verification (Negative Lookups) ==========
    printf("[Phase 5] Negative lookups (0%% Hit Rate Expected)...\n");
    const uint32_t NUM_MISSING_KEYS = 50000;
    std::vector<uint32_t> missing_keys(NUM_MISSING_KEYS);
    
    for (uint32_t i = 0; i < NUM_MISSING_KEYS; ++i) {
        missing_keys[i] = dist(rng) + 0x80000000; 
    }
    
    uint32_t false_positives = 0;
    
    for (uint32_t offset = 0; offset < NUM_MISSING_KEYS; offset += BATCH_SIZE) {
        uint32_t current_batch_size = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_MISSING_KEYS - offset));
        
        std::vector<uint32_t> out_values(current_batch_size);
        
        engine.submit_lookup_batch(&missing_keys[offset], out_values.data(), current_batch_size);
        
        for (uint32_t i = 0; i < current_batch_size; ++i) {
            if (out_values[i] != NOT_FOUND) false_positives++;
        }
    }
    
    printf("  False positives: %u / %u\n\n", false_positives, NUM_MISSING_KEYS);

    // ========== Final Report ==========
    printf("============================================================================\n");
    printf("GPU Synthetic Load Test Results (EMPIRICAL)\n");
    printf("============================================================================\n");
    printf("Insertions:\n");
    printf("  Total:       %u\n", NUM_KEYS);
    printf("  Successful:  %u (%.2f%%)\n\n", total_success, (float)total_success / NUM_KEYS * 100);

    bool data_loss = (found_count != total_success) || (value_mismatch > 0) || (false_positives > 0);
    if (!data_loss) {
        printf("Data Integrity:\n");
        printf("  ✓ NO DATA LOSS DETECTED\n");
        printf("  ✓ All bucket insertions retrieved successfully\n");
        printf("  ✓ Zero value mismatches\n");
        printf("  ✓ Zero false positives\n");
    } else {
        printf("Data Integrity:\n");
        printf("  ✗ DATA LOSS OR CORRUPTION DETECTED!\n");
    }
    printf("============================================================================\n");
    printf("GPU Synthetic Load Test %s\n", data_loss ? "FAILED ✗" : "PASSED ✓");
    printf("============================================================================\n\n");

    return data_loss ? 1 : 0;
}
