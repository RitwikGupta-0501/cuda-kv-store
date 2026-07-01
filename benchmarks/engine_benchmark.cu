#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <algorithm>
#include "../src/engine/warpkv_engine.h"
#include "../src/gpu/bucket_cuckoo.h"

using namespace warpkv;

#define NUM_KEYS 500000
#define BATCH_SIZE 4096
#define INITIAL_BUCKETS 32768

int main(int argc, char** argv) {
    printf("\n");
    printf("============================================================================\n");
    printf("  WarpKV Engine End-to-End Benchmark (Pipeline + EBR + Rehash)\n");
    printf("============================================================================\n\n");

    printf("[Phase 1] Initializing WarpKVEngine...\n");
    WarpKVEngine engine;
    try {
        engine.init(INITIAL_BUCKETS);
    } catch (const std::exception& e) {
        printf("Init Error: %s\n", e.what());
        return 1;
    }
    printf("  ✓ Engine initialized with %u initial buckets\n\n", INITIAL_BUCKETS);

    printf("[Phase 2] Preparing %u unique test keys...\n", NUM_KEYS);
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
    printf("  ✓ Test data ready\n\n");

    printf("[Phase 3] Insertion Benchmark...\n");
    
    auto t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t count = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_KEYS - offset));
        
        // This pushes through the pipeline and blocks until completion
        engine.submit_insert_batch(&keys[offset], &values[offset], count);
        
        if (((offset / BATCH_SIZE) + 1) % 20 == 0) {
            printf("  [Batch %3u] %u keys submitted (background rehashing may trigger)\n", 
                   (offset / BATCH_SIZE) + 1, offset + count);
        }
    }
    engine.sync_all();
    auto t_end = std::chrono::high_resolution_clock::now();
    double insert_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    printf("  ✓ Insertion complete in %.2fs\n", insert_sec);
    printf("  ✓ Insert throughput: %.2f M keys/sec\n\n", (NUM_KEYS / 1e6) / insert_sec);

    printf("[Phase 4] Lookup Benchmark (Positive)...\n");
    std::vector<uint32_t> out_values(NUM_KEYS, 0);
    
    uint32_t found_count = 0;
    uint32_t value_mismatch = 0;
    
    t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t count = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_KEYS - offset));
        
        engine.submit_lookup_batch(&keys[offset], &out_values[offset], count);
    }
    engine.sync_all();
    t_end = std::chrono::high_resolution_clock::now();
    double lookup_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        if (out_values[i] != warpkv::NOT_FOUND) {
            found_count++;
            if (out_values[i] != values[i]) {
                value_mismatch++;
            }
        }
    }
    
    printf("  ✓ Lookup complete in %.2fs\n", lookup_sec);
    printf("  ✓ Lookup throughput: %.2f M keys/sec\n", (NUM_KEYS / 1e6) / lookup_sec);
    printf("  ✓ Keys found: %u / %u\n", found_count, NUM_KEYS);
    printf("  ✓ Value mismatches: %u\n\n", value_mismatch);

    printf("[Phase 5] Negative Lookup Benchmark...\n");
    const uint32_t NUM_MISSING_KEYS = 100000;
    std::vector<uint32_t> missing_keys(NUM_MISSING_KEYS);
    std::vector<uint32_t> out_missing(NUM_MISSING_KEYS, 0);
    
    for (uint32_t i = 0; i < NUM_MISSING_KEYS; ++i) {
        uint32_t mk;
        do {
            mk = dist(rng) + 0x80000000;
        } while (unique_keys.count(mk) > 0);
        missing_keys[i] = mk;
    }
    
    uint32_t false_positives = 0;
    t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t offset = 0; offset < NUM_MISSING_KEYS; offset += BATCH_SIZE) {
        uint32_t count = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_MISSING_KEYS - offset));
        engine.submit_lookup_batch(&missing_keys[offset], &out_missing[offset], count);
    }
    engine.sync_all();
    t_end = std::chrono::high_resolution_clock::now();
    double neg_lookup_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    for (uint32_t i = 0; i < NUM_MISSING_KEYS; ++i) {
        if (out_missing[i] != warpkv::NOT_FOUND) false_positives++;
    }
    
    printf("  ✓ Negative lookup complete in %.2fs\n", neg_lookup_sec);
    printf("  ✓ Negative lookup throughput: %.2f M keys/sec\n", (NUM_MISSING_KEYS / 1e6) / neg_lookup_sec);
    printf("  ✓ False positives: %u / %u\n\n", false_positives, NUM_MISSING_KEYS);
    
    bool passed = (found_count == NUM_KEYS) && (value_mismatch == 0) && (false_positives == 0);
    
    printf("============================================================================\n");
    printf("Engine Benchmark %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("============================================================================\n\n");
    
    return passed ? 0 : 1;
}
