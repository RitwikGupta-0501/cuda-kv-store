#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include "../src/engine/warpkv_engine.h"
#include "zipfian_generator.h"

using namespace warpkv;

#define NUM_KEYS 10000000
#define NUM_OPERATIONS 10000000
#define BATCH_SIZE 4096
#define INITIAL_BUCKETS 4194304

int main(int argc, char** argv) {
    printf("\n============================================================================\n");
    printf("  WarpKV YCSB Workload C (100%% Read, Zipfian Distribution)\n");
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

    printf("[Phase 2] Loading %u sequential keys...\n", NUM_KEYS);
    std::vector<uint32_t> keys(NUM_KEYS);
    std::vector<uint32_t> values(NUM_KEYS);
    
    for (uint32_t i = 0; i < NUM_KEYS; ++i) {
        keys[i] = i + 1; // 1-indexed to avoid EMPTY_KEY (0)
        values[i] = (i + 1) * 2;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t offset = 0; offset < NUM_KEYS; offset += BATCH_SIZE) {
        uint32_t count = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_KEYS - offset));
        engine.submit_insert_batch(&keys[offset], &values[offset], count);
    }
    engine.sync_all();
    auto t_end = std::chrono::high_resolution_clock::now();
    printf("  ✓ Insert throughput: %.2f M keys/sec\n\n", (NUM_KEYS / 1e6) / std::chrono::duration<double>(t_end - t_start).count());

    printf("[Phase 3] Generating %u Zipfian queries...\n", NUM_OPERATIONS);
    std::vector<uint32_t> query_keys(NUM_OPERATIONS);
    std::vector<uint32_t> out_values(NUM_OPERATIONS, 0);

    ScrambledZipfianGenerator zipf(1, NUM_KEYS);
    for (uint32_t i = 0; i < NUM_OPERATIONS; ++i) {
        query_keys[i] = zipf.next();
    }
    printf("  ✓ Queries ready\n\n");

    printf("[Phase 4] Executing YCSB Workload C...\n");
    t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t offset = 0; offset < NUM_OPERATIONS; offset += BATCH_SIZE) {
        uint32_t count = std::min((uint32_t)BATCH_SIZE, (uint32_t)(NUM_OPERATIONS - offset));
        engine.submit_lookup_batch(&query_keys[offset], &out_values[offset], count);
    }
    engine.sync_all();
    t_end = std::chrono::high_resolution_clock::now();

    double lookup_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint32_t errors = 0;
    for (uint32_t i = 0; i < NUM_OPERATIONS; ++i) {
        if (out_values[i] != query_keys[i] * 2) {
            errors++;
        }
    }

    printf("  ✓ YCSB Workload C complete in %.2fs\n", lookup_sec);
    printf("  ✓ Lookup throughput: %.2f M keys/sec\n", (NUM_OPERATIONS / 1e6) / lookup_sec);
    printf("  ✓ Mismatches: %u\n\n", errors);

    return errors == 0 ? 0 : 1;
}
