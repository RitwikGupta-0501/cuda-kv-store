#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "../src/engine/warpkv_engine.h"

using namespace warpkv;

#define BATCH_SIZE 4096
#define INITIAL_BUCKETS 8388608 // 8M buckets = 64M slots

int main(int argc, char** argv) {
    printf("\n============================================================================\n");
    printf("  WarpKV Load Factor Curve Benchmark\n");
    printf("============================================================================\n\n");

    printf("[Phase 1] Initializing WarpKVEngine with Automatic Rehashing DISABLED...\n");
    WarpKVEngine engine;
    try {
        engine.init(INITIAL_BUCKETS);
        engine.disable_automatic_rehash(true);
    } catch (const std::exception& e) {
        printf("Init Error: %s\n", e.what());
        return 1;
    }
    
    uint64_t total_slots = INITIAL_BUCKETS * 8; // 8 slots per bucket
    printf("  ✓ Engine initialized with %u buckets (%lu total slots)\n\n", INITIAL_BUCKETS, total_slots);

    std::vector<int> target_load_factors = {10, 20, 30, 40, 45, 50, 60, 70, 80, 90};
    
    uint64_t current_keys = 0;
    std::vector<uint32_t> all_keys;
    std::vector<uint32_t> all_values;
    
    printf("%-15s | %-20s | %-20s | %-15s\n", "Load Factor %", "Insert (M keys/s)", "Lookup (M keys/s)", "Missing Keys");
    printf("--------------------------------------------------------------------------------------\n");

    for (int target_pct : target_load_factors) {
        uint64_t target_keys = (total_slots * target_pct) / 100;
        uint64_t keys_to_add = target_keys - current_keys;
        
        std::vector<uint32_t> batch_keys(keys_to_add);
        std::vector<uint32_t> batch_values(keys_to_add);
        
        for (uint64_t i = 0; i < keys_to_add; ++i) {
            uint32_t k = current_keys + i + 1;
            batch_keys[i] = k;
            batch_values[i] = k * 2;
            all_keys.push_back(k);
            all_values.push_back(k * 2);
        }
        
        // --- Measure Inserts ---
        auto t_start_ins = std::chrono::high_resolution_clock::now();
        for (uint64_t offset = 0; offset < keys_to_add; offset += BATCH_SIZE) {
            uint32_t count = std::min((uint64_t)BATCH_SIZE, keys_to_add - offset);
            engine.submit_insert_batch(&batch_keys[offset], &batch_values[offset], count);
        }
        engine.sync_all();
        auto t_end_ins = std::chrono::high_resolution_clock::now();
        double insert_sec = std::chrono::duration<double>(t_end_ins - t_start_ins).count();
        double insert_m_sec = (keys_to_add / 1e6) / insert_sec;
        
        // --- Measure Lookups ---
        std::vector<uint32_t> out_values(target_keys, 0);
        auto t_start_lkp = std::chrono::high_resolution_clock::now();
        for (uint64_t offset = 0; offset < target_keys; offset += BATCH_SIZE) {
            uint32_t count = std::min((uint64_t)BATCH_SIZE, target_keys - offset);
            engine.submit_lookup_batch(&all_keys[offset], &out_values[offset], count);
        }
        engine.sync_all();
        auto t_end_lkp = std::chrono::high_resolution_clock::now();
        double lookup_sec = std::chrono::duration<double>(t_end_lkp - t_start_lkp).count();
        double lookup_m_sec = (target_keys / 1e6) / lookup_sec;
        
        // --- Verify Integrity ---
        uint64_t missing = 0;
        for (uint64_t i = 0; i < target_keys; ++i) {
            if (out_values[i] != all_values[i]) {
                missing++;
            }
        }
        
        printf("%-15d | %-20.2f | %-20.2f | %-15lu\n", target_pct, insert_m_sec, lookup_m_sec, missing);
        
        current_keys = target_keys;
    }
    
    printf("\nNote: Missing keys at >50%% load factor demonstrate Cuckoo Hash stash overflow.\n");
    printf("This is why WarpKV's automatic background rehash triggers at 50%% capacity.\n\n");

    return 0;
}
