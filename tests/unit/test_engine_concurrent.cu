#include <gtest/gtest.h>
#include "engine/warpkv_engine.h"
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <iostream>

using namespace warpkv;

TEST(EnginePipeline, ConcurrentSubmissions_10Threads_100kBatches) {
    WarpKVEngine engine;
    // 4M buckets = 33.5M slots. At 4.09M total keys (~12% load), cuckoo chains
    // almost never exhaust 32 hops, making stash pushes statistically impossible.
    uint32_t num_buckets = 4194304;
    engine.init(num_buckets);
    
    std::vector<std::thread> threads;
    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_lookups{0};
    std::vector<uint32_t> mismatches;
    std::mutex mismatch_lock;
    
    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&engine, &total_inserts, &total_lookups, &mismatches, &mismatch_lock, t]() {
            std::vector<uint32_t> keys(BATCH_SIZE);
            std::vector<uint32_t> values(BATCH_SIZE);
            std::vector<uint32_t> values_out(BATCH_SIZE);
            
            // 100 batches per thread, 4096 keys each. Total: 10 * 100 * 4096 = 4.09M keys.
            for (int b = 0; b < 100; ++b) {
                for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
                    uint32_t key = t * 10000000 + b * BATCH_SIZE + i;
                    keys[i] = key;
                    values[i] = key ^ 0xDEADBEEF;
                }
                
                // Insert
                engine.submit_insert_batch(keys.data(), values.data(), BATCH_SIZE);
                total_inserts += BATCH_SIZE;
                
                // Lookup immediately after (stress the pipeline)
                engine.submit_lookup_batch(keys.data(), values_out.data(), BATCH_SIZE);
                total_lookups += BATCH_SIZE;
                
                for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
                    if (values_out[i] != values[i]) {
                        std::lock_guard<std::mutex> lock(mismatch_lock);
                        mismatches.push_back(keys[i]);
                    }
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(mismatches.size(), 0) 
        << "Concurrent pipeline corrupted " << mismatches.size() << " keys";
    EXPECT_EQ(total_inserts, 10 * 100 * 4096);
    EXPECT_EQ(total_lookups, 10 * 100 * 4096);
}
