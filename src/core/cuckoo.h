#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace kvstore {

// Cuckoo hash table structure (Structure of Arrays layout)
struct CuckooTable {
  // Table 1
  uint64_t *keys_t1; // [capacity]
  uint64_t *vals_t1; // [capacity]

  // Table 2
  uint64_t *keys_t2; // [capacity]
  uint64_t *vals_t2; // [capacity]

  uint32_t capacity; // Number of slots per table
  uint32_t count;    // Number of entries currently stored

  bool is_on_device; // True if allocated on GPU
};

// Special marker for empty slots
constexpr uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;

// Create empty hash table on CPU
bool create_table_host(CuckooTable &table, uint32_t capacity);

// Insert key-value pair (CPU only, for construction)
bool insert(CuckooTable &table, uint64_t key, uint64_t value);

// Copy table from CPU to GPU
bool copy_table_to_device(const CuckooTable &host_table,
                          CuckooTable &device_table);

// Free memory
void free_table_host(CuckooTable &table);
void free_table_device(CuckooTable &table);

// Print statistics
void print_table_stats(const CuckooTable &table);

} // namespace kvstore
