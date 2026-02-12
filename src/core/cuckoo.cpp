#include "core/cuckoo.h"
#include "core/config.h"
#include "core/hash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kvstore {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

bool create_table_host(CuckooTable &table, uint32_t capacity) {
  table.capacity = capacity;
  table.count = 0;
  table.is_on_device = false;

  // Allocate memory for both tables
  size_t bytes_per_array = capacity * sizeof(uint64_t);

  table.keys_t1 = (uint64_t *)malloc(bytes_per_array);
  table.vals_t1 = (uint64_t *)malloc(bytes_per_array);
  table.keys_t2 = (uint64_t *)malloc(bytes_per_array);
  table.vals_t2 = (uint64_t *)malloc(bytes_per_array);

  if (!table.keys_t1 || !table.vals_t1 || !table.keys_t2 || !table.vals_t2) {
    fprintf(stderr, "Failed to allocate host memory for hash table\n");
    free_table_host(table);
    return false;
  }

  // Initialize all slots to EMPTY
  for (uint32_t i = 0; i < capacity; i++) {
    table.keys_t1[i] = EMPTY_KEY;
    table.keys_t2[i] = EMPTY_KEY;
    table.vals_t1[i] = 0;
    table.vals_t2[i] = 0;
  }

  return true;
}

bool insert(CuckooTable &table, uint64_t key, uint64_t value) {
  if (table.is_on_device) {
    fprintf(stderr, "Cannot insert into device table from CPU\n");
    return false;
  }

  if (key == EMPTY_KEY) {
    fprintf(stderr, "Cannot insert EMPTY_KEY (reserved value)\n");
    return false;
  }

  uint64_t current_key = key;
  uint64_t current_val = value;

  // Cuckoo insertion with eviction
  for (size_t iteration = 0; iteration < TuningParams::CUCKOO_MAX_EVICTIONS;
       iteration++) {
    // Try table 1
    uint32_t pos1 = hash1(current_key, table.capacity);

    if (table.keys_t1[pos1] == EMPTY_KEY) {
      // Slot is empty - insert here
      table.keys_t1[pos1] = current_key;
      table.vals_t1[pos1] = current_val;
      table.count++;
      return true;
    }

    if (table.keys_t1[pos1] == current_key) {
      // Key already exists - update value
      table.vals_t1[pos1] = current_val;
      return true;
    }

    // Evict existing item from table 1
    uint64_t evicted_key = table.keys_t1[pos1];
    uint64_t evicted_val = table.vals_t1[pos1];

    table.keys_t1[pos1] = current_key;
    table.vals_t1[pos1] = current_val;

    current_key = evicted_key;
    current_val = evicted_val;

    // Try table 2
    uint32_t pos2 = hash2(current_key, table.capacity);

    if (table.keys_t2[pos2] == EMPTY_KEY) {
      table.keys_t2[pos2] = current_key;
      table.vals_t2[pos2] = current_val;
      table.count++;
      return true;
    }

    if (table.keys_t2[pos2] == current_key) {
      table.vals_t2[pos2] = current_val;
      return true;
    }

    // Evict from table 2
    evicted_key = table.keys_t2[pos2];
    evicted_val = table.vals_t2[pos2];

    table.keys_t2[pos2] = current_key;
    table.vals_t2[pos2] = current_val;

    current_key = evicted_key;
    current_val = evicted_val;
  }

  // Failed to insert after max evictions
  fprintf(stderr, "Insert failed: max evictions reached (table too full?)\n");
  return false;
}

bool copy_table_to_device(const CuckooTable &host_table,
                          CuckooTable &device_table) {
  if (host_table.is_on_device) {
    fprintf(stderr, "Source table is already on device\n");
    return false;
  }

  device_table.capacity = host_table.capacity;
  device_table.count = host_table.count;
  device_table.is_on_device = true;

  size_t bytes_per_array = host_table.capacity * sizeof(uint64_t);

  // Allocate GPU memory
  CUDA_CHECK(cudaMalloc(&device_table.keys_t1, bytes_per_array));
  CUDA_CHECK(cudaMalloc(&device_table.vals_t1, bytes_per_array));
  CUDA_CHECK(cudaMalloc(&device_table.keys_t2, bytes_per_array));
  CUDA_CHECK(cudaMalloc(&device_table.vals_t2, bytes_per_array));

  // Copy data to GPU
  CUDA_CHECK(cudaMemcpy(device_table.keys_t1, host_table.keys_t1,
                        bytes_per_array, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_table.vals_t1, host_table.vals_t1,
                        bytes_per_array, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_table.keys_t2, host_table.keys_t2,
                        bytes_per_array, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_table.vals_t2, host_table.vals_t2,
                        bytes_per_array, cudaMemcpyHostToDevice));

  return true;
}

void free_table_host(CuckooTable &table) {
  if (!table.is_on_device) {
    free(table.keys_t1);
    free(table.vals_t1);
    free(table.keys_t2);
    free(table.vals_t2);
  }

  table.keys_t1 = nullptr;
  table.vals_t1 = nullptr;
  table.keys_t2 = nullptr;
  table.vals_t2 = nullptr;
}

void free_table_device(CuckooTable &table) {
  if (table.is_on_device) {
    cudaFree(table.keys_t1);
    cudaFree(table.vals_t1);
    cudaFree(table.keys_t2);
    cudaFree(table.vals_t2);
  }

  table.keys_t1 = nullptr;
  table.vals_t1 = nullptr;
  table.keys_t2 = nullptr;
  table.vals_t2 = nullptr;
}

void print_table_stats(const CuckooTable &table) {
  printf("\n");
  printf("┌─ Hash Table Statistics ───────────────────────────────────┐\n");
  printf("│ Location:        %-41s │\n",
         table.is_on_device ? "GPU (Device)" : "CPU (Host)");
  printf("│ Capacity:        %-41zu slots │\n", (size_t)table.capacity);
  printf("│ Entries:         %-41zu │\n", (size_t)table.count);
  printf("│ Load Factor:     %-40.2f%% │\n",
         (table.count * 100.0) / table.capacity);
  printf("│                                                           │\n");

  // Count entries in each table
  uint32_t count_t1 = 0, count_t2 = 0;

  if (!table.is_on_device) {
    for (uint32_t i = 0; i < table.capacity; i++) {
      if (table.keys_t1[i] != EMPTY_KEY)
        count_t1++;
      if (table.keys_t2[i] != EMPTY_KEY)
        count_t2++;
    }

    printf("│ Table 1 usage:   %8u / %-8u (%5.2f%%)              │\n", count_t1,
           table.capacity, (count_t1 * 100.0) / table.capacity);
    printf("│ Table 2 usage:   %8u / %-8u (%5.2f%%)              │\n", count_t2,
           table.capacity, (count_t2 * 100.0) / table.capacity);
  }

  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");
}

} // namespace kvstore
