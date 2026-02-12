#include "core/config.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>

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

bool init_system_config(SystemConfig &config) {
  // Zero out the struct
  memset(&config, 0, sizeof(SystemConfig));

  // === GPU Detection ===
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Store GPU info
  strncpy(config.gpu_name, prop.name, sizeof(config.gpu_name) - 1);
  config.gpu_compute_major = prop.major;
  config.gpu_compute_minor = prop.minor;
  config.gpu_multiprocessors = prop.multiProcessorCount;

  // Get VRAM info
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  config.gpu_vram_total = total_mem;
  config.gpu_vram_free = free_mem;
  config.gpu_vram_usable =
      (size_t)(free_mem * TuningParams::VRAM_SAFETY_MARGIN);

  // === CPU RAM Detection (Linux) ===
  struct sysinfo info;
  if (sysinfo(&info) != 0) {
    fprintf(stderr, "Failed to get system RAM info\n");
    return false;
  }

  config.cpu_ram_total = info.totalram * info.mem_unit;
  config.cpu_ram_free = info.freeram * info.mem_unit;

  // === Hash Table Sizing ===
  config.hash_table_bytes =
      (size_t)(config.gpu_vram_usable * TuningParams::HASH_TABLE_RATIO);

  // Each entry: 16 bytes (8-byte key + 8-byte value)
  // But cuckoo hashing needs TWO tables, so 32 bytes per entry
  size_t bytes_per_entry = 32;

  config.hash_table_capacity = config.hash_table_bytes / bytes_per_entry;

  // Apply load factor (90% full max)
  config.hash_table_entries =
      (size_t)(config.hash_table_capacity * TuningParams::CUCKOO_LOAD_FACTOR);

  // === Batch Buffer Sizing ===
  config.batch_buffer_bytes =
      (size_t)(config.gpu_vram_usable * TuningParams::BATCH_BUFFER_RATIO);

  // Each query needs:
  // - Input: 8 bytes (key) + 4 bytes (hash1) + 4 bytes (hash2) = 16 bytes
  // - Output: 8 bytes (value) + 1 byte (found flag) = 9 bytes
  // - Total per query: 25 bytes
  // - Triple buffering (3 streams): 75 bytes per query slot
  size_t bytes_per_query = 25 * TuningParams::NUM_STREAMS;

  config.batch_size_max = config.batch_buffer_bytes / bytes_per_query;

  // Sanity check: cap at 65536 (GPU grid dimension limits)
  if (config.batch_size_max > 65536) {
    config.batch_size_max = 65536;
  }

  return true;
}

void print_system_config(const SystemConfig &config) {
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║          GPU Key-Value Store - System Configuration       ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n");
  printf("\n");

  // GPU Info
  printf("┌─ GPU Information ─────────────────────────────────────────┐\n");
  printf("│ Device:              %-37s │\n", config.gpu_name);
  printf("│ Compute Capability:  %d.%-37d │\n", config.gpu_compute_major,
         config.gpu_compute_minor);
  printf("│ Multiprocessors:     %-37d │\n", config.gpu_multiprocessors);
  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");

  // Memory
  printf("┌─ Memory Resources ────────────────────────────────────────┐\n");
  printf("│ GPU VRAM Total:      %8.2f GB                         │\n",
         config.gpu_vram_total / 1e9);
  printf("│ GPU VRAM Free:       %8.2f GB                         │\n",
         config.gpu_vram_free / 1e9);
  printf("│ GPU VRAM Usable:     %8.2f GB (80%% of free)           │\n",
         config.gpu_vram_usable / 1e9);
  printf("│                                                           │\n");
  printf("│ CPU RAM Total:       %8.2f GB                         │\n",
         config.cpu_ram_total / 1e9);
  printf("│ CPU RAM Free:        %8.2f GB                         │\n",
         config.cpu_ram_free / 1e9);
  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");

  // Hash Table
  printf("┌─ Hash Table Configuration ────────────────────────────────┐\n");
  printf("│ Allocated VRAM:      %8.2f GB (70%% of usable)         │\n",
         config.hash_table_bytes / 1e9);
  printf("│ Table Capacity:      %8zu slots                       │\n",
         config.hash_table_capacity);
  printf("│ Max Entries:         %8zu entries (90%% load)         │\n",
         config.hash_table_entries);
  printf("│ Entry Size:          32 bytes (2 tables × 16 bytes)      │\n");
  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");

  // Batching
  printf("┌─ Batch Processing ────────────────────────────────────────┐\n");
  printf("│ Buffer VRAM:         %8.2f GB (30%% of usable)         │\n",
         config.batch_buffer_bytes / 1e9);
  printf("│ Max Batch Size:      %8zu queries                     │\n",
         config.batch_size_max);
  printf("│ Num Streams:         %8zu (triple buffering)          │\n",
         TuningParams::NUM_STREAMS);
  printf("│ Batch Timeout:       %8zu μs                          │\n",
         TuningParams::BATCH_TIMEOUT_US);
  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");

  // Summary
  printf("┌─ Summary ─────────────────────────────────────────────────┐\n");
  printf("│ This configuration can handle:                            │\n");
  printf("│   • Up to %zu million key-value pairs              │\n",
         config.hash_table_entries / 1000000);
  printf("│   • Batches of %zu queries processed in parallel     │\n",
         config.batch_size_max);
  printf("│   • ~%zu million queries/sec (estimated)             │\n",
         (config.batch_size_max * 1000) / 1000000); // Rough estimate
  printf("└───────────────────────────────────────────────────────────┘\n");
  printf("\n");
}

} // namespace kvstore
