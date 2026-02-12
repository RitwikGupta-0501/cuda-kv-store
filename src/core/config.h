#pragma once

#include <cstddef>
#include <cstdint>

namespace kvstore {

// Dynamic limits (filled at runtime)
struct SystemConfig {
  // GPU resources
  size_t gpu_vram_total;
  size_t gpu_vram_free;
  size_t gpu_vram_usable; // 80% of free

  // CPU resources
  size_t cpu_ram_total;
  size_t cpu_ram_free;

  // Hash table sizing
  size_t hash_table_bytes;    // 70% of usable VRAM
  size_t hash_table_capacity; // Max number of entries
  size_t hash_table_entries;  // Actual entries (90% of capacity)

  // Batch buffers
  size_t batch_buffer_bytes; // 30% of usable VRAM
  size_t batch_size_max;     // Max queries per batch

  // GPU info
  char gpu_name[256];
  int gpu_compute_major;
  int gpu_compute_minor;
  int gpu_multiprocessors;
};

// Fixed tuning parameters
struct TuningParams {
  static constexpr size_t NUM_STREAMS = 3;
  static constexpr size_t BATCH_TIMEOUT_US = 500;
  static constexpr size_t CUCKOO_MAX_EVICTIONS = 100;
  static constexpr double CUCKOO_LOAD_FACTOR = 0.90;
  static constexpr double VRAM_SAFETY_MARGIN = 0.80; // Use 80% of free VRAM
  static constexpr double HASH_TABLE_RATIO = 0.70;   // 70% for hash table
  static constexpr double BATCH_BUFFER_RATIO = 0.30; // 30% for batching
};

// Initialize system configuration
bool init_system_config(SystemConfig &config);

// Print configuration summary
void print_system_config(const SystemConfig &config);

} // namespace kvstore
