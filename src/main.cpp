#include "core/config.h"
#include "core/cuckoo.h"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  std::cout << "=== GPU Key-Value Store - Phase 1.4 ===\n\n";

  // Initialize system
  kvstore::SystemConfig config;
  if (!kvstore::init_system_config(config)) {
    std::cerr << "Failed to initialize system!\n";
    return 1;
  }

  kvstore::print_system_config(config);

  // Create hash table on CPU
  std::cout << "Creating hash table...\n";
  kvstore::CuckooTable host_table;

  if (!kvstore::create_table_host(host_table, config.hash_table_capacity)) {
    std::cerr << "Failed to create hash table!\n";
    return 1;
  }

  std::cout << "✓ Hash table created\n\n";

  // Insert test data (1 million entries)
  std::cout << "Inserting 1,000,000 test entries...\n";

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist(1, UINT64_MAX - 1);

  uint32_t num_inserts = 1000000;
  uint32_t failed = 0;

  for (uint32_t i = 0; i < num_inserts; i++) {
    uint64_t key = dist(rng);
    uint64_t value = key * 2; // Simple value: key * 2

    if (!kvstore::insert(host_table, key, value)) {
      failed++;
    }

    // Progress indicator
    if ((i + 1) % 100000 == 0) {
      printf("  Inserted %u / %u (%.1f%%)...\n", i + 1, num_inserts,
             ((i + 1) * 100.0) / num_inserts);
    }
  }

  std::cout << "\n✓ Insertion complete\n";
  std::cout << "  Success: " << (num_inserts - failed) << "\n";
  std::cout << "  Failed: " << failed << "\n\n";

  kvstore::print_table_stats(host_table);

  // Copy to GPU
  std::cout << "Copying table to GPU...\n";
  kvstore::CuckooTable device_table;

  if (!kvstore::copy_table_to_device(host_table, device_table)) {
    std::cerr << "Failed to copy to GPU!\n";
    kvstore::free_table_host(host_table);
    return 1;
  }

  std::cout << "✓ Table copied to GPU\n\n";
  kvstore::print_table_stats(device_table);

  // Cleanup
  kvstore::free_table_host(host_table);
  kvstore::free_table_device(device_table);

  std::cout << "✓ Phase 1.4 complete!\n";
  std::cout << "  Next: GPU lookup kernel\n\n";

  return 0;
}
