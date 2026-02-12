#include "core/config.h"
#include <iostream>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  std::cout << "Initializing GPU Key-Value Store...\n";

  kvstore::SystemConfig config;

  if (!kvstore::init_system_config(config)) {
    std::cerr << "Failed to initialize system configuration!\n";
    return 1;
  }

  kvstore::print_system_config(config);

  std::cout << "✓ Initialization complete\n";
  std::cout << "  Ready to proceed to hash table construction...\n\n";

  return 0;
}
