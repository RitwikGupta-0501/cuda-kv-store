#include "core/hash.h"
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>

using namespace kvstore;

// Test 1: Basic functionality
bool test_basic() {
  std::cout << "Test 1: Basic functionality... ";

  uint32_t capacity = 1000;

  // Test that hashes are in range
  for (uint64_t key = 0; key < 100; key++) {
    uint32_t h1 = hash1(key, capacity);
    uint32_t h2 = hash2(key, capacity);

    if (h1 >= capacity || h2 >= capacity) {
      std::cout << "FAILED (out of range)\n";
      return false;
    }
  }

  std::cout << "PASSED\n";
  return true;
}

// Test 2: Determinism (same key = same hash)
bool test_determinism() {
  std::cout << "Test 2: Determinism... ";

  uint32_t capacity = 1000;

  for (uint64_t key = 0; key < 100; key++) {
    uint32_t h1_a = hash1(key, capacity);
    uint32_t h1_b = hash1(key, capacity);
    uint32_t h2_a = hash2(key, capacity);
    uint32_t h2_b = hash2(key, capacity);

    if (h1_a != h1_b || h2_a != h2_b) {
      std::cout << "FAILED (non-deterministic)\n";
      return false;
    }
  }

  std::cout << "PASSED\n";
  return true;
}

// Test 3: Distribution (low collision rate)
bool test_distribution() {
    std::cout << "Test 3: Distribution quality... ";
    
    uint32_t capacity = 100000;
    uint32_t num_keys = 10000;
    
    std::unordered_set<uint32_t> h1_values;
    std::unordered_set<uint32_t> h2_values;
    
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist;
    
    for (uint32_t i = 0; i < num_keys; i++) {
        uint64_t key = dist(rng);
        h1_values.insert(hash1(key, capacity));
        h2_values.insert(hash2(key, capacity));
    }
    
    double h1_unique_ratio = (double)h1_values.size() / num_keys;
    double h2_unique_ratio = (double)h2_values.size() / num_keys;
    
    // Realistic threshold: >93% is good for a hash function
    if (h1_unique_ratio < 0.93 || h2_unique_ratio < 0.93) {
        std::cout << "FAILED (too many collisions)\n";
        std::cout << "  h1 unique: " << h1_unique_ratio * 100 << "%\n";
        std::cout << "  h2 unique: " << h2_unique_ratio * 100 << "%\n";
        return false;
    }
    
    std::cout << "PASSED (h1: " << h1_unique_ratio * 100 << "%, h2: " 
              << h2_unique_ratio * 100 << "%)\n";
    return true;
}

// Test 4: Independence (h1 and h2 should not correlate)
bool test_independence() {
  std::cout << "Test 4: Hash independence... ";

  uint32_t capacity = 10000;

  // Count how often h1 == h2
  int same_count = 0;
  int total = 10000;

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist;

  for (int i = 0; i < total; i++) {
    uint64_t key = dist(rng);
    if (hash1(key, capacity) == hash2(key, capacity)) {
      same_count++;
    }
  }

  // Expected: ~1% same (1/capacity chance)
  double same_ratio = (double)same_count / total;
  double expected_ratio = 1.0 / capacity;

  // Allow 3x deviation
  if (same_ratio > expected_ratio * 3) {
    std::cout << "FAILED (functions too correlated)\n";
    std::cout << "  Same ratio: " << same_ratio * 100 << "%\n";
    std::cout << "  Expected: ~" << expected_ratio * 100 << "%\n";
    return false;
  }

  std::cout << "PASSED (collision rate: " << same_ratio * 100 << "%)\n";
  return true;
}

int main() {
  std::cout << "\n=== Hash Function Tests ===\n\n";

  bool all_passed = true;

  all_passed &= test_basic();
  all_passed &= test_determinism();
  all_passed &= test_distribution();
  all_passed &= test_independence();

  std::cout << "\n";
  if (all_passed) {
    std::cout << "✓ All tests PASSED\n\n";
    return 0;
  } else {
    std::cout << "✗ Some tests FAILED\n\n";
    return 1;
  }
}
