#pragma once
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Fast approximation of Zipfian distribution used in YCSB
class ZipfianGenerator {
private:
    uint64_t n;
    double theta;
    double alpha;
    double zetan;
    double eta;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist;

    double zeta(uint64_t n, double theta) {
        double sum = 0.0;
        for (uint64_t i = 1; i <= n; i++) {
            sum += 1.0 / std::pow((double)i, theta);
        }
        return sum;
    }

public:
    ZipfianGenerator(uint64_t _n, double _theta = 0.99, uint64_t seed = 42) 
        : n(_n), theta(_theta), rng(seed), dist(0.0, 1.0) {
        if (n == 0) throw std::invalid_argument("n must be > 0");
        zetan = zeta(n, theta);
        alpha = 1.0 / (1.0 - theta);
        eta = (1.0 - std::pow(2.0 / n, 1.0 - theta)) / (1.0 - zeta(2, theta) / zetan);
    }

    uint64_t next() {
        double u = dist(rng);
        double uz = u * zetan;
        if (uz < 1.0) return 1;
        if (uz < 1.0 + std::pow(0.5, theta)) return 2;
        return 1 + (uint64_t)(n * std::pow(eta * u - eta + 1.0, alpha));
    }
};

// Scatters the Zipfian distribution randomly across the key space.
// This prevents sequential keys from being grouped into the same cache lines,
// properly stressing the L1/L2 caches like YCSB.
class ScrambledZipfianGenerator {
private:
    ZipfianGenerator zipf;
    uint64_t min_val;
    uint64_t max_val;
    uint64_t item_count;

    // Fast FNV-1a hash to scramble the integer
    uint64_t fnv1a(uint64_t val) {
        uint64_t hash = 0xcbf29ce484222325ull;
        for (int i = 0; i < 8; i++) {
            hash ^= (val & 0xFF);
            hash *= 0x100000001b3ull;
            val >>= 8;
        }
        return hash;
    }

public:
    ScrambledZipfianGenerator(uint64_t _min, uint64_t _max, double _theta = 0.99, uint64_t seed = 42)
        : zipf(_max - _min + 1, _theta, seed), min_val(_min), max_val(_max) {
        if (_max < _min) throw std::invalid_argument("max must be >= min");
        item_count = _max - _min + 1;
    }

    uint64_t next() {
        uint64_t z = zipf.next(); // 1 to item_count
        return min_val + (fnv1a(z) % item_count);
    }
};
