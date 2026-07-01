#include <iostream>
#include <vector>
#include <map>
#include <random>

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, 0xFFFFFFFE);
    std::map<uint32_t, int> counts;
    int dups = 0;
    for (int i = 0; i < 100000; i++) {
        uint32_t k = dist(rng);
        dist(rng); // value
        if (counts[k]++ == 1) dups++;
    }
    std::cout << "Duplicates: " << dups << std::endl;
}
