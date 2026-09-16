# WarpKV Architecture: GPU-Optimized Hashing

## 1. Overview
A hash table is only as fast as its hash function. For a GPU-based KV store, the hash function must not only exhibit excellent avalanche properties (to minimize collisions) but also execute extremely fast on the GPU's SIMT (Single Instruction, Multiple Thread) architecture. 

WarpKV uses a custom device-side **integer hash finalizer** derived from MurmurHash3's `fmix32`. While some stores use cryptographic or complex non-cryptographic hashes like CityHash or XXHash, adapting them for the GPU can introduce unnecessary register pressure and instruction overhead for simple 32-bit keys.

## 2. Avoiding Warp Divergence
The most critical rule of GPU programming is avoiding warp divergence. If a hash function contains branches (e.g., `if (key > threshold)`), threads within a warp will diverge, forcing the GPU to serialize the execution paths. 

Our hash implementation is entirely **branchless**. It relies solely on a sequence of:
- Bitwise XORs (`^`)
- Bitwise Shifts (`>>`)
- Multiplications (`*`)

Because every thread executes the exact same sequence of instructions regardless of the input key, the GPU operates at peak arithmetic intensity.

## 3. Inline Execution & PTX Optimization
The hash function is marked with `__device__ __forceinline__`. 
- **Inline**: Prevents function call overhead and stack memory usage, which is expensive on a GPU.
- **PTX Translation**: Because it is purely arithmetic, the NVCC compiler can aggressively unroll the operations and translate them into highly optimized PTX assembly, pipelining the math alongside memory latency.

## 4. Hash Pair Splitting (b1, b2, fingerprint)
WarpKV uses Cuckoo Hashing, which requires two distinct bucket indices (`b1` and `b2`) and an 8-bit fingerprint per key. Rather than hashing the key three completely separate times, we generate a primary 32-bit hash, then derive the secondary hash through an additional mixing step to ensure strong independence.

```cpp
// Generate primary 32-bit hash (fmix32 derived)
uint32_t h = warpkv_hash32(key);

// Slice 1: Primary Bucket (b1)
uint32_t b1 = h & bucket_mask;

// Slice 2: Fingerprint (8 bits)
uint8_t fingerprint = (uint8_t)(h >> 24);

// Slice 3: Alternate Bucket (b2)
// We run the primary hash through a second mixing step to ensure 
// b2 is fully independent of b1, preventing clustering.
uint32_t h2 = h;
h2 ^= h2 >> 16;
h2 *= 0x45d9f3bu;
h2 ^= h2 >> 16;
uint32_t b2 = h2 & bucket_mask;

// Collision guard: guarantee b1 != b2 for small tables
if (b2 == b1) {
    b2 = (b2 + 1) & bucket_mask;
}
```

This ensures we get maximum entropy for bucket routing and fingerprinting while keeping the ALU cost minimal on the GPU.
