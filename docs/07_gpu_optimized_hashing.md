# WarpKV Architecture: GPU-Optimized Hashing (xxhash3)

## 1. Overview
A hash table is only as fast as its hash function. For a GPU-based KV store, the hash function must not only exhibit excellent avalanche properties (to minimize collisions) but also execute extremely fast on the GPU's SIMT (Single Instruction, Multiple Thread) architecture. 

WarpKV uses a custom device-side port of **xxhash3** (specifically `XXH3_64bits`). While xxhash is typically heavily optimized for superscalar CPUs using vector intrinsics (AVX), adapting it for the GPU required specific architectural considerations.

## 2. Avoiding Warp Divergence
The most critical rule of GPU programming is avoiding warp divergence. If a hash function contains branches (e.g., `if (key > threshold)`), threads within a warp will diverge, forcing the GPU to serialize the execution paths. 

Our xxhash3 implementation is entirely **branchless**. It relies solely on a sequence of:
- Bitwise XORs (`^`)
- Bitwise Shifts (`>>`, `<<`)
- Multiplications (`*`)
- Byte-swapping (using CUDA intrinsics)

Because every thread executes the exact same sequence of instructions regardless of the input key, the GPU operates at peak arithmetic intensity.

## 3. Inline Execution & PTX Optimization
The hash function is marked with `__device__ __forceinline__`. 
- **Inline**: Prevents function call overhead and stack memory usage, which is expensive on a GPU.
- **PTX Translation**: Because it is purely arithmetic, the NVCC compiler can aggressively unroll the operations and translate them into highly optimized PTX assembly, pipelining the math alongside memory latency.

## 4. 64-bit Hash Splitting
WarpKV uses Cuckoo Hashing, which requires two distinct bucket indices (`b1` and `b2`) and an 8-bit fingerprint per key. Rather than hashing the key three separate times (which would waste ALUs), we hash the key *once* using the 64-bit xxhash3 and slice the result.

```cpp
// 64-bit hash generation
uint64_t hash = xxhash3_64(key);

// Slice 1: Primary Bucket (b1)
uint32_t b1 = hash & bucket_mask;

// Slice 2: Alternate Bucket (b2)
// We mix the upper 32 bits into the lower 32 bits, then mask
uint32_t b2 = (b1 ^ murmur_mix(hash >> 32)) & bucket_mask;

// Slice 3: Fingerprint (8 bits)
// We extract a byte from the upper half
uint8_t fingerprint = (hash >> 56) & 0xFF;
```

This ensures we get maximum entropy for bucket routing and fingerprinting from a single algorithmic pass.

## 5. Endianness and Intrinsics
xxhash is sensitive to endianness. While modern CPUs and GPUs are both Little Endian, porting CPU hash code requires replacing CPU byte-swap intrinsics (like `__builtin_bswap64`) with their CUDA equivalents (e.g., `__byte_perm` or PTX inline assembly) to ensure the hash generates deterministic results if we ever need to match them against host-side hashing.
