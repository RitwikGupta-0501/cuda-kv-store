# WarpKV Architecture: GPU Memory Coalescing & Bucket Layout

## 1. Overview
The most severe performance penalty on a GPU is uncoalesced global memory access. When threads in a warp (32 threads) read from scattered memory addresses, the GPU memory controller is forced to issue multiple memory transactions. WarpKV's internal data structures are specifically engineered to guarantee **100% coalesced memory access** during table lookups and insertions.

## 2. The Bucket Structure
Instead of storing key-value pairs as an array of structs (`AoS`), which leads to divergent reads, WarpKV uses a Struct of Arrays (`SoA`) approach at the bucket level.

Each `Bucket` in the hash table is exactly **128 bytes** (a standard cache line size) and contains `SLOTS_PER_BUCKET` (typically 8) entries.

```cpp
struct alignas(128) Bucket {
    // 8 slots * 4 bytes = 32 bytes
    uint32_t keys[SLOTS_PER_BUCKET];    
    
    // 8 slots * 4 bytes = 32 bytes
    uint32_t values[SLOTS_PER_BUCKET];  
    
    // 8 slots * 1 byte = 8 bytes
    uint8_t fingerprint[SLOTS_PER_BUCKET]; 
    
    // 1 byte bitmap for occupancy
    uint8_t occupied_bitmap;            
    
    // 55 bytes padding to hit exactly 128 bytes
    uint8_t padding[55];                
};
```

## 3. Coalesced Reads
Because the `Bucket` is exactly 128 bytes and explicitly aligned (`alignas(128)`), reading an entire bucket fits perfectly into a single maximum-width memory transaction on modern NVIDIA GPUs.

When a warp performs a lookup, it does not have one thread read one slot. Instead:
1. The warp computes the target bucket address.
2. The threads cooperatively read the 32-byte `keys` array and the `occupied_bitmap` in a single coalesced burst.
3. The threads can then locally compare the target key against all 8 slots in registers (using warp-level primitives like `__ballot_sync` or just parallel independent checks).
4. Only if a match is found does the warp issue a second coalesced read for the 32-byte `values` array.

## 4. The Fingerprint Optimization
To further reduce memory bandwidth during cuckoo evictions, we store an 8-bit `fingerprint` (derived from the hash) for each slot.
When the GPU is searching for an empty slot or a victim to evict, it can read the 8-byte fingerprint array and the `occupied_bitmap` without ever loading the full 32-byte `keys` array. This allows the cuckoo chain logic to navigate the hash table using minimal memory bandwidth, only loading the full key/value when an eviction is actually finalized.

## 5. Padding and Cache Lines
The 55 bytes of padding might seem like wasted VRAM, but it is a critical optimization. 
If buckets were packed tightly (e.g., 73 bytes each), a single bucket would span across multiple 128-byte L2 cache lines. Reading it would require the GPU to fetch two separate cache lines. By padding to exactly 128 bytes, every bucket perfectly aligns with a single cache line, guaranteeing one transaction per read.
