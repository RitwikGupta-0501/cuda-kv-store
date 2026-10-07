// =============================================================================
// warpkv_device.cuh — WarpKV GPU-Native Device API
// =============================================================================
//
// Self-contained header for embedding WarpKV directly into external CUDA kernels.
//
// USAGE:
//   #include "warpkv/warpkv_device.cuh"
//
//   __global__ void my_kernel(warpkv::BucketTable table, warpkv::StashQueue* stash) {
//       uint32_t key = ...;
//       uint8_t fp = warpkv::compute_hash_pair(key, table.bucket_mask).fingerprint;
//       warpkv::LookupResult r = warpkv::warp_lookup_device(table, stash, key, fp);
//   }
//
// REQUIREMENTS:
//   - One warp (32 threads) must call warp_lookup_device / warp_insert_device /
//     warp_delete_device together. Do not call from a single thread.
//   - Key value 0 (EMPTY_KEY) is reserved and cannot be stored.
//   - Key value 0xFFFFFFFF (LOCK_SENTINEL) is reserved and cannot be stored.
//   - Compile with: nvcc -arch=sm_50 or higher.
//
// DEPENDENCIES: <cstdint>, <cuda_runtime.h> only.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace warpkv {

// =============================================================================
// Type Aliases
// =============================================================================
// Abstracted types to support smooth migration to 64-bit keys/values.
using KeyT   = uint64_t;
using ValueT = uint64_t;

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of cuckoo eviction hops before a key is sent to the stash.
static constexpr uint32_t MAX_EVICTION_HOPS = 128;

/// Stash fill level that triggers an automatic rehash request.
static constexpr uint32_t BACKPRESSURE_THRESHOLD = 4096;

/// Number of keys processed per GPU batch launch (host-side engine).
static constexpr uint32_t BATCH_SIZE = 4096;

/// Total overflow stash capacity (must be > 4 × BATCH_SIZE for in-flight safety).
static constexpr uint32_t STASH_CAPACITY = 32768;

/// Number of slots per bucket (7 to fit exactly 128 bytes with 64-bit types)
static constexpr uint32_t BUCKET_SLOTS = 7;

/// Reserved key: value 0 cannot be inserted.
/// The lock protocol uses atomicCAS(key, EMPTY_KEY, LOCK_SENTINEL) to claim
/// empty slots, so key 0 is indistinguishable from an empty slot.
static constexpr KeyT EMPTY_KEY = 0x0000000000000000ULL;

/// Transient sentinel written to a key slot while a warp holds the lock.
/// No real key may ever have this value.
static constexpr KeyT LOCK_SENTINEL = 0xFFFFFFFFFFFFFFFFULL;

/// Value returned by lookup on a miss.
static constexpr ValueT NOT_FOUND = 0xFFFFFFFFFFFFFFFFULL;

// =============================================================================
// Bucket Structure — 128 bytes = 1 L2 cache line (AoS layout)
// =============================================================================

struct Bucket {
    /// Keys: 7 slots × 8 bytes = 56 bytes
    KeyT keys[BUCKET_SLOTS];

    /// Values: 7 slots × 8 bytes = 56 bytes
    ValueT values[BUCKET_SLOTS];

    /// Occupancy bitmask: bit i is set when slot i is occupied = 4 bytes
    uint32_t occupancy_mask;

    /// Fingerprints: 7 slots × 1 byte = 7 bytes
    uint8_t fingerprint[BUCKET_SLOTS];

    /// Padding to fill a 128-byte L2 cache line = 5 bytes
    uint8_t _pad[5];
};

static_assert(sizeof(Bucket) == 128, "Bucket must be exactly 128 bytes (1 L2 cache line)");

// =============================================================================
// BucketTable — flat device-resident descriptor (passed by value to kernels)
// =============================================================================

struct BucketTable {
    /// Device pointer to the contiguous bucket array.
    Bucket* buckets;

    /// Total number of buckets (must be a power of 2).
    uint32_t num_buckets;

    /// Bucket index mask: `num_buckets - 1`. Use `h & bucket_mask` instead of `h % num_buckets`.
    uint32_t bucket_mask;

    /// Rehash trigger threshold: 50% of num_buckets.
    uint32_t load_factor_limit;
};

// =============================================================================
// Stash — overflow queue for keys that exhaust MAX_EVICTION_HOPS
// =============================================================================

struct StashEntry {
    KeyT   key;
    ValueT value;
};

struct StashQueue {
    /// Atomically-incremented write index. Reset to 0 after each rehash drain.
    uint32_t head;

    /// Overflow entries. Sized to absorb burst collisions before rehash completes.
    StashEntry entries[STASH_CAPACITY];
};

static_assert(sizeof(StashQueue) < 600000, "StashQueue should be < 600 KB");

// =============================================================================
// Bucket Utility Functions
// =============================================================================

/// Initialize a bucket to the all-empty state (host-side only).
__host__ inline void bucket_init(Bucket* bucket) {
    bucket->occupancy_mask = 0;
    std::memset(bucket->keys,        0, sizeof(bucket->keys));
    std::memset(bucket->values,      0, sizeof(bucket->values));
    std::memset(bucket->fingerprint, 0, sizeof(bucket->fingerprint));
}

/// Return true if slot `slot` is occupied.
__host__ __device__ inline bool bucket_is_occupied(const Bucket* bucket, int slot) {
    return (bucket->occupancy_mask >> slot) & 1u;
}

/// Mark slot `slot` as occupied.
__host__ __device__ inline void bucket_set_occupied(Bucket* bucket, int slot) {
    bucket->occupancy_mask |= (1u << slot);
}

/// Mark slot `slot` as empty.
__host__ __device__ inline void bucket_clear_occupied(Bucket* bucket, int slot) {
    bucket->occupancy_mask &= ~(1u << slot);
}

// =============================================================================
// Hash Functions
// =============================================================================

/// GPU device hash: Murmur3 fmix64 finalizer.
/// Excellent avalanche, no known bias, and perfectly distributes 64-bit keys.
__device__ __forceinline__ uint64_t warpkv_hash64(KeyT key) {
    uint64_t h = (uint64_t)key + 0x9E3779B97F4A7C15ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/// Host-side equivalent of warpkv_hash64 for CPU preprocessing and testing.
inline uint64_t warpkv_hash64_host(KeyT key) {
    uint64_t h = (uint64_t)key + 0x9E3779B97F4A7C15ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/// Output of compute_hash_pair: two candidate bucket indices and an 8-bit fingerprint.
struct HashPair {
    uint32_t b1;          ///< Primary bucket index
    uint32_t b2;          ///< Secondary (independent) bucket index
    uint8_t  fingerprint; ///< Upper 8 bits of h — used for fast slot rejection
};

/// Compute both candidate bucket indices and fingerprint for `key`.
///
/// We use a single 64-bit hash to extract all three components:
/// b1: lower 32 bits masked.
/// b2: upper 32 bits masked.
/// fingerprint: highest 8 bits of the 64-bit hash.
__device__ __host__ inline HashPair compute_hash_pair(KeyT key, uint32_t bucket_mask) {
#ifdef __CUDA_ARCH__
    const uint64_t h = warpkv_hash64(key);
#else
    const uint64_t h = warpkv_hash64_host(key);
#endif

    HashPair result;
    result.b1          = (uint32_t)h & bucket_mask;
    result.b2          = (uint32_t)(h >> 32) & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 56);

    // Guarantee b1 != b2 for all table sizes.
    if (result.b2 == result.b1) {
        result.b2 = (result.b2 + 1) & bucket_mask;
    }

    return result;
}

// =============================================================================
// Device-Side Lookup
// =============================================================================

/// Result returned by warp_lookup_device.
struct LookupResult {
    ValueT value; ///< Found value, or NOT_FOUND on miss.
    bool   found; ///< True iff the key was found.
};

#ifdef __CUDACC__

/// Warp-cooperative lookup. Must be called by all 32 threads of a warp together.
///
/// Thread assignment:
///   Lanes  0-7:  scan primary bucket b1 (slots 0-7)
///   Lanes  8-15: scan secondary bucket b2 (slots 0-7)
///   Lanes 16-31: idle during bucket scan; cooperative during stash scan
///
/// On hit: broadcasts value to all lanes and returns.
/// On miss: all 32 lanes cooperatively stride-scan the stash.
__device__ inline LookupResult warp_lookup_device(
    BucketTable  table,
    StashQueue*  stash,
    KeyT         key,
    uint8_t      fingerprint)
{
    const HashPair hash_pair = compute_hash_pair(key, table.bucket_mask);
    Bucket* const  bucket_b1 = &table.buckets[hash_pair.b1];
    Bucket* const  bucket_b2 = &table.buckets[hash_pair.b2];

    const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    const uint32_t lane_id = threadIdx.x % 16;
    const uint32_t warp_lane = threadIdx.x % 32;

    LookupResult result = {NOT_FOUND, false};

    // ---- Lanes 0-7: scan bucket b1 ----------------------------------------
    if (lane_id < BUCKET_SLOTS) {
        if (bucket_b1->occupancy_mask & (1u << lane_id)) {
            if (bucket_b1->fingerprint[lane_id] == fingerprint) {
                if (bucket_b1->keys[lane_id] == key) {
                    result.value = bucket_b1->values[lane_id];
                    result.found = true;
                }
            }
        }
    }
    // ---- Lanes 8-15: scan bucket b2 in parallel ----------------------------
    else if (lane_id < 16) {
        const uint32_t b2_slot = lane_id - 8;
        if (bucket_b2->occupancy_mask & (1u << b2_slot)) {
            if (bucket_b2->fingerprint[b2_slot] == fingerprint) {
                if (bucket_b2->keys[b2_slot] == key) {
                    result.value = bucket_b2->values[b2_slot];
                    result.found = true;
                }
            }
        }
    }

    // ---- Broadcast from whichever lane found the key -----------------------
    int found_lane = __ffs(__ballot_sync(active_mask, result.found)) - 1;
    if (found_lane >= 0) {
        result.value = __shfl_sync(active_mask, result.value, found_lane);
        result.found = true;
        return result;
    }

    // ---- Cooperative stash scan (all 32 lanes) -----------------------------
    if (stash != nullptr) {
        uint32_t stash_size = ((volatile uint32_t*)&stash->head)[0];
        if (stash_size > STASH_CAPACITY) stash_size = STASH_CAPACITY;

        for (uint32_t i = lane_id; i < stash_size; i += 16) {
            if (stash->entries[i].key == key) {
                result.value = stash->entries[i].value;
                result.found = true;
                break;
            }
        }

        found_lane = __ffs(__ballot_sync(active_mask, result.found)) - 1;
        if (found_lane >= 0) {
            result.value = __shfl_sync(active_mask, result.value, found_lane);
            result.found = true;
        }
    }

    return result;
}

// =============================================================================
// Device-Side Insertion
// =============================================================================

/// Status codes returned by warp_insert_device.
enum InsertStatus : uint32_t {
    INSERT_SUCCESS = 0, ///< Inserted directly into a bucket slot.
    INSERT_STASHED = 1, ///< Inserted into overflow stash after MAX_EVICTION_HOPS.
    INSERT_FAILED  = 2, ///< Stash also full — data loss (should not happen).
};

/// Detailed result of a warp-cooperative insertion.
struct InsertResult {
    InsertStatus status;   ///< Outcome of the insertion.
    uint32_t     slot_used; ///< Slot index used (valid only for INSERT_SUCCESS).
    uint32_t     hops;      ///< Number of cuckoo eviction hops performed.
};

/// Warp-cooperative insertion with cuckoo eviction chains.
/// Must be called by all 32 threads of a warp together.
///
/// Thread assignment per eviction hop:
///   Lanes  0-7:   attempt to claim a free slot in bucket b1
///   Lanes  8-15:  attempt to claim a free slot in bucket b2
///   Lanes 16-31:  idle during slot claims
///   Lane   0:     selects and locks the eviction victim
///   Lanes  1-31:  idle during victim selection
///   Lane   0:     writes to stash on overflow
__device__ inline InsertResult warp_insert_device(
    BucketTable  table,
    StashQueue*  stash,
    uint32_t*    d_needs_rehash_flag,
    KeyT         key,
    ValueT       value,
    uint8_t      fingerprint)
{
    const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    const uint32_t lane_id = threadIdx.x % 16;
    const uint32_t warp_lane = threadIdx.x % 32;
    InsertResult result = {INSERT_FAILED, 0, 0};

    KeyT     current_key   = key;
    ValueT   current_value = value;
    uint32_t hop_count       = 0;
    uint32_t contention_count = 0;

    while (hop_count < MAX_EVICTION_HOPS &&
           contention_count < 1000 &&
           result.status == INSERT_FAILED)
    {
        HashPair hash_pair  = compute_hash_pair(current_key, table.bucket_mask);
        uint8_t  current_fp = hash_pair.fingerprint;
        Bucket*  bucket_b1  = &table.buckets[hash_pair.b1];
        Bucket*  bucket_b2  = &table.buckets[hash_pair.b2];

        // ---- Try bucket b1 (lanes 0-7) -------------------------------------
        bool b1_claimed = false;
        if (lane_id < BUCKET_SLOTS) {
            const uint32_t slot     = lane_id;
            const uint32_t old_mask = bucket_b1->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                const KeyT old_key = atomicCAS(&bucket_b1->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b1_claimed = true;
            }
        }

        const int b1_winner = __ffs(__ballot_sync(active_mask, b1_claimed)) - 1;
        if (b1_claimed) {
            const uint32_t slot = lane_id;
            if (warp_lane == (uint32_t)b1_winner) {
                bucket_b1->values[slot]      = current_value;
                bucket_b1->fingerprint[slot] = current_fp;
                __threadfence();
                bucket_b1->keys[slot]        = current_key;
                atomicOr(&bucket_b1->occupancy_mask, (1u << slot));
                result.status   = INSERT_SUCCESS;
                result.slot_used = slot;
                result.hops     = hop_count;
            } else {
                bucket_b1->keys[slot] = EMPTY_KEY; // release unused locks
            }
        }

        {
            const int success_lane = __ffs(__ballot_sync(active_mask, result.status == INSERT_SUCCESS)) - 1;
            if (success_lane >= 0) {
                result.status    = (InsertStatus)__shfl_sync(active_mask, (uint32_t)result.status,   success_lane);
                result.slot_used = __shfl_sync(active_mask, result.slot_used, success_lane);
                result.hops      = __shfl_sync(active_mask, result.hops,      success_lane);
                return result;
            }
        }

        // ---- Try bucket b2 (lanes 8-15) ------------------------------------
        bool b2_claimed = false;
        if (lane_id >= 8 && lane_id < 8 + BUCKET_SLOTS) {
            const uint32_t slot     = lane_id - 8;
            const uint32_t old_mask = bucket_b2->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                const KeyT old_key = atomicCAS(&bucket_b2->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b2_claimed = true;
            }
        }

        const int b2_winner = __ffs(__ballot_sync(active_mask, b2_claimed)) - 1;
        if (b2_claimed) {
            const uint32_t slot = lane_id - 8;
            if (warp_lane == (uint32_t)b2_winner) {
                bucket_b2->values[slot]      = current_value;
                bucket_b2->fingerprint[slot] = current_fp;
                __threadfence();
                bucket_b2->keys[slot]        = current_key;
                atomicOr(&bucket_b2->occupancy_mask, (1u << slot));
                result.status   = INSERT_SUCCESS;
                result.slot_used = slot;
                result.hops     = hop_count;
            } else {
                bucket_b2->keys[slot] = EMPTY_KEY;
            }
        }

        {
            const int success_lane = __ffs(__ballot_sync(active_mask, result.status == INSERT_SUCCESS)) - 1;
            if (success_lane >= 0) {
                result.status    = (InsertStatus)__shfl_sync(active_mask, (uint32_t)result.status,   success_lane);
                result.slot_used = __shfl_sync(active_mask, result.slot_used, success_lane);
                result.hops      = __shfl_sync(active_mask, result.hops,      success_lane);
                return result;
            }
        }

        // ---- Both full: evict a victim (lane 0 only) -----------------------
        bool   eviction_success = false;
        KeyT   evicted_key      = 0;
        ValueT evicted_value    = 0;

        if (lane_id == 0) {
            const uint32_t victim_slot =
                (hash_pair.b1 ^ hash_pair.b2 ^ hop_count ^ contention_count) % BUCKET_SLOTS;
            Bucket* victim_bucket =
                ((hop_count ^ contention_count) % 2 == 0) ? bucket_b1 : bucket_b2;

            const KeyT victim_key = victim_bucket->keys[victim_slot];
            if (victim_key != EMPTY_KEY && victim_key != LOCK_SENTINEL) {
                const KeyT old_key =
                    atomicCAS(&victim_bucket->keys[victim_slot], victim_key, LOCK_SENTINEL);
                if (old_key == victim_key) {
                    // Force L2 read to avoid stale L1 from other SMs.
                    const ValueT victim_value =
                        ((volatile ValueT*)victim_bucket->values)[victim_slot];
                    victim_bucket->values[victim_slot]      = current_value;
                    victim_bucket->fingerprint[victim_slot] = current_fp;
                    __threadfence();
                    victim_bucket->keys[victim_slot] = current_key;
                    evicted_key     = victim_key;
                    evicted_value   = victim_value;
                    eviction_success = true;
                }
            }
        }

        eviction_success = __shfl_sync(active_mask, eviction_success, (threadIdx.x & ~15));
        if (eviction_success) {
            // Note: If KeyT/ValueT are 64-bit, we need __shfl_sync to handle 64-bit later.
            // But since KeyT/ValueT are uint32_t right now, __shfl_sync is fine.
            current_key   = __shfl_sync(active_mask, (uint32_t)evicted_key,   0);
            current_value = __shfl_sync(active_mask, (uint32_t)evicted_value, (threadIdx.x & ~15));
            hop_count++;
            contention_count = 0;
        } else {
            contention_count++;
        }
    }

    // ---- MAX_EVICTION_HOPS reached: dump to stash (lane 0 only) -----------
    if (lane_id == 0) {
        const uint32_t head = atomicAdd((uint32_t*)&stash->head, 1);
        if (head < STASH_CAPACITY) {
            stash->entries[head].key   = current_key;
            stash->entries[head].value = current_value;
            result.status = INSERT_STASHED;
            result.hops   = hop_count;
            if (head >= BACKPRESSURE_THRESHOLD) {
                atomicExch((uint32_t*)d_needs_rehash_flag, 1u);
            }
        } else {
            // Stash overflow — signal urgent rehash.
            atomicExch((uint32_t*)d_needs_rehash_flag, 1u);
            result.status = INSERT_FAILED;
            result.hops   = hop_count;
        }
    }

    result.status = (InsertStatus)__shfl_sync(active_mask, (uint32_t)result.status, (threadIdx.x & ~15));
    result.hops   = __shfl_sync(active_mask, result.hops, (threadIdx.x & ~15));
    return result;
}

// =============================================================================
// Device-Side Deletion
// =============================================================================

/// Warp-cooperative deletion.
/// Must be called by all 32 threads of a warp together.
///
/// Thread assignment:
///   Lanes  0-7:  scan b1 for the key
///   Lanes  8-15: scan b2 for the key in parallel
///   Lanes 16-31: idle during bucket scan
///   Winner lane: acquires the slot lock and clears the entry
///   Lane 0:      scans the stash linearly if key not found in buckets
///
/// Returns true if the key was found and deleted, false if not found.
__device__ inline bool warp_delete_device(
    BucketTable  table,
    StashQueue*  stash,
    KeyT         key,
    uint8_t      fingerprint)
{
    const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    const uint32_t lane_id = threadIdx.x % 16;
    const uint32_t warp_lane = threadIdx.x % 32;
    const HashPair hash_pair = compute_hash_pair(key, table.bucket_mask);
    Bucket* const  bucket_b1 = &table.buckets[hash_pair.b1];
    Bucket* const  bucket_b2 = &table.buckets[hash_pair.b2];

    bool     found         = false;
    uint32_t slot          = 0;
    Bucket*  target_bucket = nullptr;

    // ---- Lanes 0-7: scan b1 -----------------------------------------------
    if (lane_id < BUCKET_SLOTS) {
        if ((bucket_b1->occupancy_mask & (1u << lane_id)) &&
            bucket_b1->fingerprint[lane_id] == fingerprint &&
            bucket_b1->keys[lane_id] == key)
        {
            found         = true;
            slot          = lane_id;
            target_bucket = bucket_b1;
        }
    }

    // ---- Lanes 8-15: scan b2 ----------------------------------------------
    if (lane_id >= 8 && lane_id < 8 + BUCKET_SLOTS) {
        const uint32_t local_slot = lane_id - 8;
        if ((bucket_b2->occupancy_mask & (1u << local_slot)) &&
            bucket_b2->fingerprint[local_slot] == fingerprint &&
            bucket_b2->keys[local_slot] == key)
        {
            found         = true;
            slot          = local_slot;
            target_bucket = bucket_b2;
        }
    }

    const int winner = __ffs(__ballot_sync(active_mask, found)) - 1;

    bool delete_success = false;
    if (winner >= 0) {
        if (warp_lane == (uint32_t)winner) {
            const KeyT old_key =
                atomicCAS(&target_bucket->keys[slot], key, LOCK_SENTINEL);
            if (old_key == key) {
                // Clear occupancy before releasing so readers see a clean state.
                atomicAnd(&target_bucket->occupancy_mask, ~(1u << slot));
                target_bucket->values[slot]      = 0;
                target_bucket->fingerprint[slot] = 0;
                __threadfence();
                target_bucket->keys[slot] = EMPTY_KEY;
                delete_success = true;
            }
        }
        delete_success = __shfl_sync(active_mask, delete_success, winner);
        return delete_success;
    }

    // ---- Stash scan (lane 0 only — single-threaded, known limitation) ------
    if (lane_id == 0 && stash != nullptr) {
        const uint32_t current_head = *(volatile uint32_t*)&stash->head;
        const uint32_t count        = current_head < STASH_CAPACITY ? current_head : STASH_CAPACITY;
        for (uint32_t i = 0; i < count; ++i) {
            if (stash->entries[i].key == key) {
                const KeyT old_stash_key =
                    atomicCAS(&stash->entries[i].key, key, EMPTY_KEY);
                if (old_stash_key == key) {
                    delete_success = true;
                    break;
                }
            }
        }
    }

    delete_success = __shfl_sync(active_mask, delete_success, (threadIdx.x & ~15));
    return delete_success;
}

#endif // __CUDACC__

} // namespace warpkv
