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

/// Reserved key: value 0 cannot be inserted.
/// The lock protocol uses atomicCAS(key, EMPTY_KEY, LOCK_SENTINEL) to claim
/// empty slots, so key 0 is indistinguishable from an empty slot.
static constexpr uint32_t EMPTY_KEY = 0x00000000u;

/// Transient sentinel written to a key slot while a warp holds the lock.
/// No real key may ever have this value.
static constexpr uint32_t LOCK_SENTINEL = 0xFFFFFFFFu;

/// Value returned by lookup on a miss.
static constexpr uint32_t NOT_FOUND = 0xFFFFFFFFu;

// =============================================================================
// Bucket Structure — 128 bytes = 1 L2 cache line (AoS layout)
// =============================================================================

struct Bucket {
    /// Keys: 8 slots × uint32_t = 32 bytes
    uint32_t keys[8];

    /// Values: 8 slots × uint32_t = 32 bytes
    uint32_t values[8];

    /// Fingerprints: 8 slots × uint8_t = 8 bytes (fast-reject before key compare)
    uint8_t fingerprint[8];

    /// Occupancy bitmask: bit i is set when slot i is occupied = 4 bytes
    uint32_t occupancy_mask;

    /// Padding to fill a 128-byte L2 cache line = 52 bytes
    uint8_t _pad[52];
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
    uint32_t key;
    uint32_t value;
};

struct StashQueue {
    /// Atomically-incremented write index. Reset to 0 after each rehash drain.
    uint32_t head;

    /// Overflow entries. Sized to absorb burst collisions before rehash completes.
    StashEntry entries[STASH_CAPACITY];
};

static_assert(sizeof(StashQueue) < 300000, "StashQueue should be < 300 KB");

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

/// GPU device hash: Murmur3 fmix32 finalizer.
/// Excellent avalanche, no known bias at low moduli, minimal instruction count.
__device__ __forceinline__ uint32_t warpkv_hash32(uint32_t key) {
    uint32_t h = key + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16;
    return h;
}

/// Host-side equivalent of warpkv_hash32 for CPU preprocessing and testing.
inline uint32_t warpkv_hash32_host(uint32_t key) {
    uint32_t h = key + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16;
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
/// b1: primary hash masked to table size.
/// b2: independent second hash via an additional mixing step. If b1 == b2
///     (possible on very small tables), b2 is nudged by +1.
/// fingerprint: upper 8 bits of h — checked before full key comparison.
__device__ __host__ inline HashPair compute_hash_pair(uint32_t key, uint32_t bucket_mask) {
#ifdef __CUDA_ARCH__
    const uint32_t h = warpkv_hash32(key);
#else
    const uint32_t h = warpkv_hash32_host(key);
#endif

    // Independent secondary hash via an additional mixing round.
    uint32_t h2 = h;
    h2 ^= h2 >> 16;
    h2 *= 0x45d9f3bu;
    h2 ^= h2 >> 16;

    HashPair result;
    result.b1          = h  & bucket_mask;
    result.b2          = h2 & bucket_mask;
    result.fingerprint = (uint8_t)(h >> 24);

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
    uint32_t value; ///< Found value, or NOT_FOUND on miss.
    bool     found; ///< True iff the key was found.
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
    uint32_t     key,
    uint8_t      fingerprint)
{
    const HashPair hash_pair = compute_hash_pair(key, table.bucket_mask);
    Bucket* const  bucket_b1 = &table.buckets[hash_pair.b1];
    Bucket* const  bucket_b2 = &table.buckets[hash_pair.b2];

    const uint32_t lane_id = threadIdx.x % 32;

    LookupResult result = {NOT_FOUND, false};

    // ---- Lanes 0-7: scan bucket b1 ----------------------------------------
    if (lane_id < 8) {
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
    int found_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.found)) - 1;
    if (found_lane >= 0) {
        result.value = __shfl_sync(0xFFFFFFFFu, result.value, found_lane);
        result.found = true;
        return result;
    }

    // ---- Cooperative stash scan (all 32 lanes) -----------------------------
    if (stash != nullptr) {
        uint32_t stash_size = ((volatile uint32_t*)&stash->head)[0];
        if (stash_size > STASH_CAPACITY) stash_size = STASH_CAPACITY;

        for (uint32_t i = lane_id; i < stash_size; i += 32) {
            if (stash->entries[i].key == key) {
                result.value = stash->entries[i].value;
                result.found = true;
                break;
            }
        }

        found_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.found)) - 1;
        if (found_lane >= 0) {
            result.value = __shfl_sync(0xFFFFFFFFu, result.value, found_lane);
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
    uint32_t     key,
    uint32_t     value,
    uint8_t      fingerprint)
{
    const uint32_t lane_id = threadIdx.x % 32;
    InsertResult result = {INSERT_FAILED, 0, 0};

    uint32_t current_key   = key;
    uint32_t current_value = value;
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
        if (lane_id < 8) {
            const uint32_t slot     = lane_id;
            const uint32_t old_mask = bucket_b1->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                const uint32_t old_key = atomicCAS(&bucket_b1->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b1_claimed = true;
            }
        }

        const int b1_winner = __ffs(__ballot_sync(0xFFFFFFFFu, b1_claimed)) - 1;
        if (b1_claimed) {
            const uint32_t slot = lane_id;
            if (lane_id == (uint32_t)b1_winner) {
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
            const int success_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.status == INSERT_SUCCESS)) - 1;
            if (success_lane >= 0) {
                result.status    = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status,   success_lane);
                result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, success_lane);
                result.hops      = __shfl_sync(0xFFFFFFFFu, result.hops,      success_lane);
                return result;
            }
        }

        // ---- Try bucket b2 (lanes 8-15) ------------------------------------
        bool b2_claimed = false;
        if (lane_id >= 8 && lane_id < 16) {
            const uint32_t slot     = lane_id - 8;
            const uint32_t old_mask = bucket_b2->occupancy_mask;
            if (!(old_mask & (1u << slot))) {
                const uint32_t old_key = atomicCAS(&bucket_b2->keys[slot], EMPTY_KEY, LOCK_SENTINEL);
                if (old_key == EMPTY_KEY) b2_claimed = true;
            }
        }

        const int b2_winner = __ffs(__ballot_sync(0xFFFFFFFFu, b2_claimed)) - 1;
        if (b2_claimed) {
            const uint32_t slot = lane_id - 8;
            if (lane_id == (uint32_t)b2_winner) {
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
            const int success_lane = __ffs(__ballot_sync(0xFFFFFFFFu, result.status == INSERT_SUCCESS)) - 1;
            if (success_lane >= 0) {
                result.status    = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status,   success_lane);
                result.slot_used = __shfl_sync(0xFFFFFFFFu, result.slot_used, success_lane);
                result.hops      = __shfl_sync(0xFFFFFFFFu, result.hops,      success_lane);
                return result;
            }
        }

        // ---- Both full: evict a victim (lane 0 only) -----------------------
        bool     eviction_success = false;
        uint32_t evicted_key      = 0;
        uint32_t evicted_value    = 0;

        if (lane_id == 0) {
            const uint32_t victim_slot =
                (hash_pair.b1 ^ hash_pair.b2 ^ hop_count ^ contention_count) % 8;
            Bucket* victim_bucket =
                ((hop_count ^ contention_count) % 2 == 0) ? bucket_b1 : bucket_b2;

            const uint32_t victim_key = victim_bucket->keys[victim_slot];
            if (victim_key != EMPTY_KEY && victim_key != LOCK_SENTINEL) {
                const uint32_t old_key =
                    atomicCAS(&victim_bucket->keys[victim_slot], victim_key, LOCK_SENTINEL);
                if (old_key == victim_key) {
                    // Force L2 read to avoid stale L1 from other SMs.
                    const uint32_t victim_value =
                        ((volatile uint32_t*)victim_bucket->values)[victim_slot];
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

        eviction_success = __shfl_sync(0xFFFFFFFFu, eviction_success, 0);
        if (eviction_success) {
            current_key   = __shfl_sync(0xFFFFFFFFu, evicted_key,   0);
            current_value = __shfl_sync(0xFFFFFFFFu, evicted_value, 0);
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

    result.status = (InsertStatus)__shfl_sync(0xFFFFFFFFu, (uint32_t)result.status, 0);
    result.hops   = __shfl_sync(0xFFFFFFFFu, result.hops, 0);
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
    uint32_t     key,
    uint8_t      fingerprint)
{
    const uint32_t lane_id   = threadIdx.x % 32;
    const HashPair hash_pair = compute_hash_pair(key, table.bucket_mask);
    Bucket* const  bucket_b1 = &table.buckets[hash_pair.b1];
    Bucket* const  bucket_b2 = &table.buckets[hash_pair.b2];

    bool     found         = false;
    uint32_t slot          = 0;
    Bucket*  target_bucket = nullptr;

    // ---- Lanes 0-7: scan b1 -----------------------------------------------
    if (lane_id < 8) {
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
    if (lane_id >= 8 && lane_id < 16) {
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

    const int winner = __ffs(__ballot_sync(0xFFFFFFFFu, found)) - 1;

    bool delete_success = false;
    if (winner >= 0) {
        if (lane_id == (uint32_t)winner) {
            const uint32_t old_key =
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
        delete_success = __shfl_sync(0xFFFFFFFFu, delete_success, winner);
        return delete_success;
    }

    // ---- Stash scan (lane 0 only — single-threaded, known limitation) ------
    if (lane_id == 0 && stash != nullptr) {
        const uint32_t current_head = *(volatile uint32_t*)&stash->head;
        const uint32_t count        = current_head < STASH_CAPACITY ? current_head : STASH_CAPACITY;
        for (uint32_t i = 0; i < count; ++i) {
            if (stash->entries[i].key == key) {
                const uint32_t old_stash_key =
                    atomicCAS(&stash->entries[i].key, key, EMPTY_KEY);
                if (old_stash_key == key) {
                    delete_success = true;
                    break;
                }
            }
        }
    }

    delete_success = __shfl_sync(0xFFFFFFFFu, delete_success, 0);
    return delete_success;
}

#endif // __CUDACC__

} // namespace warpkv
