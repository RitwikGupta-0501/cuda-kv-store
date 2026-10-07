import os
import re

def replace_in_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        content = re.sub(old, new, content)
        
    with open(filepath, 'w') as f:
        f.write(content)

# warpkv_device.cuh
cuh_replacements = [
    (r'#pragma once', '#pragma once\n\n#include <cstdint>\n#include <cstring>\n#include <cuda_runtime.h>\n\nnamespace warpkv {\n\nusing KeyT = uint32_t;\nusing ValueT = uint32_t;\n'),
    (r'#include <cstdint>\n#include <cstring>\n#include <cuda_runtime.h>\n\nnamespace warpkv {\n', ''), # Cleanup duplicate
    (r'uint32_t EMPTY_KEY', 'KeyT EMPTY_KEY'),
    (r'uint32_t LOCK_SENTINEL', 'KeyT LOCK_SENTINEL'),
    (r'uint32_t NOT_FOUND', 'ValueT NOT_FOUND'),
    (r'uint32_t keys\[8\];', 'KeyT keys[8];'),
    (r'uint32_t values\[8\];', 'ValueT values[8];'),
    (r'uint32_t key;\n    uint32_t value;', 'KeyT key;\n    ValueT value;'),
    (r'uint32_t warpkv_hash32\(uint32_t key\)', 'uint32_t warpkv_hash32(KeyT key)'),
    (r'uint32_t warpkv_hash32_host\(uint32_t key\)', 'uint32_t warpkv_hash32_host(KeyT key)'),
    (r'compute_hash_pair\(uint32_t key,', 'compute_hash_pair(KeyT key,'),
    (r'ValueT value; // \Found value', 'ValueT value; ///< Found value'),
    (r'uint32_t value; ///< Found value', 'ValueT value; ///< Found value'),
    (r'warp_lookup_device\(\n    BucketTable  table,\n    StashQueue\*  stash,\n    uint32_t     key,', 'warp_lookup_device(\n    BucketTable  table,\n    StashQueue*  stash,\n    KeyT         key,'),
    (r'warp_insert_device\(\n    BucketTable  table,\n    StashQueue\*  stash,\n    uint32_t\*    d_needs_rehash_flag,\n    uint32_t     key,\n    uint32_t     value,', 'warp_insert_device(\n    BucketTable  table,\n    StashQueue*  stash,\n    uint32_t*    d_needs_rehash_flag,\n    KeyT         key,\n    ValueT       value,'),
    (r'warp_delete_device\(\n    BucketTable  table,\n    StashQueue\*  stash,\n    uint32_t     key,', 'warp_delete_device(\n    BucketTable  table,\n    StashQueue*  stash,\n    KeyT         key,'),
    (r'uint32_t old_key = atomicCAS', 'KeyT old_key = atomicCAS'),
    (r'uint32_t current_key   = key;', 'KeyT current_key   = key;'),
    (r'uint32_t current_value = value;', 'ValueT current_value = value;'),
    (r'uint32_t victim_key = victim_bucket', 'KeyT victim_key = victim_bucket'),
    (r'uint32_t victim_value =\n                        \(\(volatile uint32_t\*\)', 'ValueT victim_value =\n                        ((volatile ValueT*)'),
    (r'uint32_t evicted_key      = 0;', 'KeyT evicted_key      = 0;'),
    (r'uint32_t evicted_value    = 0;', 'ValueT evicted_value    = 0;'),
    (r'uint32_t old_stash_key =', 'KeyT old_stash_key ='),
]
replace_in_file('include/warpkv/warpkv_device.cuh', cuh_replacements)

# GPU Header wrappers
for f in ['warp_lookup', 'cuckoo_insert', 'cuckoo_delete']:
    hdr = f"src/gpu/{f}.h"
    reps = [
        (r'const uint32_t\* __restrict__ keys', 'const KeyT* __restrict__ keys'),
        (r'const uint32_t\* __restrict__ values', 'const ValueT* __restrict__ values'),
        (r'uint32_t\*    values,', 'ValueT*       values,'),
        (r'const uint32_t\* key', 'const KeyT* key'),
        (r'uint32_t\* h_keys;', 'KeyT*     h_keys;'),
        (r'const uint32_t\* h_keys;', 'const KeyT* h_keys;'),
        (r'uint32_t\* h_values;', 'ValueT*   h_values;'),
        (r'uint32_t key = keys', 'KeyT key = keys'),
        (r'uint32_t value = values', 'ValueT value = values'),
    ]
    replace_in_file(hdr, reps)

# Entry points
reps = [
    (r'const uint32_t\* __restrict__ d_keys', 'const KeyT* __restrict__ d_keys'),
    (r'uint32_t\*                    d_values_out', 'ValueT*                      d_values_out'),
    (r'const uint32_t\* __restrict__ d_values', 'const ValueT* __restrict__ d_values'),
    (r'uint32_t key = d_keys', 'KeyT key = d_keys'),
    (r'uint32_t value  = d_values', 'ValueT value  = d_values'),
]
replace_in_file('include/warpkv/warpkv_entry.h', reps)
replace_in_file('src/gpu/warpkv_entry_kernels.cu', reps)

# Engine Header
reps = [
    (r'std::vector<uint32_t> values;', 'std::vector<ValueT> values;'),
    (r'uint32_t\*     h_keys_in', 'KeyT*         h_keys_in'),
    (r'uint32_t\*     h_values_in', 'ValueT*       h_values_in'),
    (r'uint32_t\*     h_values_out', 'ValueT*       h_values_out'),
    (r'uint32_t\*     d_keys_in', 'KeyT*         d_keys_in'),
    (r'uint32_t\*     d_values_in', 'ValueT*       d_values_in'),
    (r'uint32_t\*     d_values_out', 'ValueT*       d_values_out'),
    (r'const uint32_t\* keys', 'const KeyT* keys'),
    (r'const uint32_t\* values', 'const ValueT* values'),
    (r'uint32_t\*       values_out', 'ValueT*       values_out'),
]
replace_in_file('src/engine/warpkv_engine.h', reps)
print("Done refactoring headers.")
