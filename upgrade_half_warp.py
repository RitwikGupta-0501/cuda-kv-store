import re
import sys

def replace_in_file(path, replacements):
    with open(path, 'r') as f:
        c = f.read()
    for old, new in replacements:
        if old not in c:
            print(f"Failed to find {old[:40]} in {path}")
        c = c.replace(old, new)
    with open(path, 'w') as f:
        f.write(c)

replace_in_file('src/gpu/warpkv_entry_kernels.cu', [
    ('blockDim.x / 32', 'blockDim.x / 16'),
    ('threadIdx.x / 32', 'threadIdx.x / 16'),
    ('(threadIdx.x % 32) == 0', '(threadIdx.x % 16) == 0'),
])

replace_in_file('src/gpu/warp_lookup.h', [
    ('blockDim.x / 32', 'blockDim.x / 16'),
    ('threadIdx.x / 32', 'threadIdx.x / 16'),
    ('(threadIdx.x % 32) == 0', '(threadIdx.x % 16) == 0'),
])

replace_in_file('src/gpu/cuckoo_insert.h', [
    ('blockDim.x / 32', 'blockDim.x / 16'),
    ('threadIdx.x / 32', 'threadIdx.x / 16'),
    ('(threadIdx.x % 32) == 0', '(threadIdx.x % 16) == 0'),
])

replace_in_file('src/gpu/cuckoo_delete.h', [
    ('blockDim.x / 32', 'blockDim.x / 16'),
    ('threadIdx.x / 32', 'threadIdx.x / 16'),
    ('(threadIdx.x % 32) == 0', '(threadIdx.x % 16) == 0'),
])

# rehash_kernel.h
replace_in_file('src/gpu/rehash_kernel.h', [
    ('blockDim.x / 32', 'blockDim.x / 16'),
    ('threadIdx.x / 32', 'threadIdx.x / 16'),
    ('threadIdx.x % 32', 'threadIdx.x % 16'),
    ('__ballot_sync(0xFFFFFFFFu', '__ballot_sync(active_mask'),
    ('__shfl_sync(0xFFFFFFFFu', '__shfl_sync(active_mask'),
])

