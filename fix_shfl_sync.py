import re

files = ['include/warpkv/warpkv_device.cuh', 'src/gpu/rehash_kernel.h']

for path in files:
    with open(path, 'r') as f:
        c = f.read()

    # Replace hardcoded 0 with (threadIdx.x & ~15)
    c = re.sub(r'__shfl_sync\(active_mask,([^,]+), 0\)', r'__shfl_sync(active_mask,\1, (threadIdx.x & ~15))', c)

    with open(path, 'w') as f:
        f.write(c)

