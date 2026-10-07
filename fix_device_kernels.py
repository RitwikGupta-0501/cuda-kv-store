import re

with open('include/warpkv/warpkv_device.cuh', 'r') as f:
    c = f.read()

# Replace lane_id definition
c = c.replace('const uint32_t lane_id = threadIdx.x % 32;', 
              'const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;\n    const uint32_t lane_id = threadIdx.x % 16;\n    const uint32_t warp_lane = threadIdx.x % 32;')

c = c.replace('const uint32_t lane_id   = threadIdx.x % 32;',
              'const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;\n    const uint32_t lane_id = threadIdx.x % 16;\n    const uint32_t warp_lane = threadIdx.x % 32;')

c = c.replace('__ballot_sync(0xFFFFFFFFu', '__ballot_sync(active_mask')
c = c.replace('__shfl_sync(0xFFFFFFFFu', '__shfl_sync(active_mask')

# Fix stash scans which were hardcoded to 32
c = c.replace('i += 32', 'i += 16')
# For stash scan in delete, it says: "if (lane_id == 0 && stash != nullptr)" -> This works fine for half-warps since lane_id is now hw_lane.

with open('include/warpkv/warpkv_device.cuh', 'w') as f:
    f.write(c)


with open('src/gpu/rehash_kernel.h', 'r') as f:
    c = f.read()

c = c.replace('uint32_t lane_id = threadIdx.x % 16;',
              'const uint32_t active_mask = (threadIdx.x % 32 < 16) ? 0x0000FFFFu : 0xFFFF0000u;\n    uint32_t lane_id = threadIdx.x % 16;')

with open('src/gpu/rehash_kernel.h', 'w') as f:
    f.write(c)

