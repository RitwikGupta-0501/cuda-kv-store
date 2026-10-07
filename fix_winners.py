import re

files = ['include/warpkv/warpkv_device.cuh', 'src/gpu/rehash_kernel.h']

for path in files:
    with open(path, 'r') as f:
        c = f.read()

    c = c.replace('if (lane_id == (uint32_t)b1_winner)', 'if (warp_lane == (uint32_t)b1_winner)')
    c = c.replace('if (lane_id == (uint32_t)b2_winner)', 'if (warp_lane == (uint32_t)b2_winner)')
    c = c.replace('if (lane_id == b1_winner)', 'if (warp_lane == (uint32_t)b1_winner)')
    c = c.replace('if (lane_id == b2_winner)', 'if (warp_lane == (uint32_t)b2_winner)')
    c = c.replace('if (lane_id == (uint32_t)winner)', 'if (warp_lane == (uint32_t)winner)')
    
    with open(path, 'w') as f:
        f.write(c)

