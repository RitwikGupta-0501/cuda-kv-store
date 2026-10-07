import re

with open('src/engine/warpkv_engine.cu', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # values_in and values_out should use sizeof(ValueT)
    if 'h_values' in line or 'd_values' in line or 'values_out' in line:
        line = line.replace('sizeof(KeyT)', 'sizeof(ValueT)')
    # found and needs_rehash_flag should use sizeof(uint32_t)
    if 'h_lookup_found' in line or 'd_lookup_found' in line or 'h_needs_rehash_flag' in line:
        line = line.replace('sizeof(KeyT)', 'sizeof(uint32_t)')
        line = line.replace('sizeof(ValueT)', 'sizeof(uint32_t)')
    
    # signature fixes:
    line = line.replace('const uint32_t* keys', 'const KeyT* keys')
    line = line.replace('const uint32_t* values', 'const ValueT* values')
    line = line.replace('uint32_t*       values_out', 'ValueT*       values_out')

    lines[i] = line

with open('src/engine/warpkv_engine.cu', 'w') as f:
    f.writelines(lines)

