import re

with open('src/python/warpkv_binding.cpp', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Fix the generic replacement issues
    line = line.replace('py::array_t<uint32_t>', 'py::array_t<KeyT>') # Mostly correct, but we'll fix values later
    
    if 'values' in line and 'keys' not in line and 'array_t<KeyT>' in line:
        line = line.replace('array_t<KeyT>', 'array_t<ValueT>')
    
    if 'py::array_t<KeyT> values' in line:
        line = line.replace('py::array_t<KeyT> values', 'py::array_t<ValueT> values')
        
    if 'py::array_t<KeyT> values_out' in line:
        line = line.replace('py::array_t<KeyT> values_out', 'py::array_t<ValueT> values_out')
        
    if 'static_cast<const uint32_t*>(values_info.ptr)' in line:
        line = line.replace('static_cast<const uint32_t*>', 'static_cast<const ValueT*>')
        
    if 'static_cast<uint32_t*>(out_info.ptr)' in line:
        line = line.replace('static_cast<uint32_t*>', 'static_cast<ValueT*>')
        
    lines[i] = line

with open('src/python/warpkv_binding.cpp', 'w') as f:
    f.writelines(lines)

