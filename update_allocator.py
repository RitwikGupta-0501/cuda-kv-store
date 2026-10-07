import re

with open('src/engine/warpkv_engine.cu', 'r') as f:
    c = f.read()

# Setup allocator_ in init
c = c.replace('void WarpKVEngine::init(uint32_t num_buckets, WarpKVAllocator* allocator) {',
              'void WarpKVEngine::init(uint32_t num_buckets, WarpKVAllocator* allocator) {\n    if (allocator) {\n        allocator_ = allocator;\n    } else {\n        default_allocator_ = std::make_unique<DefaultCudaAllocator>();\n        allocator_ = default_allocator_.get();\n    }\n')

# Replace CUDA_CHECK(cudaMalloc(&ptr, size)) with ptr = (type)allocator_->allocate(size)
def replace_malloc(match):
    ptr = match.group(1)
    size = match.group(2)
    # The pointer is passed as address, so we remove the '&'
    if ptr.startswith('&'):
        ptr = ptr[1:]
    
    # We need to cast the return of allocate
    return f'{ptr} = ({ptr.split("[")[0]}__TYPE__)allocator_->allocate({size});'.replace('__TYPE__', '') # type casting is messy to infer, let's just use reinterpret_cast
    
c = re.sub(r'CUDA_CHECK\(cudaMalloc\(([^,]+),\s*([^)]+)\)\);',
           r'\1 = reinterpret_cast<decltype(\1)>(allocator_->allocate(\2));'.replace('&', ''), c)
           
# Fix the & issue in the replacement output
c = c.replace('epoch_table.arenas[0]->buckets = reinterpret_cast<decltype(epoch_table.arenas[0]->buckets)>', 'epoch_table.arenas[0]->buckets = reinterpret_cast<Bucket*>')
c = c.replace('d_stash_queue = reinterpret_cast<decltype(d_stash_queue)>', 'd_stash_queue = reinterpret_cast<StashQueue*>')
c = c.replace('d_keys_in[i] = reinterpret_cast<decltype(d_keys_in[i])>', 'd_keys_in[i] = reinterpret_cast<KeyT*>')
c = c.replace('d_values_in[i] = reinterpret_cast<decltype(d_values_in[i])>', 'd_values_in[i] = reinterpret_cast<ValueT*>')
c = c.replace('d_values_out[i] = reinterpret_cast<decltype(d_values_out[i])>', 'd_values_out[i] = reinterpret_cast<ValueT*>')
c = c.replace('d_insert_statuses[i] = reinterpret_cast<decltype(d_insert_statuses[i])>', 'd_insert_statuses[i] = reinterpret_cast<InsertStatus*>')
c = c.replace('d_lookup_found[i] = reinterpret_cast<decltype(d_lookup_found[i])>', 'd_lookup_found[i] = reinterpret_cast<uint32_t*>')
c = c.replace('new_tbl->buckets = reinterpret_cast<decltype(new_tbl->buckets)>', 'new_tbl->buckets = reinterpret_cast<Bucket*>')

# For free, replace cudaFree(ptr) with allocator_->deallocate(ptr)
c = re.sub(r'CUDA_CHECK\(cudaFree\(([^)]+)\)\);', r'allocator_->deallocate(\1);', c)

# Add #include "../include/warpkv/warpkv_allocator.h" at top if not present
if 'warpkv_allocator.h' not in c:
    c = '#include "../include/warpkv/warpkv_allocator.h"\n' + c

with open('src/engine/warpkv_engine.cu', 'w') as f:
    f.write(c)
