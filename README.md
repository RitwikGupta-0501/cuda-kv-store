# gpu-kvstore

GPU-accelerated key-value store using CUDA for high-throughput workloads.

## Build
```bash
mkdir build && cd build
cmake ..
make
```

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- GCC/G++ with C++17 support
- NVIDIA GPU with 2GB+ VRAM

## Development Status

- [x] Phase 1.1: Environment setup
- [ ] Phase 1.2: Dynamic resource detection
- [ ] Phase 1.3: Hash functions
- [ ] Phase 1.4: Cuckoo hash table
- [ ] Phase 1.5: GPU lookup kernel

## License

MIT
