# WarpKV: GPU-Accelerated Key-Value Store

WarpKV is a high-performance, hybrid database engine that leverages the massive parallelism of Nvidia GPUs to accelerate complex scan and filter operations.

Unlike traditional Key-Value stores (like Redis) that rely on CPU cycles for O(N) linear scans, WarpKV offloads these heavy compute tasks to the GPU, allowing for massive parallel throughput on large datasets.

## 🚀 The Core Concept

Standard databases are fast at retrieving single items by key (O(1)), but slow at scanning millions of items to find matches based on values (O(N)).

WarpKV splits the workload:

The Host (CPU): Acts as the Manager. It handles TCP networking, memory orchestration, and command parsing.

The Device (GPU): Acts as the Worker. It keeps data resident in VRAM and uses thousands of CUDA threads to blast through comparison logic in parallel.

## 🛠️ Tech Stack

Language: C++17 (Host), CUDA C++ (Device)

Build System: CMake

Hardware Target: Nvidia Maxwell Architecture (Optimized for MX130) and newer.

Networking: Raw Linux Sockets (TCP)

## 🏗️ Architecture

Host-Device Separation

To minimize the bottleneck of the PCIe bus, WarpKV utilizes a persistent memory model. Data is loaded into VRAM once. Queries send lightweight command packets to the GPU, and the GPU returns only the aggregate results, avoiding expensive data round-trips.

Structure of Arrays (SoA)

Instead of standard Array of Structures (AoS) which causes uncoalesced memory access on GPUs, WarpKV organizes data into parallel arrays (e.g., Ids[], Values[]). This ensures 100% memory bandwidth utilization during scan kernels.

## 🗺️ Roadmap & Status

[ ] Phase 1: The Foundation

[x] Environment Setup (Arch Linux + NVCC)

[ ] "Hello World" Kernel (Vector Addition)

[ ] Basic CMake integration

[ ] Phase 2: The Search Engine

[ ] Parallel Filter Kernel

[ ] Atomic operations for result counting

[ ] Race condition handling

[ ] Phase 3: The Persistent Store

[ ] GPUStore class implementation

[ ] VRAM persistence (load once, query many)

[ ] Structure of Arrays (SoA) implementation

[ ] Phase 4: Network Interface

[ ] TCP Server implementation

[ ] Custom text-based protocol (FILTER > 50)

## ⚡ Getting Started

### Prerequisites

Nvidia GPU (Maxwell or newer recommended)

Linux Environment (Arch Linux tested)

nvcc (CUDA Toolkit)

cmake & g++

### Build Instructions
```
# Clone the repository
git clone [https://github.com/yourusername/warpx-kv.git](https://github.com/yourusername/warpx-kv.git)
cd warpx-kv

# Create build directory
mkdir build && cd build

# Configure with CMake (Auto-detects NVCC)
cmake ..

# Compile
make

# Run the database
./warp_kv
```

This project is a deep dive into Systems Engineering, focusing on memory coalescing, warp divergence, and PCIe bandwidth optimization.
