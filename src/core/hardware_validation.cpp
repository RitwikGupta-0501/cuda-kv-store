#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// Phase 0: Hardware validation utility
// Checks MX130 specifications and measures PCIe bandwidth

struct HardwareInfo {
    int cuda_device;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int cores_per_multiprocessor;
    size_t total_global_memory;
    size_t l2_cache_size;
    int pcie_bus_id;
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  Error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

HardwareInfo query_gpu() {
    HardwareInfo info = {};

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        exit(1);
    }

    info.cuda_device = 0;
    CUDA_CHECK(cudaSetDevice(info.cuda_device));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, info.cuda_device));

    info.compute_capability_major = props.major;
    info.compute_capability_minor = props.minor;
    info.multiprocessor_count = props.multiProcessorCount;
    info.cores_per_multiprocessor = props.maxThreadsPerMultiProcessor / props.warpSize;
    info.total_global_memory = props.totalGlobalMem;
    info.l2_cache_size = props.l2CacheSize;
    info.pcie_bus_id = props.pciBusID;

    return info;
}

// Measure PCIe bandwidth with simple memcpy
struct BandwidthResult {
    double host_to_device_gbps;
    double device_to_host_gbps;
};

BandwidthResult measure_pcie_bandwidth(size_t transfer_size, int num_iterations) {
    BandwidthResult result = {};

    // Allocate host and device memory
    std::vector<uint8_t> h_buffer(transfer_size);
    uint8_t* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, transfer_size));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer.data(), transfer_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_buffer, transfer_size, cudaMemcpyDeviceToHost));
    }

    // Measure Host->Device
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer.data(), transfer_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    double total_bytes = (double)transfer_size * num_iterations;
    result.host_to_device_gbps = (total_bytes / (1024 * 1024 * 1024)) / (elapsed_us / 1e6);

    // Measure Device->Host
    CUDA_CHECK(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_buffer, transfer_size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    end = std::chrono::high_resolution_clock::now();
    elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.device_to_host_gbps = (total_bytes / (1024 * 1024 * 1024)) / (elapsed_us / 1e6);

    CUDA_CHECK(cudaFree(d_buffer));

    return result;
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "WarpKV v2.0.2 — Phase 0: Hardware Validation" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;

    // Query GPU
    std::cout << "Querying GPU Hardware..." << std::endl;
    HardwareInfo info = query_gpu();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  GPU Device:              " << info.cuda_device << std::endl;
    std::cout << "  Compute Capability:      " << info.compute_capability_major
              << "." << info.compute_capability_minor << std::endl;
    std::cout << "  Multiprocessors:         " << info.multiprocessor_count << std::endl;
    std::cout << "  Total Global Memory:     " << (info.total_global_memory / (1024 * 1024 * 1024))
              << " GB" << std::endl;
    std::cout << "  L2 Cache Size:           " << (info.l2_cache_size / 1024) << " KB" << std::endl;
    std::cout << "  PCIe Bus ID:             " << info.pcie_bus_id << std::endl << std::endl;

    // Validate CC 5.0
    if (info.compute_capability_major == 5 && info.compute_capability_minor == 0) {
        std::cout << "✓ Compute Capability 5.0 (Maxwell) — Correct for MX130" << std::endl;
    } else {
        std::cout << "✗ WARNING: Compute Capability " << info.compute_capability_major
                  << "." << info.compute_capability_minor
                  << " (expected CC 5.0 for MX130)" << std::endl;
    }

    // Validate memory
    if (info.total_global_memory >= 2ULL * 1024 * 1024 * 1024) {
        std::cout << "✓ VRAM >= 2 GB — Sufficient for WarpKV" << std::endl;
    } else {
        std::cout << "✗ WARNING: VRAM < 2 GB — May be insufficient" << std::endl;
    }

    // Validate L2 cache
    if (info.l2_cache_size >= 512 * 1024) {
        std::cout << "✓ L2 Cache >= 512 KB — Sufficient for pipelined lookups" << std::endl;
    } else {
        std::cout << "✗ WARNING: L2 Cache < 512 KB" << std::endl;
    }

    std::cout << std::endl;

    // Measure PCIe bandwidth
    std::cout << "Measuring PCIe Bandwidth..." << std::endl;
    std::cout << "  Transfer size: 1 MB, iterations: 100" << std::endl;

    BandwidthResult bandwidth = measure_pcie_bandwidth(1024 * 1024, 100);

    std::cout << "  Host → Device:           " << bandwidth.host_to_device_gbps << " GB/s" << std::endl;
    std::cout << "  Device → Host:           " << bandwidth.device_to_host_gbps << " GB/s" << std::endl;

    double effective_bw = std::min(bandwidth.host_to_device_gbps, bandwidth.device_to_host_gbps);
    std::cout << "  Effective (bidirectional): " << effective_bw << " GB/s" << std::endl << std::endl;

    // Validate PCIe bandwidth (spec: ~6 GB/s for Gen 3 x4)
    if (effective_bw >= 5.0) {
        std::cout << "✓ PCIe Bandwidth >= 5 GB/s — Acceptable for PCIe Gen 3 x4" << std::endl;
    } else {
        std::cout << "⚠ WARNING: PCIe Bandwidth < 5 GB/s — Performance may be limited" << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Phase 0 Validation Complete" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}
