# WarpKV Architecture: Python Bindings & Memory Interop

## 1. Overview
While WarpKV is built entirely in CUDA and C++ for maximum throughput, it is designed to be used seamlessly from Python data science and machine learning environments (e.g., PyTorch, NumPy). We utilize **pybind11** to expose the C++ classes and methods to Python. 

The primary challenge in Python-C++ interop for GPU databases is memory transfer overhead. If millions of keys are serialized, copied into a list, cast to C++ types, and then copied again, the CPU bottlenecks before the GPU even receives the data.

## 2. Zero-Copy NumPy Integration
WarpKV explicitly relies on the `pybind11::array_t` type to handle data ingestion. 
When a user passes a NumPy array to the Python `submit_insert_batch` method:
1. `pybind11` grabs the underlying raw memory pointer from the NumPy array.
2. It verifies that the array is contiguous and contains 32-bit unsigned integers (`uint32`).
3. No data is copied between Python and C++. The C++ engine directly reads the NumPy memory buffer.

## 3. The Pinned Memory Hop
While we avoid a Python-to-C++ copy, we still face a hardware reality: the GPU needs to read the data. 
To maximize PCIe transfer speeds (`cudaMemcpyAsync`), the source memory on the CPU must be **Pinned** (Page-Locked). Memory allocated by NumPy in Python is generally pageable, not pinned.

If we called `cudaMemcpyAsync` directly on the NumPy pointer, the NVIDIA driver would be forced to allocate a hidden pinned buffer, copy the NumPy data into it, and then perform the PCIe transfer. This hidden serialization destroys pipelining performance.

**The Solution:**
WarpKV utilizes its pre-allocated pinned memory ring buffers (`h_keys_in`, `h_values_in`, `h_values_out`).
1. The C++ binding uses `std::memcpy` to rapidly copy the data from the NumPy array pointer into the engine's pinned `h_keys_in[slot]` buffer.
2. Because `std::memcpy` is highly optimized (utilizing AVX registers on modern CPUs), this hop is nearly instantaneous.
3. The engine then issues the non-blocking `cudaMemcpyAsync` from the pinned buffer to the GPU.

## 4. Output Buffer Pre-Allocation
For lookups, the engine requires a destination array to write the found values. 
To maintain high performance and avoid Python garbage collection overhead:
1. The user allocates an empty NumPy array of the correct size in Python.
2. They pass this array to `submit_lookup_batch`.
3. The C++ binding extracts the underlying pointer.
4. After the GPU finishes the lookup and copies the results into the pinned `h_values_out[slot]`, the engine uses `std::memcpy` to move the data into the user's NumPy array.
5. The function returns, and the Python user instantly sees the populated array.

## 5. Releasing the GIL
Python's Global Interpreter Lock (GIL) prevents multi-threading. Because WarpKV's `submit` methods can block (e.g., waiting for `cudaStreamSynchronize` in the ring buffer), we explicitly release the GIL during the engine calls using `pybind11::gil_scoped_release`.
This allows other Python threads (like background data loaders) to continue running concurrently while the C++ engine waits on the GPU.
