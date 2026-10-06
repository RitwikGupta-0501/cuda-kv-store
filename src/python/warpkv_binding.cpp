#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../engine/warpkv_engine.h"
#include <stdexcept>

namespace py = pybind11;
using namespace warpkv;

PYBIND11_MODULE(warpkv, m) {
    m.doc() = "WarpKV Python bindings — GPU-accelerated key-value store";

    py::class_<WarpKVEngine>(m, "Engine")
        // Constructor: init(num_buckets)
        .def(py::init([](uint32_t num_buckets) {
            auto engine = std::make_unique<WarpKVEngine>();
            engine->init(num_buckets);
            return engine;
        }), py::arg("num_buckets"),
        "Create and initialize a WarpKV engine with `num_buckets` buckets.\n"
        "num_buckets must be a power of 2. Key 0 and 0xFFFFFFFF are reserved.")

        // ----------------------------------------------------------------
        // insert_batch
        // ----------------------------------------------------------------
        .def("insert_batch",
            [](WarpKVEngine& engine,
               py::array_t<uint32_t> keys,
               py::array_t<uint32_t> values)
        {
            const py::buffer_info keys_info   = keys.request();
            const py::buffer_info values_info = values.request();

            if (keys_info.size != values_info.size) {
                throw std::invalid_argument("keys and values must have the same length");
            }
            if (keys_info.size == 0) return;
            if (static_cast<size_t>(keys_info.size) > static_cast<size_t>(warpkv::BATCH_SIZE)) {
                throw std::invalid_argument(
                    "Batch size exceeds BATCH_SIZE (" +
                    std::to_string(warpkv::BATCH_SIZE) + ")");
            }

            auto* keys_ptr   = static_cast<const uint32_t*>(keys_info.ptr);
            auto* values_ptr = static_cast<const uint32_t*>(values_info.ptr);
            const uint32_t count = static_cast<uint32_t>(keys_info.size);

            // Release the GIL while the blocking CUDA pipeline runs.
            py::gil_scoped_release release;
            engine.submit_insert_batch_sync(keys_ptr, values_ptr, count);
        },
        py::arg("keys"), py::arg("values"),
        "Insert a batch of (key, value) pairs. Max batch size: BATCH_SIZE (4096).\n"
        "This call blocks until the GPU pipeline completes.")

        // ----------------------------------------------------------------
        // lookup_batch
        // ----------------------------------------------------------------
        .def("lookup_batch",
            [](WarpKVEngine& engine, py::array_t<uint32_t> keys) -> py::array_t<uint32_t>
        {
            const py::buffer_info keys_info = keys.request();
            if (keys_info.size == 0) return py::array_t<uint32_t>(0);
            if (static_cast<size_t>(keys_info.size) > static_cast<size_t>(warpkv::BATCH_SIZE)) {
                throw std::invalid_argument(
                    "Batch size exceeds BATCH_SIZE (" +
                    std::to_string(warpkv::BATCH_SIZE) + ")");
            }

            auto* keys_ptr    = static_cast<const uint32_t*>(keys_info.ptr);
            const uint32_t count = static_cast<uint32_t>(keys_info.size);

            // Allocate output array before releasing the GIL (pybind11 operations need GIL).
            py::array_t<uint32_t> values_out(count);
            py::buffer_info out_info = values_out.request();
            auto* out_ptr = static_cast<uint32_t*>(out_info.ptr);

            {
                py::gil_scoped_release release;
                engine.submit_lookup_batch_sync(keys_ptr, out_ptr, count);
            }

            return values_out;
        },
        py::arg("keys"),
        "Look up a batch of keys. Returns a NumPy array of values.\n"
        "Missing keys are returned as 0xFFFFFFFF (NOT_FOUND).\n"
        "Max batch size: BATCH_SIZE (4096). Blocks until GPU completes.")

        // ----------------------------------------------------------------
        // delete_batch
        // ----------------------------------------------------------------
        .def("delete_batch",
            [](WarpKVEngine& engine, py::array_t<uint32_t> keys)
        {
            const py::buffer_info keys_info = keys.request();
            if (keys_info.size == 0) return;
            if (static_cast<size_t>(keys_info.size) > static_cast<size_t>(warpkv::BATCH_SIZE)) {
                throw std::invalid_argument(
                    "Batch size exceeds BATCH_SIZE (" +
                    std::to_string(warpkv::BATCH_SIZE) + ")");
            }

            auto* keys_ptr   = static_cast<const uint32_t*>(keys_info.ptr);
            const uint32_t count = static_cast<uint32_t>(keys_info.size);

            py::gil_scoped_release release;
            engine.submit_delete_batch_sync(keys_ptr, count);
        },
        py::arg("keys"),
        "Delete a batch of keys. Max batch size: BATCH_SIZE (4096).\n"
        "Blocks until GPU pipeline completes.");
}
