#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../engine/warpkv_engine.h"
#include <stdexcept>

#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace warpkv;

PYBIND11_MODULE(warpkv, m) {
    m.doc() = "WarpKV Python bindings";

    py::class_<WarpKVEngine>(m, "Engine")
        .def(py::init([](uint32_t num_buckets) {
            auto engine = std::make_unique<WarpKVEngine>();
            engine->init(num_buckets);
            return engine;
        }), py::arg("num_buckets"))
        .def("insert_batch", [](WarpKVEngine& engine, py::array_t<uint32_t> keys, py::array_t<uint32_t> values) {
            py::buffer_info keys_info = keys.request();
            py::buffer_info values_info = values.request();
            
            if (keys_info.size != values_info.size) {
                throw std::invalid_argument("Keys and values arrays must have the same length");
            }
            if (keys_info.size == 0) return;
            if (keys_info.size > warpkv::BATCH_SIZE) {
                throw std::invalid_argument("Batch size exceeds BATCH_SIZE (4096)");
            }
            
            uint32_t* keys_ptr = static_cast<uint32_t*>(keys_info.ptr);
            uint32_t* values_ptr = static_cast<uint32_t*>(values_info.ptr);
            
            // Release the GIL while the CUDA operations run
            py::gil_scoped_release release;
            engine.submit_insert_batch(keys_ptr, values_ptr, static_cast<uint32_t>(keys_info.size));
        }, py::arg("keys"), py::arg("values"), "Insert a batch of keys and values (max 4096)")
        .def("lookup_batch", [](WarpKVEngine& engine, py::array_t<uint32_t> keys) {
            py::buffer_info keys_info = keys.request();
            if (keys_info.size == 0) return py::array_t<uint32_t>(0);
            if (keys_info.size > warpkv::BATCH_SIZE) {
                throw std::invalid_argument("Batch size exceeds BATCH_SIZE (4096)");
            }
            
            uint32_t* keys_ptr = static_cast<uint32_t*>(keys_info.ptr);
            py::array_t<uint32_t> values_out(keys_info.size);
            py::buffer_info out_info = values_out.request();
            uint32_t* out_ptr = static_cast<uint32_t*>(out_info.ptr);
            
            {
                // Release the GIL during pipeline submission and execution
                py::gil_scoped_release release;
                engine.submit_lookup_batch(keys_ptr, out_ptr, static_cast<uint32_t>(keys_info.size));
            }
            
            return values_out;
        }, py::arg("keys"), "Lookup a batch of keys (max 4096)")
        .def("delete_batch", [](WarpKVEngine& engine, py::array_t<uint32_t> keys) {
            py::buffer_info keys_info = keys.request();
            if (keys_info.size == 0) return;
            if (keys_info.size > warpkv::BATCH_SIZE) {
                throw std::invalid_argument("Batch size exceeds BATCH_SIZE (4096)");
            }
            
            uint32_t* keys_ptr = static_cast<uint32_t*>(keys_info.ptr);
            
            // Release the GIL while the CUDA operations run
            py::gil_scoped_release release;
            engine.submit_delete_batch(keys_ptr, static_cast<uint32_t>(keys_info.size));
        }, py::arg("keys"), "Delete a batch of keys (max 4096)");
}
