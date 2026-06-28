#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../engine/warpkv_engine.h"
#include <stdexcept>

namespace py = pybind11;
using namespace warpkv;

PYBIND11_MODULE(warpkv, m) {
    m.doc() = "WarpKV Python bindings";

    py::class_<WarpKVEngine>(m, "WarpKVEngine")
        .def(py::init<>())
        .def("init", &WarpKVEngine::init, py::arg("num_buckets"))
        .def("insert", [](WarpKVEngine& engine, const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values) {
            if (keys.size() != values.size()) {
                throw std::invalid_argument("Keys and values lists must have the same length");
            }
            if (keys.size() == 0) return;
            if (keys.size() > warpkv::BATCH_SIZE) {
                throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
            }
            // Release the GIL while the CUDA operations run
            py::gil_scoped_release release;
            engine.submit_insert_batch(keys.data(), values.data(), static_cast<uint32_t>(keys.size()));
        }, py::arg("keys"), py::arg("values"), "Insert a batch of keys and values (max 4096)")
        .def("lookup", [](WarpKVEngine& engine, const std::vector<uint32_t>& keys) {
            if (keys.size() == 0) return std::vector<uint32_t>{};
            if (keys.size() > warpkv::BATCH_SIZE) {
                throw std::invalid_argument("Batch size exceeds BATCH_SIZE");
            }
            std::vector<uint32_t> values_out(keys.size());
            {
                // Release the GIL during pipeline submission and execution
                py::gil_scoped_release release;
                engine.submit_lookup_batch(keys.data(), values_out.data(), static_cast<uint32_t>(keys.size()));
            }
            
            return values_out;
        }, py::arg("keys"), "Lookup a batch of keys (max 4096)");
}
