#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include "../engine/warpkv_engine.h"

using namespace warpkv;

// Wrapper for PyTorch CUDACachingAllocator
class TorchAllocator : public WarpKVAllocator {
public:
    void* allocate(size_t bytes) override {
        return c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
    }
    void deallocate(void* ptr) override {
        if (ptr) {
            c10::cuda::CUDACachingAllocator::raw_delete(ptr);
        }
    }
};

class TorchEngine {
private:
    std::unique_ptr<WarpKVEngine> engine_;
    std::unique_ptr<TorchAllocator> allocator_;

public:
    TorchEngine(uint32_t num_buckets) {
        engine_ = std::make_unique<WarpKVEngine>();
        allocator_ = std::make_unique<TorchAllocator>();
        engine_->init(num_buckets, allocator_.get());
    }

    void insert_batch(torch::Tensor keys, torch::Tensor values) {
        TORCH_CHECK(keys.is_cuda(), "Keys must be a CUDA tensor");
        TORCH_CHECK(values.is_cuda(), "Values must be a CUDA tensor");
        TORCH_CHECK(keys.dim() == 1, "Keys must be 1D");
        TORCH_CHECK(values.dim() == 1, "Values must be 1D");
        TORCH_CHECK(keys.size(0) == values.size(0), "Keys and values must have same size");

        const KeyT* d_keys = reinterpret_cast<const KeyT*>(keys.data_ptr());
        const ValueT* d_values = reinterpret_cast<const ValueT*>(values.data_ptr());
        uint32_t count = keys.size(0);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // Release GIL while blocking on the future
        pybind11::gil_scoped_release release;
        engine_->submit_insert_batch_device(d_keys, d_values, count, stream).get();
    }

    torch::Tensor lookup_batch(torch::Tensor keys) {
        TORCH_CHECK(keys.is_cuda(), "Keys must be a CUDA tensor");
        TORCH_CHECK(keys.dim() == 1, "Keys must be 1D");

        uint32_t count = keys.size(0);
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(keys.device());
        torch::Tensor values_out = torch::empty({count}, options);

        const KeyT* d_keys = reinterpret_cast<const KeyT*>(keys.data_ptr());
        ValueT* d_values = reinterpret_cast<ValueT*>(values_out.data_ptr());

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        {
            pybind11::gil_scoped_release release;
            engine_->submit_lookup_batch_device(d_keys, d_values, count, stream).get();
        }
        
        return values_out;
    }

    void delete_batch(torch::Tensor keys) {
        TORCH_CHECK(keys.is_cuda(), "Keys must be a CUDA tensor");
        TORCH_CHECK(keys.dim() == 1, "Keys must be 1D");

        const KeyT* d_keys = reinterpret_cast<const KeyT*>(keys.data_ptr());
        uint32_t count = keys.size(0);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        pybind11::gil_scoped_release release;
        engine_->submit_delete_batch_device(d_keys, count, stream).get();
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<TorchEngine>(m, "Engine")
        .def(pybind11::init<uint32_t>(), pybind11::arg("num_buckets"))
        .def("insert_batch", &TorchEngine::insert_batch, pybind11::arg("keys"), pybind11::arg("values"))
        .def("lookup_batch", &TorchEngine::lookup_batch, pybind11::arg("keys"))
        .def("delete_batch", &TorchEngine::delete_batch, pybind11::arg("keys"));
}
