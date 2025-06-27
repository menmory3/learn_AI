
// custom_transpose.cpp

// custom_transpose.cpp

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// 声明 launcher
template <typename T>
void transpose_nchw_to_nhwc_launcher(
    const T* input,
    T* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

at::Tensor transpose_nchw_to_nhwc_cuda(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    auto options = input.options().dtype(input.scalar_type());
    

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // 👇 正确构造 NHWC 的 shape
    std::vector<int64_t> sizes{N, H, W, C};

    // 创建输出张量
    auto output = at::empty(sizes, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 手动分发：根据 scalar_type 实例化对应 kernel
    switch (input.scalar_type()) {
        case at::ScalarType::Float:
            transpose_nchw_to_nhwc_launcher<float>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                N, C, H, W,
                stream);
            break;

        case at::ScalarType::Half:
            transpose_nchw_to_nhwc_launcher<at::Half>(
                input.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                N, C, H, W,
                stream);
            break;

        case at::ScalarType::Char:  // int8_t
            transpose_nchw_to_nhwc_launcher<int8_t>(
                input.data_ptr<int8_t>(),
                output.data_ptr<int8_t>(),
                N, C, H, W,
                stream);
            break;

        case at::ScalarType::QInt8:  // qint8
            transpose_nchw_to_nhwc_launcher<int8_t>(
                input.data_ptr<int8_t>(),
                output.data_ptr<int8_t>(),
                N, C, H, W,
                stream);
            break;

        default:
            AT_ERROR("transpose_nchw_to_nhwc_cuda not implemented for ", input.scalar_type());
    }

    return output;
}

// custom_transpose.cpp 最后添加：
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_nchw_to_nhwc", &transpose_nchw_to_nhwc_cuda, "NCHW to NHWC transpose (CUDA)");
}