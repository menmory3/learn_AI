// transpose_kernel.cu

#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// template <typename T>
// __global__ void transpose_nchw_to_nhwc_kernel(
//     const int n,
//     const T* input,
//     T* output,
//     int N, int C, int H, int W) {

//     for (int index = blockIdx.x * blockDim.x + threadIdx.x;
//          index < n;
//          index += blockDim.x * gridDim.x) {

//         const int w = index % W;
//         const int h = (index / W) % H;
//         const int c = (index / W / H) % C;
//         const int n_idx = (index / W / H / C);

//         const int in_idx = ((n_idx * C + c) * H + h) * W + w;
//         const int out_idx = ((n_idx * H + h) * W + w) * C + c;

//         output[out_idx] = input[in_idx];
//     }
// }

template <typename T>
__global__ void transpose_nchw_to_nhwc_kernel(
    const int n,
    const T* input,
    T* output,
    int N, int C, int H, int W) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / W / H) % C;
    const int n_idx = (index / W / H / C);

    const int in_idx = ((n_idx * C + c) * H + h) * W + w;
    const int out_idx = ((n_idx * H + h) * W + w) * C + c;

    // for (int index = blockIdx.x * blockDim.x + threadIdx.x;
    //      index < n;
    //      index += blockDim.x * gridDim.x) {

    //     const int w = index % W;
    //     const int h = (index / W) % H;
    //     const int c = (index / W / H) % C;
    //     const int n_idx = (index / W / H / C);

    //     const int in_idx = ((n_idx * C + c) * H + h) * W + w;
    //     const int out_idx = ((n_idx * H + h) * W + w) * C + c;

    //     output[out_idx] = input[in_idx];
    // }

    if (index < n) {
         output[out_idx] = input[in_idx];
    }
}



template <typename T>
void transpose_nchw_to_nhwc_launcher(
    const T* input,
    T* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {
}

// 显式模板特例化
template<>
void transpose_nchw_to_nhwc_launcher<float>(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    const int total_elements = N * H * W * C;

    constexpr int block_size = 1024;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    transpose_nchw_to_nhwc_kernel<float><<<grid_size, block_size, 0, stream>>>(
        total_elements, input, output, N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

template<>
void transpose_nchw_to_nhwc_launcher<at::Half>(
    const at::Half* input,
    at::Half* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    const int total_elements = N * H * W * C;

    constexpr int block_size = 1024;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    transpose_nchw_to_nhwc_kernel<at::Half><<<grid_size, block_size, 0, stream>>>(
        total_elements, input, output, N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

template<>
void transpose_nchw_to_nhwc_launcher<int8_t>(
    const int8_t* input,
    int8_t* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    const int total_elements = N * H * W * C;

    constexpr int block_size = 1024;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    transpose_nchw_to_nhwc_kernel<int8_t><<<grid_size, block_size, 0, stream>>>(
        total_elements, input, output, N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}
