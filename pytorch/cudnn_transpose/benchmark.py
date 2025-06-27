import torch
import time
import sys
sys.path.append('.')
import custom_transpose
import cudnn_transpose

def benchmark_custom_vs_torch(
    N=1, C=3, H=224, W=224,
    num_warmup=10,
    num_runs=10,
    dtype=torch.float,
    device='cuda'
):
    # 构造输入张量
    x = torch.randn(N, C, H, W, dtype=dtype, device=device)
    # x = torch.randint(-128, 127, (1, 3, 224, 224), dtype=torch.int8).cuda()
    # x = torch.quantize_per_tensor(
    #     torch.randn(1, 3, 224, 224), scale=0.1, zero_point=10, dtype=torch.qint8
    # ).cuda()
    # Warm up
    for _ in range(num_warmup):
        # y_custom = custom_transpose.transpose_nchw_to_nhwc(x)
        y_custom = cudnn_transpose.cudnnNchwToNhwc(x)
        # y_torch = x.permute(0, 2, 3, 1).contiguous()

    torch.cuda.synchronize()

    # 计时器
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 测量自定义 kernel 时间
    start_event.record()
    for _ in range(num_runs):
        y_custom = cudnn_transpose.cudnnNchwToNhwc(x)
        # y_custom = custom_transpose.transpose_nchw_to_nhwc(x)
    end_event.record()
    torch.cuda.synchronize()
    custom_time_ms = start_event.elapsed_time(end_event) / num_runs

    # # 测量 PyTorch permute 时间
    # start_event.record()
    # for _ in range(num_runs):
    #     y_torch = x.permute(0, 2, 3, 1).contiguous()
    # end_event.record()
    # torch.cuda.synchronize()
    # torch_time_ms = start_event.elapsed_time(end_event) / num_runs

    # # 打印结果
    # print(f"Input size: {x.shape}")
    # print(f"Custom kernel time: {custom_time_ms:.4f} ms")
    # print(f"PyTorch permute time: {torch_time_ms:.4f} ms")
    # print(f"Speedup: {torch_time_ms / custom_time_ms:.2f}x")
    # print("------------------------------")
    return custom_time_ms
    # return custom_time_ms, torch_time_ms

if __name__ == '__main__':
    print("Benchmarking FP32:")
    benchmark_custom_vs_torch(N=4, C=8, H=448, W=448, dtype=torch.float)

    # print("Benchmarking INT8:")
    # x_int8 = torch.randint(-128, 127, (1, 3, 224, 224), dtype=torch.int8).cuda()
    # benchmark_custom_vs_torch(N=1, C=3, H=224, W=224, dtype=torch.int8)

    # print("Benchmarking QINT8:")
    # x_qint8 = torch.quantize_per_tensor(
    #     torch.randn(1, 3, 224, 224), scale=0.1, zero_point=10, dtype=torch.qint8
    # ).cuda()
    # benchmark_custom_vs_torch(N=1, C=3, H=224, W=224, dtype=torch.qint8)

    # print("Benchmarking Half (FP16):")
    # benchmark_custom_vs_torch(N=1, C=3, H=224, W=224, dtype=torch.half)