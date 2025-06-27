import torch
import torch.profiler as profiler
import numpy as np

# 设置分析环境
device = torch.device('cuda:0')
torch.cuda.set_device(device)

# 创建示例输入张量 (NCHW 格式)
batch_size, channels, height, width = 32, 3, 224, 224
x = torch.randn(batch_size, channels, height, width, device=device)

# 使用 profiler 分析内存格式转换
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/to_memory_format'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(5):  # 多次迭代获取稳定结果
        with profiler.record_function("format_conversion"):
            # 执行内存格式转换
            x_nhwc = x.to(memory_format=torch.channels_last)
        
        # 确保完成所有GPU操作
        torch.cuda.synchronize()
        prof.step()

# 打印分析结果
print(prof.key_averages().table(
    sort_by="cuda_time_total", 
    row_limit=20
))

# 可选：保存详细结果用于TensorBoard
#prof.export_chrome_trace("trace.json")
