import torch
import time

# 设置设备和 dtype
device = 'cuda'
dtype = torch.float16

# 配置参数
B, C1, H, W = 18, 3, 928, 1600
C2 = 4

# 创建输入数据
x1 = torch.randn(B, C1, H, W, device=device, dtype=dtype).contiguous()
x2 = torch.randn(B, C2, H, W, device=device, dtype=dtype).contiguous()

# 转换为 NHWC 格式
x1_nhwc = x1.to(memory_format=torch.channels_last)
x2_nhwc = x2.to(memory_format=torch.channels_last)

# 定义卷积层
conv1 = torch.nn.Conv2d(C1, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device, dtype=dtype).to(memory_format=torch.channels_last)
conv2 = torch.nn.Conv2d(C2, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device, dtype=dtype).to(memory_format=torch.channels_last)

print ("x1",x1.shape)
print ("x1.is_contiguous()",x1.is_contiguous())
print ("x1.stride()",x1.stride())
print ("x1_nhwc",x1_nhwc.shape)
print ("x1_nhwc.is_contiguous()",x1_nhwc.is_contiguous())
print ("x1_nhwc.stride()",x1_nhwc.stride())
print("is_allclose",torch.allclose(x1, x1_nhwc))
#print(conv1.weight)            # 查看整个权重 tensor
print(conv1.weight.shape)      # 查看形状
print(conv1.weight.dtype)      # 数据类型
print(conv1.weight.device)     # 设备位置


 # Warmup
for _ in range(100):
    out = conv1(x1_nhwc)
torch.cuda.synchronize()

for _ in range(100):
    out2 = conv2(x2_nhwc)
torch.cuda.synchronize()

import time
start_time = time.time()
out = conv1(x1_nhwc)
end_time  = time.time()
print ("conv1",end_time-start_time)


import time
start_time = time.time()
out2 = conv2(x2_nhwc)
end_time  = time.time()
print ("conv2",end_time-start_time)
