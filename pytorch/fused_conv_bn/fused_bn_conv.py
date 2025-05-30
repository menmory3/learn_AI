# BN在前和conv 融合，BN 转conv
# https://zhuanlan.zhihu.com/p/600453512

import copy
 
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch import optim
 
def fuse_bn_conv_eval(bn, conv, transpose=False):
    assert(not (bn.training or conv.training)), "Fusion only for eval!"
 
    if any(k != 1 for k in conv.kernel_size) and conv.padding_mode == 'zeros':
    # if any(k != 1 for k in conv.kernel_size) :
        print ("##################")
        print ("transpose=False")
        # identity_conv = conv.__class__(
        #     bn.num_features, bn.num_features, 1
        #     # bn.num_features, bn.num_features, 1, groups=bn.num_features
        # )
        identity_conv = nn.Conv2d(
            # bn.num_features, bn.num_features, 1
            bn.num_features, bn.num_features, 1, groups=bn.num_features
        )
        nn.init.ones_(identity_conv.weight)
        # nn.init.zeros_(identity_conv.weight)
        nn.init.zeros_(identity_conv.bias)
        identity_conv.eval()
        fused_conv = fuse_conv_bn_eval(identity_conv, bn)
        return nn.Sequential(fused_conv, conv)
 
    fused_conv = copy.deepcopy(conv)
 
    fused_conv.weight, fused_conv.bias = fuse_bn_conv_weights(
        bn_rm=bn.running_mean,
        bn_rv=bn.running_var,
        bn_eps=bn.eps,
        bn_w=bn.weight,
        bn_b=bn.bias,
        conv_w=fused_conv.weight,
        conv_b=fused_conv.bias,
        transpose=transpose,
    )
 
    return fused_conv
 
 
def fuse_bn_conv_weights(
    bn_rm, bn_rv, bn_eps, bn_w, bn_b, conv_w, conv_b, transpose=False
):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    if conv_b is None:
        if transpose:
            conv_b = conv_w.new_zeros(conv_w.shape[1])
        else:
            conv_b = conv_w.new_zeros(conv_w.shape[0])
 
    if transpose:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
 
    fused_w = conv_w * bn_scale.reshape(shape)
    fused_b = torch.addmv(conv_b, conv_w.sum(tuple(range(2, len(conv_w.shape)))), bn_b - bn_rm * bn_scale)
 
    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)
 
 
def fuse_bn_linear_eval(linear, bn):
    assert(not (bn.training or linear.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)
 
    fused_linear.weight, fused_linear.bias = fuse_bn_linear_weights(
        bn_rm=bn.running_mean,
        bn_rv=bn.running_var,
        bn_eps=bn.eps,
        bn_w=bn.weight,
        bn_b=bn.bias,
        linear_w=fused_linear.weight,
        linear_b=fused_linear.bias,
    )
 
    return fused_linear
 
 
def fuse_bn_linear_weights(
    bn_rm, bn_rv, bn_eps, bn_w, bn_b, linear_w, linear_b
):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
 
    fused_w = linear_w * bn_scale.unsqueeze(0)
    fused_b = torch.addmv(linear_b, linear_w, bn_b - bn_rm * bn_scale)
 
    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)
 
 
import copy
 
import torch
from torch import nn
from torch import optim
 
 
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(8)
        self.conv = nn.Conv2d(8, 16, 3)
        # self.conv = nn.Conv2d(8, 16, 3, padding=1)
        # self.conv = nn.Conv2d(8, 16, 1)
        # self.relu = nn.ReLU()
 
    def forward(self, x):
        return self.conv(self.bn(x))
 
 
model = Model()
print(model)
 
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters())
 
for _ in range(10):
    input = torch.randn(4, 8, 5, 5)
    target = torch.randn(4, 16, 3, 3)
    pred = model(input)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
 
dummpy_inputs = torch.randn(16, 8, 256, 256)
model.eval()
model_fused = torch.nn.Sequential(
    model.bn,
    model.conv
)
model_fused.eval()
f1= model_fused(dummpy_inputs)
model_to_fuse = fuse_bn_conv_eval(model.bn, model.conv)
model_to_fuse.eval()
print (model_to_fuse)
f2= model_to_fuse(dummpy_inputs)
d = (f1 - f2).mean().item()
print("error:",d)
# print (f1-f2)
 
# output_onnx_name = 'test_net.onnx'
 
# ################ pytorch onnx 模型导出
# torch.onnx.export(model_to_fuse,
#     dummpy_inputs,
#     output_onnx_name,
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=11,
#     # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}}
# )