import torch
import torch.nn as nn
from pytorch_quantization import nn as quant_nn

try:
    from mmcv.ops import ModulatedDeformConv2dPack
    MMCV_AVAILABLE = True
except ImportError:
    MMCV_AVAILABLE = False
    ModulatedDeformConv2dPack = None

def _fuse_conv_bn(conv, bn):
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


# def fuse_conv_bn(module):
#     """Recursively fuse conv and bn in a module.

#     During inference, the functionary of batch norm layers is turned off
#     but only the mean and var alone channels are used, which exposes the
#     chance to fuse it with the preceding conv layers to save computations and
#     simplify network structures.

#     Args:
#         module (nn.Module): Module to be fused.

#     Returns:
#         nn.Module: Fused module.
#     """
#     last_conv = None
#     last_conv_name = None

#     for name, child in module.named_children():
#         if isinstance(child,
#                       (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
#             if last_conv is None:  # only fuse BN that is after Conv
#                 continue
#             fused_conv = _fuse_conv_bn(last_conv, child)
#             module._modules[last_conv_name] = fused_conv
#             # To reduce changes, set BN as Identity instead of deleting it.
#             module._modules[name] = nn.Identity()
#             last_conv = None
#         elif isinstance(child, nn.Conv2d):
#             last_conv = child
#             last_conv_name = name
#         else:
#             fuse_conv_bn(child)
#     return module
def fuse_conv_bn(module):
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:
                continue  # 只融合紧跟在 Conv 后面的 BN

            # 如果是标准 Conv 或 DCN，就执行融合
            if isinstance(last_conv, (nn.Conv2d, quant_nn.QuantConv2d)) or \
               (MMCV_AVAILABLE and isinstance(last_conv, ModulatedDeformConv2dPack)):

                fused_conv = _fuse_conv_bn(last_conv, child)
                module._modules[last_conv_name] = fused_conv
                module._modules[name] = nn.Identity()  # 替换为 Identity 避免结构变化
            last_conv = None
        elif isinstance(child, (nn.Conv2d, quant_nn.QuantConv2d)) or (MMCV_AVAILABLE and isinstance(child, ModulatedDeformConv2dPack)):
            # 记录当前卷积层
            last_conv = child
            last_conv_name = name
        else:
            # 递归处理子模块
            fuse_conv_bn(child)

    return module



import torch
import torch.nn as nn
 
from mmcv.cnn import ConvModule
 
 
def test_fuse_conv_bn():
    inputs = torch.rand((1, 3, 5, 5))
    modules = nn.ModuleList()
    modules.append(nn.BatchNorm2d(3))
    modules.append(ConvModule(3, 5, 3, norm_cfg=dict(type='BN')))
    modules.append(ConvModule(5, 5, 3, norm_cfg=dict(type='BN')))
    modules = nn.Sequential(*modules)
    fused_modules = fuse_conv_bn(modules)
    assert torch.equal(modules(inputs), fused_modules(inputs))