import torch
import sys
sys.path.append('.')
import cudnn_transpose

class NHWCToNCHW_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = cudnn_transpose.cudnnNhwcToNchw(x)
        return y
 
    @staticmethod
    def backward(ctx, y_grad):
        x_grad = cudnn_transpose.cudnnNchwToNhwc(y_grad)
        return x_grad

class NCHWToNHWC_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = cudnn_transpose.cudnnNchwToNhwc(x)
        return y
 
    @staticmethod
    def backward(ctx, y_grad):
        x_grad = cudnn_transpose.cudnnNhwcToNchw(y_grad)
        return x_grad

class NHWCToNCHW(torch.nn.Module):
    def __init__(self):
        super(NHWCToNCHW, self).__init__()
    def forward(self, x):
        return NHWCToNCHW_Impl.apply(x)

class NCHWToNHWC(torch.nn.Module):
    def __init__(self):
        super(NCHWToNHWC, self).__init__()
    def forward(self, x):
        return NCHWToNHWC_Impl.apply(x)
