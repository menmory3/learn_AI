import torch
import sys
sys.path.append('.')
import custom_transpose

def test_transpose():
    x = torch.randn(32, 3, 224, 224).cuda()
    print("Original shape (NCHW):", x.shape)

    y = custom_transpose.transpose_nchw_to_nhwc(x)
    print("Converted shape (NHWC):", y.shape)

    # 使用 PyTorch 对比结果
    ref = x.permute(0, 2, 3, 1).contiguous()
    assert torch.allclose(y, ref), "Transpose result mismatch!"
    print("✅ Test passed!")

if __name__ == '__main__':
    test_transpose()