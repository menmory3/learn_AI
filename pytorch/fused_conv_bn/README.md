# fuse_conv_bn
```python
model = fuse_conv_bn(model)
```

# Efficient ConvBN Blocks for Transfer Learning and Beyond

[论文](https://arxiv.org/abs/2305.11624)

[MMEngine GitHub Repository](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/efficient_conv_bn_eval.py)

[test case](https://github.com/open-mmlab/mmengine/blob/main/tests/test_model/test_efficient_conv_bn_eval.py#L52)

[how to use](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1757-L1762)

--cfg-options efficient_conv_bn_eval="[backbone]"

## other 
https://mmengine.readthedocs.io/en/latest/common_usage/save_gpu_memory.html
https://github.com/open-mmlab/mmengine/discussions/1252
https://github.com/open-mmlab/mmcv/pull/2807/files#diff-7546171755af3a2ab5bf99bd286ed3ff2492442b2a43e7cdefa7767778ea03b4
https://github.com/open-mmlab/mmcv/pull/2807