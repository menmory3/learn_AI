# setup.py
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [this_dir, "/opt/pytorch/pytorch/third_party/pybind11/include", "/opt/pytorch/apex/csrc"]
setup(
    name='cudnn_transpose',
    ext_modules=[
        CUDAExtension(
            'cudnn_transpose',
            sources=['transforms.cpp','Descriptors.cpp','Types.cpp'],
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
