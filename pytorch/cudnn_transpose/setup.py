# setup.py
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_transpose',
    ext_modules=[
        CUDAExtension(
            'custom_transpose',
            sources=['custom_transpose.cpp', 'transpose_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
