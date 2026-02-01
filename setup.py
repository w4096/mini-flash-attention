import glob
import os
import setuptools
import torch
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, BuildExtension
from pathlib import Path

cwd = os.path.dirname(os.path.abspath(__file__))

sources = [
    'csrc/api.cpp',
    *glob.glob('csrc/mfa/*.cu'),
    *glob.glob('csrc/mfa/*.cpp'),
    *glob.glob('csrc/mfa/cuda/*.cuh'),
    *glob.glob('csrc/mfa/cuda/*.h')
]

build_include_dirs = [
    Path(cwd) / 'csrc',
    Path(cwd) / 'csrc/mfa',
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    Path(cwd) / '3rd/cutlass/include',
]
build_libraries = ['cudart', 'nvrtc']
build_library_dirs = [f'{CUDA_HOME}/lib64']

ext_modules = [
    CUDAExtension(
        name='mini_flash_attention._C',
        sources=sources,
        include_dirs=build_include_dirs,
        libraries=build_libraries,
        library_dirs=build_library_dirs,
        extra_compile_args={
            'cxx': [
                '-std=c++20',
                '-O3',
            ],
            'nvcc': [
                '-std=c++20',
                "--generate-line-info",
                "--use_fast_math",
                "-Xptxas=-warn-spills",
            ],
        })
]

if __name__ == '__main__':
    setuptools.setup(
        name='mini_flash_attention',
        version='0.1.0',
        packages=setuptools.find_packages('.'),
        zip_safe=False,
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
    )
