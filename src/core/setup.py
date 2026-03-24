import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# UPDATE THESE PATHS to match your machine
SDK_DIR = r"C:\Project\Optical_Flow_SDK_5.0.7"
PYBIND11_INCLUDE = r"C:\Users\finn\miniconda3\envs\blackwell_vfi\Lib\site-packages\pybind11\include"

setup(
    name='blackwell_ofa',
    ext_modules=[
        CUDAExtension(
            name='blackwell_ofa',
            sources=['blackwell_ofa.cpp'],
            include_dirs=[
                os.path.join(SDK_DIR, 'nvofinterface'),
                os.path.join(SDK_DIR, 'common'),
                PYBIND11_INCLUDE,
            ],
            # ADD THIS NEW LINE HERE:
            library_dirs=[
                os.path.join(SDK_DIR, 'Lib', 'x64'), 
            ],
            extra_compile_args={
                'cxx': ['/O2', '/std:c++17'],
                'nvcc': [
                    '-arch=sm_120', 
                    '-O3'
                ]
            },
            #libraries=['nvOfAPI64'] # It will now look for this file inside library_dirs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)