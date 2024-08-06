from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='hamming_flat',
    ext_modules=[
        CppExtension(
            'hamming_flat',
            ['hamming_flat.cpp'],
            extra_compile_args=['-fopenmp'],  # 添加OpenMP支持
            extra_link_args=['-lgomp']       # 链接OpenMP库
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
