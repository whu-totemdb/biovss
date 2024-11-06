from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cal_distance',
    ext_modules=[
        CppExtension(
            'cal_distance',
            ['cal_distance_lsh.cpp'],
            extra_compile_args=['-fopenmp'],  # 添加OpenMP支持
            extra_link_args=['-lgomp']       # 链接OpenMP库
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
