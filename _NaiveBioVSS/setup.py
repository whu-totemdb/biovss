from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "hausdorff_distance_naive_lsh",
        ["hausdorff_distance_naive_lsh.cpp"],
        extra_compile_args=['-O3', '-march=native', '-ffast-math'],
    ),
]

setup(
    name="hausdorff_distance_naive_lsh",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
