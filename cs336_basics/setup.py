from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "bpe_ops",
        ["bpe_ops.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-pthread"], # Linux/Mac
        # extra_compile_args=["/O2", "/std:c++17"], # Windows 用户用这个
    ),
]

setup(
    name="bpe_ops",
    version="0.1",
    ext_modules=ext_modules,
)