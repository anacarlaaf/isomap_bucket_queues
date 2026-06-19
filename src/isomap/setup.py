from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# rm -f shortest_path.cpp && rm -rf build/
# python3 setup.py build_ext --inplace

# rm -f shortest_path.cpp && rm -rf build/
# CFLAGS="-fsanitize=address -g" LDFLAGS="-fsanitize=address" \
# python3 setup.py build_ext --inplace

ext = Extension(
    name="shortest_path",
    sources=["shortest_path.pyx"],
    include_dirs=[
        np.get_include(),
        ".",
    ],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3", "-march=native"],    
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="shortest_path",
    ext_modules=cythonize([ext], language_level=3),
)