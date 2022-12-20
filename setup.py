from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('xi_sum_cython.pyx'), include_dirs=[numpy.get_include()])