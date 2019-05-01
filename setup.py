from setuptools import setup
from Cython.Build import cythonize
import numpy

def readme():
    with open('README.rst', 'w+') as f:
        return f.read()

setup(name='PLS_tool',
      version='0.11',
      description='Implementation of partial least square',
      author='Xiaoming Liu & Yanlin Yu',
      ext_modules=cythonize("./PLS/PLS_cython.pyx"),
      include_dirs=numpy.get_include(),
      license='MIT',
      packages=['PLS'],
      install_requires=[
        'numpy',
        'pandas',
        'Cython',
        'Numba'

      ],
      zip_safe=False)