from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='PLS_tool',
      version='0.1',
      description='Implementation of partial least square',
      author='Xiaoming Liu & Yanlin Yu',
      license='MIT',
      packages=['PLS'],
      install_requires=[
        'numpy',
        'pandas'
      ],
      zip_safe=False)