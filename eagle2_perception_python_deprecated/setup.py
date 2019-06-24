from setuptools import setup, find_packages

setup(name='eaglemk4_nn_controller',
      version='0.0.1',
      description='Neural networks based controller for Eagle MK4 robot',
      url='http://github.com/r7vme/eaglemk4_nn_controller',
      author='Roma Sokolkov',
      author_email='rsokolkov@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[])
