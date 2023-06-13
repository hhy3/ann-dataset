import os
import sys
import platform

import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '1.0.0'

libraries = []
extra_objects = []
ext_modules = []

requirements = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='ann-dataset',
    version=__version__,
    description='Approximate Nearest Neighbor Search Dataset',
    author='hhy3',
    long_description="""Approximate Nearest Neighbor Search Dataset""",
    ext_modules=ext_modules,
    install_requires=requirements,
    packages=['ann_dataset'],
    zip_safe=False,
)
