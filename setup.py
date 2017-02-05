#!/usr/bin/env python
import numpy as np

#from distutils.core import setup
import setuptools

setuptools.setup(name='emcrecon2d',
    version='1.0',
    author='Julien Lhermitte',
    description="Image Reconstruction 2D",
    include_dirs=[np.get_include()],
    author_email='lhermitte@bnl.gov',
#   install_requires=['six', 'numpy'],  # essential deps only
    keywords='Image Processing Analysis',
    license='BSD',
)
