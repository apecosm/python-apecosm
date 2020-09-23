#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
setup for packaging Apecosm "
"""

__docformat__ = "restructuredtext en"

import os
from setuptools import setup, find_packages

VERSION_FILE = 'VERSION'
version = open(VERSION_FILE).read().strip()

setup(
    name="Apecosm",
    version=version,
    author="APECOSM team",
    author_email="nicolas.barrier@ird.fr",
    maintainer='Nicolas Barrier',
    maintainer_email='nicolas.barrier@ird.fr',
    description="Python package for manipulating Apecosm model input and outputs",
    license="CeCILL", 
    keywords="ocean grid model transport",
    include_package_data=True,
    url="http://www.apecosm.org/",
    packages=find_packages(),
    install_requires=['docutils>=0.12',
                      'sphinx>=1.3.1',
                      'pylint>=1.4.2',
                      'pyenchant>=1.6.6',
                      'xarray>=0.9.6',
                      'pep8>=1.6.2',
                      'pyflakes>=0.9.2',
                      'check-manifest>=0.25',
                      'numpy>=1.9',
                      'netCDF4>=1.1', 
                     ],
    requires=['numpy(>=1.9.2)',
              'netcdf4(>1.1.9)',
             ],
    long_description=open('README.txt').read(),

    classifiers = [
        #"Development Status :: 5 - Production/Stable",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Free To Use But Restricted",  # .. todo::
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
    ],

    # ++ test_suite =
    # ++ download_url
    platforms=['linux', 'mac osx'],
    scripts = []
)
