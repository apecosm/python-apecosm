#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup for packaging Apecosm package"
"""

__docformat__ = "restructuredtext en"

import os
from setuptools import setup, find_packages, Extension
import subprocess
import re

""" Recovers the NetCDF configuration for NetCDF library """
def get_ncconfig():

    try:
        cmd = ["nc-config", "--all"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("The 'nc-config' command is not found. Make sure that NetCDF is installed")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        raise
    try:
        o, e = proc.communicate()
    except:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Could not recover output of the 'nc-config' command")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        raise

    o = o.decode("utf-8")
    o = o.split("\n")

    regex = re.compile(" *--.* *-> *")
    output = {}
    for l in o:
        if(regex.match(l)): 
            key, val = l.split("->")
            key = key.replace("--", "").strip()
            val = val.strip()
            output[key] = val

    return output

params = get_ncconfig()
ext_cmodule = Extension('apecosm_clib', ['apecosm/src/apecosm_clib.c'], library_dirs=[params["libdir"]], libraries=["netcdf"])

VERSION_FILE = 'VERSION'
with open(VERSION_FILE) as fv:
    version = fv.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="apecosm",
    version=version,
    author="Apecosm team",
    author_email="nicolas.barrier@ird.fr",
    maintainer='Nicolas Barrier',
    maintainer_email='nicolas.barrier@ird.fr',
    description="Python package for the analysis of Apecosm outputs",
    long_description_content_type="text/markdown",
    keywords="ocean; grid model; ocean ecosystem; biology; deb",
    include_package_data=True,
    package_data = {'apecosm': ['resources/*']},
    url="https://github.com/apecosm/python-apecosm",
    packages=find_packages(),
    install_requires=['xarray>=0.1',
                      'numpy>=1.9',
                      'netCDF4>=1.1', 
                      'matplotlib>=1.4',
                      'cartopy',
                      'pandas'
                     ],

    long_description = long_description,

    classifiers = [
        #"Development Status :: 5 - Production/Stable",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
   

    ext_modules = [ext_cmodule],

    # ++ test_suite =
    # ++ download_url
    platforms=['linux', 'mac osx'],
)
