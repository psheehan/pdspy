#!/usr/bin/env python
'''
Name  : Version, version.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : File for verifying version control of file versions and python version
'''

def package_version():
    return "0.1"

def python_version():
    from sys import version_info
    return version_info

def assertion():
    v = python_version()
    return ((v[0] >= 3) and (v[1] >= 6))

if __name__ == "__main__":
    print('Testing module\n')
    print("{}".format(__doc__))
