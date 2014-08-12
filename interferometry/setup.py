from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# define the extension module
#libinterferometry = Extension('libinterferometry', 
#        sources=['interferometry.cc'], include_dirs=[numpy.get_include()])

# run the setup
#setup(ext_modules=[libinterferometry])

setup(ext_modules = cythonize("libinterferometry.pyx"))
