from distutils.core import setup, Extension
import numpy

# define the extension module
libinterferometry = Extension('libinterferometry', 
        sources=['interferometry.cc'], include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[libinterferometry])
