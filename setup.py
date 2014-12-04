from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

libinterferometry = cythonize("pdspy/interferometry/libinterferometry.pyx")[0]
libimaging = cythonize("pdspy/imaging/libimaging.pyx")[0]
bhmie = Extension('pdspy.dust.bhmie', sources=['pdspy/dust/bhmie.f90'])
bhcoat = Extension('pdspy.dust.bhcoat', sources=['pdspy/dust/bhcoat.f90'])

setup(ext_modules=[libinterferometry, libimaging, bhmie, bhcoat])
