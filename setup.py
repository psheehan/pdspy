from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

libinterferometry = cythonize([Extension('pdspy.interferometry.libinterferometry',["pdspy/interferometry/libinterferometry.pyx"],libraries=["m"],extra_compile_args=['-ffast-math'])])[0]

#libinterferometry = cythonize("pdspy/interferometry/libinterferometry.pyx")[0]
#libimaging = cythonize("pdspy/imaging/libimaging.pyx")[0]

libimaging = cythonize([Extension('pdspy.imaging.libimaging',["pdspy/imaging/libimaging.pyx"],libraries=[],extra_compile_args=[])])[0]

bhmie = Extension('pdspy.dust.bhmie', sources=['pdspy/dust/bhmie.f90'])

bhcoat = Extension('pdspy.dust.bhcoat', sources=['pdspy/dust/bhcoat.f90'])

dmilay = Extension('pdspy.dust.dmilay', sources=['pdspy/dust/DMiLay.f90'])

read = cythonize([Extension('pdspy.radmc3d.read',["pdspy/radmc3d/read.pyx"],libraries=[],extra_compile_args=[])])[0]

setup(ext_modules=[libinterferometry, libimaging, bhmie, bhcoat, dmilay, read])
