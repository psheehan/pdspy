from setuptools import setup
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

from distutils.command.sdist import sdist
cmdclass={'sdist': sdist}

# Set up the extension modules.

libinterferometry = cythonize([\
        Extension('pdspy.interferometry.libinterferometry',\
            ["pdspy/interferometry/libinterferometry.pyx"],\
            libraries=["m"], extra_compile_args=['-ffast-math'])])[0]

libimaging = cythonize([Extension('pdspy.imaging.libimaging',\
        ["pdspy/imaging/libimaging.pyx"], libraries=[], \
        extra_compile_args=[])])[0]

bhmie = Extension('pdspy.dust.bhmie', sources=['pdspy/dust/bhmie.f90'])

bhcoat = Extension('pdspy.dust.bhcoat', sources=['pdspy/dust/bhcoat.f90'])

dmilay = Extension('pdspy.dust.dmilay', sources=['pdspy/dust/DMiLay.f90'])

read = cythonize([Extension('pdspy.radmc3d.read', ["pdspy/radmc3d/read.pyx"], \
        libraries=[], extra_compile_args=[])])[0]

# Now define the setup for the package.

setup(name="pdspy", \
        version="1.5.7", \
        author="Patrick Sheehan", \
        author_email="psheehan@northwestern.edu", \
        description="Radiative transfer modeling of protoplanetary disks", \
        long_description=open("README.md","r").read(), \
        long_description_content_type="text/markdown", \
        url="https://github.com/psheehan/pdspy", \
        packages=[\
        "pdspy",\
        "pdspy.constants", \
        "pdspy.dust",\
        "pdspy.gas",\
        "pdspy.imaging",\
        "pdspy.interferometry", \
        "pdspy.mcmc",\
        "pdspy.misc",\
        "pdspy.modeling",\
        "pdspy.plotting", \
        "pdspy.radmc3d",\
        "pdspy.spectroscopy",\
        "pdspy.stars",\
        "pdspy.statistics", \
        "pdspy.table", \
        "pdspy.utils"], \
        package_dir={\
        "pdspy.dust": 'pdspy/dust', \
        "pdspy.gas": 'pdspy/gas', \
        "pdspy.spectroscopy": 'pdspy/spectroscopy', \
        "pdspy.stars": 'pdspy/stars'}, \
        package_data={\
        'pdspy.dust': ['data/*','reddening/*.dat'], \
        'pdspy.imaging': ['*.pyx'], \
        'pdspy.interferometry': ['*.pyx'], \
        'pdspy.gas': ['data/*.dat'], \
        'pdspy.radmc3d': ['*.pyx'], \
        'pdspy.spectroscopy': ['btsettle_data/*.txt']}, \
        ext_modules=[libinterferometry, libimaging, bhmie, \
        bhcoat, dmilay, read], \
        scripts=[\
        'bin/config_template.py',\
        'bin/disk_model.py',\
        'bin/disk_model_powerlaw.py',\
        'bin/flared_model.py',\
        'bin/flared_model_ptsampler.py',\
        'bin/flared_model_nested.py'], \
        install_requires=['numpy','scipy','matplotlib','emcee','corner',\
        'hyperion','h5py','mpi4py','Cython','astropy','schwimmbad','dynesty',\
        'scikit-learn'], \
        cmdclass=cmdclass)

