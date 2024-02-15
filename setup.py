from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Set up the extension modules.

libinterferometry = cythonize([\
        Extension('pdspy.interferometry.libinterferometry',\
            ["pdspy/interferometry/libinterferometry.pyx"],\
            libraries=["m"], extra_compile_args=['-ffast-math'], \
            include_dirs=[np.get_include()], \
            define_macros=[('NPY_NO_DEPRECATED_API', 0)])])[0]

libimaging = cythonize([Extension('pdspy.imaging.libimaging',\
        ["pdspy/imaging/libimaging.pyx"], libraries=[], \
        extra_compile_args=[], include_dirs=[np.get_include()])])[0]

bhmie = Extension('pdspy.dust.bhmie', sources=['pdspy/dust/bhmie.f90'])

bhcoat = Extension('pdspy.dust.bhcoat', sources=['pdspy/dust/bhcoat.f90'])

dmilay = Extension('pdspy.dust.dmilay', sources=['pdspy/dust/DMiLay.f90'])

read = cythonize([Extension('pdspy.radmc3d.read', ["pdspy/radmc3d/read.pyx"], \
        libraries=[], extra_compile_args=[], \
        include_dirs=[np.get_include()])])[0]

# Now define the setup for the package.

setup(name="pdspy", \
        version="2.0.7", \
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
        #ext_modules=[libinterferometry, libimaging, bhmie, \
        #bhcoat, dmilay, read], \
        ext_modules=[libinterferometry, libimaging, read], \
        scripts=[\
        'bin/config_template.py',\
        'bin/upgrade_to_pdspy2.py',\
        'bin/generate_surrogate_model.py',\
        'bin/disk_model_emcee3.py',\
        'bin/disk_model_nested.py',\
        'bin/disk_model_dynesty.py',\
        'bin/disk_model_powerlaw.py',\
        'bin/flared_model_emcee3.py',\
        'bin/flared_model_nested.py', \
        'bin/flared_model_dynesty.py'], \
        install_requires=['numpy','scipy','matplotlib','emcee','corner',\
        'hyperion','h5py','mpi4py','Cython','astropy','schwimmbad','dynesty',\
        'scikit-learn'])
