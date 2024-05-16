from setuptools import setup

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
        "pdspy.dust",\
        "pdspy.gas",\
        "pdspy.misc",\
        "pdspy.modeling",\
        "pdspy.plotting", \
        "pdspy.radmc3d",\
        "pdspy.utils"], \
        package_dir={\
        "pdspy.dust": 'pdspy/dust', \
        "pdspy.gas": 'pdspy/gas'}, \
        package_data={\
        'pdspy.dust': ['data/*','reddening/*.dat'], \
        'pdspy.gas': ['data/*.dat']}, \
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
        'hyperion','h5py','mpi4py','astropy','schwimmbad','dynesty',\
        'scikit-learn','dishes @ git+https://github.com/psheehan/dishes.git'])
