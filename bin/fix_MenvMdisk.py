#!/usr/bin/env python3

from pdspy.constants.astronomy import AU, M_sun
import matplotlib.pyplot as plt
import pdspy.table as table
import pdspy.modeling as modeling
import astropy.stats as stats
import astropy.table
import scipy.signal
import scipy.stats
import importlib
import corner
import numpy
import glob
import sys
import os

# Read in the configuration information.

import config
importlib.reload(config)

# Make sure the parameters are all ok.

config.parameters = modeling.check_parameters(config.parameters)

# Make a dictionary of the best fit parameters.

keys = []
for key in sorted(config.parameters.keys()):
    if not config.parameters[key]["fixed"]:
        keys.append(key)

# Get the chain and the likelihoods.

chain = numpy.load("chain.npy")

# Create a backup of the chain.

numpy.save("chain_backup.npy", chain)

# Fix the disk mass, as it was not calculated correctly.

for i in range(chain.shape[0]):
    for j in range(chain.shape[1]):
        params = dict(zip(keys,chain[i,j,:]))

        # Set all of the gaps and cavity values to 1 to mimic how the density 
        # was normalized before.

        for key in keys:
            if "delta_" in key:
                params[key] = 0. #because logdelta...
            elif key == "f_cav":
                params[key] = 1.
            elif key == "f_M_large": # in case we did a 2 layer disk.
                params[key] = 1.

        # Now generate the model.

        m = modeling.run_disk_model(config.visibilities, config.images, \
                config.spectra, params, config.parameters, plot=False, \
                ncpus=4, ncpus_highmass=4, source="V883Ori", \
                no_radiative_transfer=True)

        rho, theta, phi = numpy.meshgrid(m.grid.r*AU, m.grid.theta, m.grid.phi,\
                indexing='ij')

        disk_density, env_density = m.grid.density[0], m.grid.density[-1]

        disk_mass = (2*numpy.pi*scipy.integrate.trapz(scipy.integrate.trapz(\
                disk_density*rho**2*numpy.sin(theta), theta, axis=1), \
                rho[:,0,:],axis=0))[0]

        env_mass = (2*numpy.pi*scipy.integrate.trapz(scipy.integrate.trapz(\
                envelope_density*rho**2*numpy.sin(theta), theta, axis=1), \
                rho[:,0,:],axis=0))[0]

        for k, key in enumerate(keys):
            if key == "logM_disk":
                chain[i,j,k] = numpy.log10(disk_mass/M_sun)
            if key == "logM_env":
                chain[i,j,k] = numpy.log10(env_mass/M_sun)

# Write out the fixed chain and pos.npy.

pos = chain[:,-1,:]

numpy.save("chain.npy", chain)
numpy.save("pos.npy", pos)

