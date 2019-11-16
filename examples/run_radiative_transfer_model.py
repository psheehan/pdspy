#!/usr/bin/env python3

import pdspy.modeling as modeling
import pdspy.interferometry as uv
import pdspy.dust as dust

import matplotlib.pyplot as plt

import numpy

# Set up a radiative transfer model:

m = modeling.YSOModel()

# Add a grid. Spherical typically makes the most sense.

nr, ntheta, nphi = 100, 100, 2
rmin, rmax = 0.1, 300

m.set_spherical_grid(rmin, rmax, nr, ntheta, nphi, code="radmc3d")

# Add a star to the model.

m.add_star(mass=0.5, luminosity=1., temperature=4000.)

#Set up dust properties for the disk.

dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")

a_max = 100 # microns
p = 3.5

d = dust_gen(a_max / 1e4, p) # dust_gen wants units of cm

# Add a disk to the model.

m.add_disk(mass=0.01, rmin=0.1, rmax=50., plrho=1., h0=0.1, plh=1., dust=d)

# Finally, we need to set the wavelength grid.

m.grid.set_wavelength_grid(0.1, 1.0e5, 500, log=True)

# Now that we have the model set up, we need to run the radiative transfer 
# temperature calculation.

m.run_thermal(nphot=1e6, modified_random_walk=True, verbose=True, setthreads=1,\
        code="radmc3d")

# With that, we can now run an image.

m.run_image(name="870um", nphot=1e5, npix=256, pixelsize=0.01, lam="870", \
        incl=45, pa=30, dpc=140, code="radmc3d", verbose=True, \
        setthreads=2)

# Or visibilities:

m.run_visibilities(name="870um", nphot=1e5, npix=256, pixelsize=0.01, \
        lam="870", incl=45, pa=30, dpc=140, code="radmc3d", verbose=True, \
        setthreads=2)

# Or a spectrum:

m.set_camera_wavelength(numpy.logspace(-1, 4, 50))

m.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=45, pa=30, dpc=140, \
        code="radmc3d", verbose=True, setthreads=2)

# You can access the synthetic observations through:

m.images
m.visibilities
m.spectra

# For example:

plt.loglog(m.spectra["SED"].wave, m.spectra["SED"].flux, "b-")
plt.show()

# Or an image. Note that an image is actually a 4D array - the last two dimensions are for frequency (in case of image cubes) and polarization.

plt.imshow(m.images["870um"].image[:,:,0,0], origin="lower", \
        interpolation="nearest")
plt.show()

# Image objects also have a few other parts that may be of use:

m.images["870um"].x
m.images["870um"].y
m.images["870um"].freq

# Finally, lets average the visibility data azimuthally and plot it. Binsize is
# in units of klambda. As is m1d.uvdist

m1d = uv.average(m.visibilities["870um"], gridsize=10000, binsize=3500, \
        radial=True)

plt.semilogx(m1d.uvdist, m1d.amp, "-")

plt.show()

# Visibility classes also have a few other parts that may be of use:

m.visibilities["870um"].u
m.visibilities["870um"].v
m.visibilities["870um"].uvdist
m.visibilities["870um"].real
m.visibilities["870um"].imag
m.visibilities["870um"].amp
m.visibilities["870um"].weights
m.visibilities["870um"].freq

# Finally, if you want to access the actual density and temperature structures,
# you can find them here:

m.grid.r
m.grid.theta
m.grid.phi

m.grid.density[0] # should be (r, theta, phi)
m.grid.temperature[0]
