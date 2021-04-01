#!/usr/bin/env python3

import pdspy.modeling as modeling
import pdspy.interferometry as uv
import pdspy.dust as dust
import pdspy.gas as gas

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

# Also set up the gas properties of the disk.

g = gas.Gas()
g.set_properties_from_lambda(gas.__path__[0]+"/data/co.dat")

gases = [g]

# Add a disk to the model.

m.add_dartois_pringle_disk(mass=0.0003, rmin=0.1, rmax=50., plrho=1., h0=5., \
        plh=1., dust=d, tmid0=20., tatm0=100, zq0=0.1, pltgas=0.25, delta=1, \
        gas=gases, abundance=[1.0e-4], freezeout=[20.], aturb=0.1)

# Finally, we need to set the wavelength grid.

m.grid.set_wavelength_grid(0.1, 1.0e5, 500, log=True)

# Now generate a plot of the number density.

r, theta = numpy.meshgrid(m.grid.r, m.grid.theta[::-1], indexing='ij')

with numpy.errstate(divide="ignore"):
    z = numpy.log10(m.grid.number_density[0][:,:,0])
    vmin = numpy.nanmax(z) - 10.
    vmax = numpy.nanmax(z)

fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection':'polar'})

ax.pcolor(theta, r, z, shading="auto", vmin=vmin, vmax=vmax)

ax.set_thetalim(0.,numpy.pi/2)

plt.show()

