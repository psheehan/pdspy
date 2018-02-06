#!/usr/bin/env python3

from pdspy.dust import *
import numpy

organics = Dust()
organics.set_optical_constants_from_henn("optical_constants/organics.txt")
organics.set_density(1.5)

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.5)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

species = [silicates,organics,water_ice]
mass_fraction = numpy.array([2.64e-3,3.53e-3,5.55e-3])
rho = numpy.array([silicates.rho,organics.rho,water_ice.rho])
abundances = (mass_fraction/rho)/(mass_fraction/rho).sum()

dust = mix_dust(species, abundances, rule="MaxGarn", filling=1.0e-1)

amin = 0.005e-4
amax = 1.000e-4
pl = 3.5

dust.calculate_opacity(1.0e1, coat_volume_fraction=0.0, nang=1)

dust.write('pollack_maxgarn_1um.hdf5')
