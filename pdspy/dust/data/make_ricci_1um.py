#!/usr/bin/env python3

from pdspy.dust import *
import numpy

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

organics = Dust()
organics.set_optical_constants_from_henn("optical_constants/amorphous_carbon_zubko1996_extrapolated.txt")
organics.set_density(2.24)
organics.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

species = [silicates,organics,water_ice]
mass_fraction = numpy.array([2.64e-3,3.53e-3,5.55e-3])
rho = numpy.array([silicates.rho,organics.rho,water_ice.rho])
abundances = (mass_fraction/rho)/(mass_fraction/rho).sum()
print(abundances)

dust = mix_dust(species, abundances, filling=0.6)

amin = 0.1e-4
amax = 1.000e-4
pl = 3.5

dust.calculate_size_distribution_opacity(amin, amax, pl, with_dhs=True, \
        coat_volume_fraction=0.0)

dust.write('ricci_1um.hdf5')
