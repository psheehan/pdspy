#!/usr/bin/env python3

from pdspy.dust import *
import numpy

iron = Dust()
iron.set_optical_constants_from_henn("optical_constants/iron.txt")
iron.set_density(7.87)

olivine = Dust()
olivine.set_optical_constants_from_henn("optical_constants/olivine.txt")
olivine.set_density(3.49)

orthopyroxene = Dust()
orthopyroxene.set_optical_constants_from_henn("optical_constants/orthopyroxene.txt")
orthopyroxene.set_density(3.4)

troilite = Dust()
troilite.set_optical_constants_from_henn("optical_constants/troilite.txt")
troilite.set_density(4.83)

organics = Dust()
organics.set_optical_constants_from_henn("optical_constants/organics.txt")
organics.set_density(1.5)

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

species = [silicates,troilite,organics,water_ice]
#mass_fraction = numpy.array([6.4e-3,7.68e-4,2.13e-3,1.4e-3])
mass_fraction = numpy.array([3.41e-3,7.68e-4,4.13e-3,5.55e-3])
rho = numpy.array([silicates.rho,troilite.rho,organics.rho,water_ice.rho])
abundances = (mass_fraction/rho)/(mass_fraction/rho).sum()

dust = mix_dust(species, abundances)

amin = 0.005e-4
amax = 1.000e-4
pl = 3.5

dust.calculate_size_distribution_opacity(amin, amax, pl, \
        coat_volume_fraction=0.0)

dust.write('pollack_1um.hdf5')
