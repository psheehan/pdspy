#!/usr/bin/env python3

from pdspy.dust import *
import numpy

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

graphite_parallel = Dust()
graphite_parallel.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
graphite_parallel.set_density(2.24)
graphite_parallel.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

#2/3
graphite_perpendicular = Dust()
graphite_perpendicular.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
graphite_perpendicular.set_density(2.24)
graphite_perpendicular.calculate_optical_constants_on_wavelength_grid(\
        water_ice.lam)

#1/3
silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

species = [silicates,graphite_parallel,graphite_perpendicular]
abundances = numpy.array([65.,35.*1./3,35.*2./3])
print(abundances)

dust = mix_dust(species, abundances)

amin = 0.005e-4
amax = 1.000e-2
pl = 3.5

dust.calculate_size_distribution_opacity(amin, amax, pl, \
        coat_volume_fraction=0.0)

dust.write('draine_100um.hdf5')
