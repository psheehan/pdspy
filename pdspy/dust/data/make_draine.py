#!/usr/bin/env python3

from pdspy.dust import *
import numpy

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

#1/3
graphite_parallel = Dust()
graphite_parallel.set_optical_constants_from_draine("optical_constants/graphite_parallel_0.01.txt")
graphite_parallel.set_density(2.24)
graphite_parallel.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

#2/3
graphite_perpendicular = Dust()
graphite_perpendicular.set_optical_constants_from_draine("optical_constants/graphite_perpendicular_0.01.txt")
graphite_perpendicular.set_density(2.24)
graphite_perpendicular.calculate_optical_constants_on_wavelength_grid(\
        water_ice.lam)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

species = [silicates,graphite_parallel,graphite_perpendicular]
abundances = numpy.array([0.65,0.35*1./3,0.35*2./3])
print(abundances)

dust = mix_dust(species, abundances)

# Create the dust generator class.

dust_gen = DustGenerator(dust)

dust_gen.write("draine.hdf5")
