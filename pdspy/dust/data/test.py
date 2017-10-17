#!/usr/bin/env python3

from pdspy.dust import *
import matplotlib.pyplot as plt
import numpy
import time

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
mass_fraction = numpy.array([6.4e-3,7.68e-4,2.13e-3,1.4e-3])
rho = numpy.array([silicates.rho,troilite.rho,organics.rho,water_ice.rho])
abundances = (mass_fraction/rho)/(mass_fraction/rho).sum()

dust1 = mix_dust(species, abundances)
dust2 = mix_dust(species, abundances)

amin = 0.005e-4
amax = 1.000e0
pl = 3.5

starttime = time.time()
dust1.calculate_size_distribution_opacity(amin, amax, pl, \
        coat_volume_fraction=0.0, nang=2)
endtime = time.time()
print(endtime - starttime)

pl = 3.0

starttime = time.time()
dust2.calculate_size_distribution_opacity(amin, amax, pl, \
        coat_volume_fraction=0.0, nang=2)
endtime = time.time()
print(endtime - starttime)

plt.loglog(dust1.lam, dust1.ksca, 'b-')
plt.loglog(dust2.lam, dust2.ksca, 'r-')
plt.show()
