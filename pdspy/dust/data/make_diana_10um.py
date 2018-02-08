#!/usr/bin/env python3

from pdspy.dust import *
import numpy

amorphous_carbon = Dust()
amorphous_carbon.set_optical_constants_from_henn("optical_constants/amorphous_carbon_zubko1996.txt")
amorphous_carbon.set_density(2.24)

silicates = Dust()
silicates.set_optical_constants_from_henn("optical_constants/amorphous_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(amorphous_carbon.lam)

species = [silicates,amorphous_carbon]
abundances = numpy.array([0.8,0.2])

dust = mix_dust(species, abundances, filling=0.75)

amin = 0.05e-4
amax = 1.000e-3
pl = 3.5

dust.calculate_size_distribution_opacity(amin, amax, pl, with_dhs=True, \
        coat_volume_fraction=0.0, nang=1)

dust.write('diana_10um.hdf5')
