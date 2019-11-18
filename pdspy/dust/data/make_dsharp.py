#!/usr/bin/env python3

from pdspy.dust import *
import numpy

troilite = Dust()
troilite.set_optical_constants_from_henn("optical_constants/troilite.txt")
troilite.set_density(4.83)

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(water_ice.lam)

organics = Dust()
organics.set_optical_constants_from_henn("optical_constants/organics.txt")
organics.set_density(1.5)

species = [silicates,troilite,organics,water_ice]
abundances = [0.1670,0.0258,0.4430,0.3642]

dust = mix_dust(species, abundances)

# Create the dust generator class.

dust_gen = DustGenerator(dust)

dust_gen.write("dsharp.hdf5")
