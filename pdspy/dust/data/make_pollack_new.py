#!/usr/bin/env python3

from pdspy.dust import *
import numpy

lam = numpy.logspace(-5.,1.,1000)

silicates = Dust()
silicates.set_optical_constants_from_draine("optical_constants/astronomical_silicates.txt")
silicates.set_density(3.3)
silicates.calculate_optical_constants_on_wavelength_grid(lam)

troilite = Dust()
troilite.set_optical_constants_from_henn("optical_constants/troilite.txt")
troilite.set_density(4.83)
troilite.calculate_optical_constants_on_wavelength_grid(lam)

organics = Dust()
organics.set_optical_constants_from_henn("optical_constants/organics.txt")
organics.set_density(1.5)
organics.calculate_optical_constants_on_wavelength_grid(lam)

water_ice = Dust()
water_ice.set_optical_constants_from_henn("optical_constants/water_ice.txt")
water_ice.set_density(0.92)
water_ice.calculate_optical_constants_on_wavelength_grid(lam)

species = [silicates,troilite,organics,water_ice]
#mass_fraction = numpy.array([6.4e-3,7.68e-4,2.13e-3,1.4e-3])
mass_fraction = numpy.array([3.41e-3,7.68e-4,4.13e-3,5.55e-3])
rho = numpy.array([silicates.rho,troilite.rho,organics.rho,water_ice.rho])
abundances = (mass_fraction/rho)/(mass_fraction/rho).sum()

dust = mix_dust(species, abundances)

# Create the dust generator class.

dust_gen = DustGenerator(dust)

dust_gen.write("pollack_new.hdf5")
