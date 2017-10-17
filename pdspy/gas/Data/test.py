#!/usr/bin/env python3

from pdspy.constants.math import pi
from pdspy.modeling import YSOModel
from pdspy.gas import Gas
import numpy
import matplotlib.pyplot as plt
import matplotlib

# Change a few of the plotting parameters.

matplotlib.rcParams['font.family'] = 'serif'

# Read in the gas.

g = Gas()
g.set_properties_from_lambda('co.dat')

# Make the model.

model = YSOModel()
model.set_spherical_grid(0.1,20.,100,2,2)
model.add_star()
model.add_disk(rmax=20., gas=g, abundance=1.0e-5)

# Save the model.

model.write_yso("test.hdf5")

