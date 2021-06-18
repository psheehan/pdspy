#!/usr/bin/env python3

from pdspy.dust import *
import numpy

dust = PAH()
dust.set_properties_from_draine("optical_constants/PAHneu_30.txt", \
        "optical_constants/PAHneu_30.txt","optical_constants/PAH_qion.txt")
dust.set_density(3.0)

amin = dust.a.min()
amax = dust.a.max()
pl = 3.5

dust.calculate_size_distribution_opacity(amin, amax, pl)

dust.write('PAH_MRN.hdf5')
