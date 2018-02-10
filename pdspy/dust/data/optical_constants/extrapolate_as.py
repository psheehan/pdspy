#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pdspy.dust as dust
import numpy

d = dust.Dust()
d.set_optical_constants_from_henn("amorphous_silicates.txt")

new_lam = numpy.logspace(numpy.log10(d.lam[-1]),numpy.log10(d.lam[-1]*10),50)

lognpoly = numpy.polyfit(numpy.log10(d.lam[d.lam > 1.0e-1]), \
        numpy.log10(d.n[d.lam > 1.0e-1]), deg=1)

logkpoly = numpy.polyfit(numpy.log10(d.lam[d.lam > 1.0e-1]), \
        numpy.log10(d.k[d.lam > 1.0e-1]), deg=1)

new_n = 10.**numpy.polyval(lognpoly, numpy.log10(new_lam))
new_k = 10.**numpy.polyval(logkpoly, numpy.log10(new_lam))

new_lam = numpy.concatenate((d.lam, new_lam[1:]))
new_n = numpy.concatenate((d.n, new_n[1:]))
new_k = numpy.concatenate((d.k, new_k[1:]))

plt.loglog(d.lam, d.n, "b-")
plt.loglog(new_lam, new_n, "b--")

plt.loglog(d.lam, d.k, "r-")
plt.loglog(new_lam, new_k, "r--")

plt.show()

f = open("amorphous_silicates_extrapolated.txt", "w")

for i in range(len(new_lam)):
    f.write("{0:f}   {1:f}   {2:f}\n".format(new_lam[i]/1.0e-4, new_n[i], \
            new_k[i]))

f.close()
