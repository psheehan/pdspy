#!/usr/bin/env python

from calc_opac import calc_opac
from mix_grains import mix_grains
from Dust_Grain import Dust_Grain
from read_henn import read_henn
from read_jena import read_jena
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy import array

# Read in the forsterite info.

lam,n,k = read_jena("optical_constants/forsterite.txt")

a=0.1e-4 # cm

rho_gr=3.27

forsterite = Dust_Grain(a,rho_gr,lam=lam,n=n,k=k)

forsterite.kabs,forsterite.ksca=calc_opac(forsterite)

# Read in the troilite info.

lam,n,k = read_henn("optical_constants/troilite.txt")

rho_gr=4.83

f = interp1d(lam,n)
n = f(forsterite.lam)
f = interp1d(lam,k)
k = f(forsterite.lam)

troilite = Dust_Grain(a,rho_gr,lam=forsterite.lam,n=n,k=k)

troilite.kabs,troilite.ksca=calc_opac(troilite)

# Mix the dust grains together.

grains = array([forsterite,troilite])
abund = array([0.6,0.4])

dust = mix_grains(a,grains,abund)

dust.kabs,dust.ksca=calc_opac(dust)

# Plot the results.

plt1 = plt.subplot(2,2,1)
plt1.semilogx(forsterite.lam,forsterite.n)
plt1.semilogx(troilite.lam,troilite.n)
plt1.semilogx(dust.lam,dust.n)

plt2 = plt.subplot(2,2,3)
plt2.loglog(forsterite.lam,forsterite.k)
plt2.loglog(troilite.lam,troilite.k)
plt2.loglog(dust.lam,dust.k)

plt3 = plt.subplot(2,2,2)
plt3.loglog(forsterite.lam,forsterite.kabs)
plt3.loglog(troilite.lam,troilite.kabs)
plt3.loglog(dust.lam,dust.kabs)

plt4 = plt.subplot(2,2,4)
plt4.loglog(forsterite.lam,forsterite.ksca)
plt4.loglog(troilite.lam,troilite.ksca)
plt4.loglog(dust.lam,dust.ksca)

plt.show()
