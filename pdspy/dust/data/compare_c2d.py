#!/usr/bin/env python3

from pdspy.dust import Dust, DustGenerator
import matplotlib
import matplotlib.pyplot as plt

# List of the files to be plotted.

species_list = ["pollack_1um.hdf5", "pollack_10um.hdf5", "draine_1um.hdf5", \
        "c2d.hdf5"]

# Change a few of the parameters to make the plot look nice.

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["text.usetex"] = "True"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{upgreek}"
matplotlib.rcParams["legend.fontsize"] = 14

# Start the plotting.

fig, ax = plt.subplots(nrows=2, ncols=2)

for i, species in enumerate(species_list):
    dust = Dust()
    dust.set_properties_from_file(species)

    # If we're using the Draine opacities, multiply by 100.

    if species == "draine_1um.hdf5":
        dust.kabs *= 100
        dust.ksca *= 100
        dust.kext *= 100

    # Make a label for each line.

    if species == "pollack_1um.hdf5":
        label = "Pollack et al. 1994, $a_{max} = 1$ $\mu$m"
    if species == "pollack_10um.hdf5":
        label = "Pollack et al. 1994, $a_{max} = 10$ $\mu$m"
    elif species == "draine_1um.hdf5":
        label = "70\% astronomical silicate, 30\% graphite"
    elif species == "c2d.hdf5":
        label = "c2d opacities"

    # Plot the opacities.

    ax[0,0].loglog(dust.lam*1e4, dust.kabs, label=label)
    ax[0,1].loglog(dust.lam*1e4, dust.ksca)
    ax[1,0].loglog(dust.lam*1e4, dust.kext)
    ax[1,1].semilogx(dust.lam*1e4, dust.albedo)

ax[0,0].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[0,0].set_ylabel("$\kappa_{abs}$ [cm$^2$ g$^{-1}$]")
ax[0,1].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[0,1].set_ylabel("$\kappa_{sca}$ [cm$^2$ g$^{-1}$]")
ax[1,0].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[1,0].set_ylabel("$\kappa_{ext}$ [cm$^2$ g$^{-1}$]")
ax[1,1].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[1,1].set_ylabel("Albedo")

ax[0,0].legend(loc="lower left", fontsize="small")

ax[0,0].axis([1e-1,1e5,1e-4,1e5])
ax[0,1].axis([1e-1,1e5,1e-4,1e5])
ax[1,0].axis([1e-1,1e5,1e-4,1e5])
ax[1,1].axis([1e-1,1e5,0,1])

fig.set_size_inches((10,10))
fig.subplots_adjust(wspace=0.25, left=0.08, top=0.97, right=0.98, bottom=0.06)
fig.savefig("c2d_plot.pdf")
