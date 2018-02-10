#!/usr/bin/env python3

from pdspy.dust import Dust, DustGenerator
import matplotlib
import matplotlib.pyplot as plt

# List of the files to be plotted.

#species_list = ["pollack_1um.hdf5", "pollack_10um.hdf5", "pollack_100um.hdf5", \
#        "pollack_1mm.hdf5", "pollack_1cm.hdf5", "pollack_10cm.hdf5"]
species_list = ["pollack_1um.hdf5", "pollack_10um.hdf5", "pollack_100um.hdf5", \
        "pollack_1mm.hdf5", "pollack_1cm.hdf5", "pollack_10cm.hdf5"]

# Maximum dust grain sizes.

a_max = [1., 10., 100., 1000., 10000., 100000.]

# Read in the dust generator class.

dust_gen = DustGenerator("pollack_new.hdf5")

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

    # Get the dust generator properties.

    dust1 = dust_gen(a_max[i] / 1.0e4, 2.5)

    # Make a label for each line.

    size = species.split('_')[1].split('.')[0]
    if (len(size.split('um')) == 2):
        label = r"$a_{max} = %s$ $\upmu$m" % size.split('um')[0]
    elif (len(size.split('mm')) == 2):
        label = "$a_{max} = %s$ mm" % size.split('mm')[0]
    elif (len(size.split('cm')) == 2):
        label = "$a_{max} = %s$ cm" % size.split('cm')[0]

    # Plot the opacities.

    ax[0,0].loglog(dust.lam*1e4, dust.kabs, label=label)
    ax[0,1].loglog(dust.lam*1e4, dust.ksca)
    ax[1,0].loglog(dust.lam*1e4, dust.kext)
    ax[1,1].semilogx(dust.lam*1e4, dust.albedo)

    ax[0,0].loglog(dust1.lam*1e4, dust1.kabs, '--')
    ax[0,1].loglog(dust1.lam*1e4, dust1.ksca, '--')
    ax[1,0].loglog(dust1.lam*1e4, dust1.kext, '--')
    ax[1,1].semilogx(dust1.lam*1e4, dust1.albedo, '--')

ax[0,0].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[0,0].set_ylabel("$\kappa_{abs}$ [cm$^2$ g$^{-1}$]")
ax[0,1].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[0,1].set_ylabel("$\kappa_{sca}$ [cm$^2$ g$^{-1}$]")
ax[1,0].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[1,0].set_ylabel("$\kappa_{ext}$ [cm$^2$ g$^{-1}$]")
ax[1,1].set_xlabel(r"$\lambda$ [$\upmu$m]")
ax[1,1].set_ylabel("Albedo")

ax[0,0].legend(loc="lower left")

ax[0,0].axis([1e-1,1e5,1e-4,1e5])
ax[0,1].axis([1e-1,1e5,1e-4,1e5])
ax[1,0].axis([1e-1,1e5,1e-4,1e5])
ax[1,1].axis([1e-1,1e5,0,1])

fig.set_size_inches((10,10))
fig.subplots_adjust(wspace=0.25, left=0.08, top=0.97, right=0.98, bottom=0.06)
fig.savefig("pollack_plot.pdf")
