#!/usr/bin/env python3

from pdspy.dust import Dust, DustGenerator
import numpy
import matplotlib
import matplotlib.pyplot as plt

# Maximum dust grain sizes.

a_max = numpy.logspace(0.,5.,1000)

# Grain size distribution power-law

p = numpy.linspace(2.5, 4.5, 10)

# Read in the dust generator class.

dust_gen = DustGenerator("diana_wice.hdf5")

# Change a few of the parameters to make the plot look nice.

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["text.usetex"] = "True"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{upgreek}"
matplotlib.rcParams["legend.fontsize"] = 14

# Start the plotting.

fig, ax = plt.subplots(nrows=1, ncols=1)

# Loop through each of the values for p.

for i in range(p.size):
    # Set up a list to put beta in.

    beta = []

    # Loop through and calculate beta.

    for j in range(a_max.size):
        # Get the dust generator properties.

        dust = dust_gen(a_max[j] / 1.0e4, p[i])

        # Calculate beta.

        tmp_beta = -(numpy.log10(dust.kabs[108])-numpy.log10(dust.kabs[102]))/\
                (numpy.log10(dust.lam[108]) - numpy.log10(dust.lam[102]))

        # Add to the array.

        beta.append(tmp_beta)

    # Plot beta.

    ax.semilogx(a_max, beta, "-", label="$p = {0:3.1f}$".format(p[i]))

# Add a legend.

plt.legend(loc="lower left")

# Adjust the figure.

ax.set_xlabel("$a_{max}$ [$\mu$m]")
ax.set_ylabel(r"$\beta$")

# Save the figure.

fig.savefig("beta_plot.pdf")
