import pdspy.interferometry as uv
import matplotlib.pyplot as plt
import numpy

def plot_1D_visibilities(visibilities, model, parameters, params, index=0, \
        fig=None):

    # Generate a figure and axes if not provided.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # Average the high resolution model radially.

    m1d = uv.average(model.visibilities[visibilities["lam"][index]+"_high"], \
            gridsize=10000, binsize=3500, radial=True)

    # Plot the visibilities.

    ax.errorbar(visibilities["data1d"][index].uvdist/1000, \
            visibilities["data1d"][index].amp, \
            yerr=numpy.sqrt(1./visibilities["data1d"][index].weights),\
            fmt="ko", markersize=8, markeredgecolor="k")

    # Plot the best fit model

    ax.plot(m1d.uvdist/1000, m1d.amp, "g-")

    # Adjust the plot and add axes labels.

    ax.axis([1,visibilities["data1d"][index].uvdist.max()/1000*3,0,\
            visibilities["data1d"][index].amp.max()*1.1])

    ax.set_xscale("log", nonposx='clip')

    ax.set_xlabel("Baseline [k$\lambda$]")
    ax.set_ylabel("Amplitude [Jy]")

    # Return the figure and axes.

    return fig, ax
