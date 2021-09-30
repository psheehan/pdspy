import pdspy.interferometry as uv
import matplotlib.pyplot as plt
import numpy

def plot_1D_visibilities(visibilities, model, parameters, params, index=0, \
        fig=None, plot_disk=False, color="k", markersize=8, linewidth=1, \
        line_color="g", disk_only_color="gray", fontsize="medium"):

    # Generate a figure and axes if not provided.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # Average the high resolution model radially.

    m1d = uv.average(model.visibilities[visibilities["lam"][index]+"_high"], \
            gridsize=10000, binsize=3500, radial=True)

    if plot_disk:
        m1d_disk = uv.average(model.visibilities[visibilities["lam"][index]+\
                "_disk"], gridsize=10000, binsize=3500, radial=True)

    # Calculate the error properly.

    real_samples = visibilities["data1d"][index].real + numpy.random.normal(\
            0, 1, (visibilities["data1d"][index].real.size, 1000)) * \
            1./visibilities["data1d"][index].weights**0.5

    imag_samples = visibilities["data1d"][index].imag + numpy.random.normal(\
            0, 1, (visibilities["data1d"][index].imag.size, 1000)) * \
            1./visibilities["data1d"][index].weights**0.5

    amp_samples = (real_samples**2 + imag_samples**2)**0.5

    amp_unc = (numpy.percentile(amp_samples, [50], axis=1) - \
            numpy.percentile(amp_samples, [16, 84], axis=1)) * \
            numpy.array([1,-1])[:,None]

    # Plot the visibilities.

    ax.errorbar(visibilities["data1d"][index].uvdist/1000, \
            visibilities["data1d"][index].amp*1000, \
            yerr=amp_unc*1000, \
            fmt="o", markersize=markersize, markeredgecolor=color, \
            markerfacecolor=color, ecolor=color)

    # Plot the best fit model

    if plot_disk:
        ax.plot(m1d_disk.uvdist/1000, m1d_disk.amp*1000, "--", \
                color=disk_only_color, linewidth=linewidth)

    ax.plot(m1d.uvdist/1000, m1d.amp*1000, "-", color=line_color, \
            linewidth=linewidth)

    # Adjust the plot and add axes labels.

    ax.axis([1,visibilities["data1d"][index].uvdist.max()/1000*3,0,\
            visibilities["data1d"][index].amp.max()*1.1*1000])

    ax.set_xscale("log", nonpositive='clip')

    ax.set_xlabel("Baseline [k$\lambda$]", fontsize=fontsize)
    ax.set_ylabel("Amplitude [mJy]", fontsize=fontsize)

    # Return the figure and axes.

    return fig, ax
