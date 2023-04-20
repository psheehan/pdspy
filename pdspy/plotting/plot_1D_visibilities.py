import pdspy.interferometry as uv
import matplotlib.pyplot as plt
import numpy

def plot_1D_visibilities(visibilities, model, parameters, params, index=0, \
        fig=None, plot_disk=False, color="k", markersize=8, linewidth=1, \
        line_color="g", disk_only_color="gray", fontsize="medium"):
    r"""
    Plot the 1D azimuthally averaged visibility data along with the specified model.

    Args:
        :attr:`visibilities` (`dict`):
            Dictionary containing the visibility data, typically as loaded by :code:`utils.load_config` and :code:`utils.load_data`.
        :attr:`model` (`modeling.Model`):
            The radiative transfer model that you would like to plot the visibilities of. Typically this is the output of modeling.run_disk_model.
        :attr:`parameters` (`dict`):
            The parameters dictionary in the config module as loaded in by :code:`utils.load_config`
        :attr:`params` (`dict`):
            The parameters of the model, typically as a dictionary mapping parameter keys from the :code:`parameters` dictionary to their values.
        :attr:`index` (`int`, optional):
            The visibilities dictionary typically contains a list of datasets. `index` indicates which one to plot.
        :attr:`fig` (`tuple`, `(matplotlib.Figure, matplotlib.Axes)`, optional):
            If you've already created a figure and axes to put the plot in, you can supply them here. Otherwise, `plot_1D_visibilities` will generate them for you. Default: `None`
        :attr:`plot_disk` (`bool`, optional):
            Should :code:`plot_1D_visibilities` show the disk-only contribution to the model? Default: `False`
        :attr:`color` (str, optional):
            The color to use for plotting the visibility data. Default: `"k"`
        :attr:`markersize` (`int`, optional):
            The size of the markers to use for plotting the visibility data. Default: `8`
        :attr:`linewidth` (int, optional):
            What linewidth to use for plotting the model. Default: 1
        :attr:`line_color` (str, optional):
            The color to use for plotting the model visibilities. Default: `"g"`
        :attr:`disk_only_color` (str, optional):
            The color to use for plotting the disk-only model visibilities. Default: `"gray"`
        :attr:`fontsize` (`str` or `int`):
            What fontsize to use for labels, ticks, etc. Default: `"medium"`

    Returns:
        :attr:`fig` (`matplotlib.Figure`):
            The matplotlib figure that was used for the plot.
        :attr:`ax` (`matplotlib.Axes`):
            The matplotlib axes that were used for the plot.
    """


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
            visibilities["data1d"][index].amp[:,0]*1000, \
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
