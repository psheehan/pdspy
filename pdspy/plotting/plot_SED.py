from pdspy.constants.astronomy import Jy
from pdspy.constants.physics import c as c_l
import matplotlib.pyplot as plt

def plot_SED(spectra, model, SED=False, fig=None, model_color="g", 
        linewidth=1, fontsize="medium"):
    r"""
    Plot the SED generated by a radiative transfer modeling run with pdspy, typically generated by the output of `modeling.run_disk_model`.

    Args:
        :attr:`spectra` (list):
            List of `Spectrum` objects with data for the object you are studying.
        :attr:`model` (`modeling.Model`):
            The radiative transfer model that you would like to plot the SED of. The `Model.spectra` dictionary must include a `"SED"` key. Typically this is the output of modeling.run_disk_model.
        :attr:`SED` (bool, optional):
            Whether to plot as a traditional SED (`True`), i.e. as :math:`\nu F_{\nu}`, or as a spectrum, i.e. :math:`F_{\nu}`. Default: `False`
        :attr:`fig` (`tuple`, `(matplotlib.Figure, matplotlib.Axes)`, optional):
            If you've already created a figure and axes to put the plot in, you can supply them here. Otherwise, `plot_SED` will generate them for you. Default: `None`
        :attr:`model_color` (str, optional):
            The color to use for plotting the model SED. Default: `"g"`
        :attr:`linewidth` (int, optional):
            What linewidth to use for plotting the model. Default: 1
        :attr:`fontsize` (`str` or `int`):
            What fontsize to use for labels, ticks, etc. Default: `"medium"`

    Returns:
        :attr:`fig` (`matplotlib.Figure`):
            The matplotlib figure that was used for the plot.
        :attr:`ax` (`matplotlib.Axes`):
            The matplotlib axes that were used for the plot.
    """

    # If no axes are provided, create them.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # Plot the SED.

    for j in range(len(spectra["file"])):
        if spectra["bin?"][j]:
            if SED:
                ax.plot(spectra["data"][j].wave, \
                        c_l / spectra["data"][j].wave / 1.0e-4 * \
                        spectra["data"][j].flux * Jy, "k-")
            else:
                ax.plot(spectra["data"][j].wave, spectra["data"][j].flux, \
                        "k-")
        else:
            if SED:
                ax.errorbar(spectra["data"][j].wave, \
                        c_l / spectra["data"][j].wave / 1.0e-4 * \
                        spectra["data"][j].flux * Jy, fmt="ko", \
                        yerr=c_l / spectra["data"][j].wave / 1.0e-4 * \
                        spectra["data"][j].unc * Jy, markeredgecolor="k")
            else:
                ax.errorbar(spectra["data"][j].wave, \
                        spectra["data"][j].flux, fmt="ko", \
                        yerr=spectra["data"][j].unc, markeredgecolor="k")

    if len(spectra["file"]) > 0:
        if SED:
            ax.plot(model.spectra["SED"].wave, c_l/model.spectra["SED"].wave / \
                    1.0e-4 * model.spectra["SED"].flux * Jy, "-", \
                    color=model_color, linewidth=linewidth)
        else:
            ax.plot(model.spectra["SED"].wave, model.spectra["SED"].flux, "-", \
                    color=model_color, linewidth=linewidth)

    # Add axes labels and adjust the plot.

    if SED:
        ax.axis([0.1,1.0e4,1e-13,1e-8])
    else:
        ax.axis([0.1,1.0e4,1e-6,1e3])

    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')

    ax.set_xlabel("$\lambda$ [$\mu$m]", fontsize=fontsize)
    if SED:
        ax.set_ylabel(r"$\nu F_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]", \
                fontsize=fontsize)
    else:
        ax.set_ylabel(r"$F_{\nu}$ [Jy]", fontsize=fontsize)

    # Return the figure and axes.

    return fig, ax
