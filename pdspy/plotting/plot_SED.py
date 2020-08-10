from pdspy.constants.astronomy import Jy
from pdspy.constants.physics import c as c_l
import matplotlib.pyplot as plt

def plot_SED(spectra, model, SED=False, fig=None, model_color="g", 
        linewidth=1, fontsize="medium"):

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
