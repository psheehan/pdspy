from ..interferometry import Visibilities, clean, average
from ..constants.physics import c
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy

def plot_channel_maps(visibilities, model, parameters, params, index=0, \
        plot_vis=False, fig=None, image="data", contours="model", \
        model_image="beam-convolve", maxiter=100, threshold=1., \
        vmin=None, vmax=None, levels=None, negative_levels=None, \
        image_cmap="viridis", contours_colors=None, fontsize="medium", \
        show_velocity=True, show_beam=True, vis_color="b", vis_model_color="g",\
        show_xlabel=True, show_ylabel=True, skip=0):

    # Set up the figure if none was provided.

    if fig == None:
        fig, ax = plt.subplots(nrows=visibilities["nrows"][index], \
                ncols=visibilities["ncols"][index], figsize=(10,9))
    else:
        fig, ax = fig

    # Calculate the velocity for each image.

    if plot_vis:
        v = c * (float(visibilities["freq"][index])*1.0e9 - \
                visibilities["data"][index].freq)/ \
                (float(visibilities["freq"][index])*1.0e9)
    else:
        v = c * (float(visibilities["freq"][index])*1.0e9 - \
                visibilities["image"][index].freq)/ \
                (float(visibilities["freq"][index])*1.0e9)

    # Set the ticks.

    ticks = visibilities["image_ticks"][index]

    for i in range(2):
        # First pass through plot the image, second pass through, plot contours.
        if i == 0:
            plot_type = image
        else:
            plot_type = contours

            # If we don't want to plot contours, just skip the second round.
            if plot_type == "none":
                continue

        # Get the correct dataset for what we are plotting (visibilities vs. 
        # image, data vs. model vs. residuals)

        if plot_vis:
            if plot_type == "data":
                vis = visibilities["data1d"][index]
            elif plot_type == "model":
                vis = model.visibilities[visibilities["lam"][index]+"1D"]
            elif plot_type == "residuals":
                residuals = Visibilities(visibilities["data"][index].u, \
                        visibilities["data"][index].v, \
                        visibilities["data"][index].freq, \
                        visibilities["data"][index].real.copy(), \
                        visibilities["data"][index].imag.copy(),\
                        visibilities["data"][index].weights)

                residuals.real -= model.visibilities[visibilities["lam"]\
                        [index]].real
                residuals.imag -= model.visibilities[visibilities["lam"]\
                        [index]].imag

                vis = average(residuals, gridsize=20, radial=True, log=True,\
                    logmin=residuals.uvdist[numpy.nonzero(residuals.uvdist)].\
                    min()*0.95, logmax=residuals.uvdist.max()*1.05, \
                    mode="spectralline")
        else:
            # Get the correct image/contours for plotting.

            if plot_type == "data":
                plot_image = visibilities["image"][index]
            elif plot_type == "model":
                if model_image == "beam-convolve":
                    plot_image = model.images[visibilities["lam"][index]]
                else:
                    model.visibilities[visibilities["lam"][index]].weights = \
                        visibilities["data"][index].weights

                    plot_image = clean(\
                            model.visibilities[visibilities["lam"][index]], \
                            imsize=visibilities["image_npix"][index], \
                            pixel_size=visibilities["image_pixelsize"][index], \
                            weighting="natural", convolution="expsinc", \
                            mfs=False, mode="spectralline", maxiter=maxiter, \
                            threshold=threshold)[0]
            elif plot_type == "residuals":
                residuals = Visibilities(visibilities["data"][index].u, \
                        visibilities["data"][index].v, \
                        visibilities["data"][index].freq, \
                        visibilities["data"][index].real.copy(), \
                        visibilities["data"][index].imag.copy(),\
                        visibilities["data"][index].weights)

                residuals.real -= model.visibilities[visibilities["lam"]\
                        [index]].real
                residuals.imag -= model.visibilities[visibilities["lam"]\
                        [index]].imag

                plot_image = clean(residuals, \
                        imsize=visibilities["image_npix"][index], \
                        pixel_size=visibilities["image_pixelsize"][index], \
                        weighting="natural", convolution="expsinc", \
                        mfs=False, mode="spectralline", maxiter=0)[0]

            # Set vmin, vmax, and the levels for plotting contours.

            if i == 0:
                if vmin is None:
                    vmin = numpy.nanmin(plot_image.image)
                if vmax is None:
                    vmax = numpy.nanmax(plot_image.image)

            if i == 1:
                if levels is None:
                    levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])*\
                            plot_image.image.max()

            # Get the correct range of pixels for making the sub-image.

            if plot_type == "data":
                if "x0" in params:
                    xmin, xmax = int(round(visibilities["image_npix"][index]/2+\
                          visibilities["x0"][index]/\
                          visibilities["image_pixelsize"][index]+ \
                          params["x0"]/visibilities["image_pixelsize"][index]+ \
                          ticks[0]/visibilities["image_pixelsize"][index])), \
                          int(round(visibilities["image_npix"][index]/2+\
                          visibilities["x0"][index]/\
                          visibilities["image_pixelsize"][index]+ \
                          params["x0"]/visibilities["image_pixelsize"][index]+ \
                          ticks[-1]/visibilities["image_pixelsize"][index]))
                else:
                    xmin, xmax = int(round(visibilities["image_npix"][index]/2+\
                          visibilities["x0"][index]/\
                          visibilities["image_pixelsize"][index]+ \
                          parameters["x0"]["value"]/\
                          visibilities["image_pixelsize"][index]+ \
                          ticks[0]/visibilities["image_pixelsize"][index])), \
                          int(round(visibilities["image_npix"][index]/2+\
                          visibilities["x0"][index]/\
                          visibilities["image_pixelsize"][index]+ \
                          parameters["x0"]["value"]/\
                          visibilities["image_pixelsize"][index]+ \
                          ticks[-1]/visibilities["image_pixelsize"][index]))
                if "y0" in params:
                    ymin, ymax = int(round(visibilities["image_npix"][index]/2-\
                          visibilities["y0"][index]/\
                          visibilities["image_pixelsize"][index]- \
                          params["y0"]/visibilities["image_pixelsize"][index]+ \
                          ticks[0]/visibilities["image_pixelsize"][index])), \
                          int(round(visibilities["image_npix"][index]/2-\
                          visibilities["y0"][index]/\
                          visibilities["image_pixelsize"][index]- \
                          params["y0"]/visibilities["image_pixelsize"][index]+ \
                          ticks[-1]/visibilities["image_pixelsize"][index]))
                else:
                    ymin, ymax = int(round(visibilities["image_npix"][index]/2-\
                          visibilities["y0"][index]/\
                          visibilities["image_pixelsize"][index]- \
                          parameters["y0"]["value"]/\
                          visibilities["image_pixelsize"][index]+ \
                          ticks[0]/visibilities["image_pixelsize"][index])), \
                          int(round(visibilities["image_npix"][index]/2-\
                          visibilities["y0"][index]/\
                          visibilities["image_pixelsize"][index]- \
                          parameters["y0"]["value"]/\
                          visibilities["image_pixelsize"][index]+ \
                          ticks[-1]/visibilities["image_pixelsize"][index]))
            else:
                xmin, xmax = int(round(visibilities["image_npix"][index]/2+1 + \
                        ticks[0]/visibilities["image_pixelsize"][index])), \
                        int(round(visibilities["image_npix"][index]/2+1 +\
                        ticks[-1]/visibilities["image_pixelsize"][index]))
                ymin, ymax = int(round(visibilities["image_npix"][index]/2+1 + \
                        ticks[0]/visibilities["image_pixelsize"][index])), \
                        int(round(visibilities["image_npix"][index]/2+1 + \
                        ticks[-1]/visibilities["image_pixelsize"][index]))

        # Now loop through the channels and plot.

        for k in range(visibilities["nrows"][index]):
            for l in range(visibilities["ncols"][index]):
                ind = (k*visibilities["ncols"][index] + l)*(skip+1) + \
                        visibilities["ind0"][index]

                # Turn off the axis if ind >= nchannels

                if ind >= v.size:
                    ax[k,l].set_axis_off()
                    continue

                # Get a fancy colormap if requested.

                if image_cmap == "BlueToRed":
                    if v[ind]/1e5 < params["v_sys"]:
                        cdict1 = {'red':   ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'green': ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'blue':  ((0.0, 1.0, 1.0),
                                            (1.0, 1.0, 1.0))}
                        blues = LinearSegmentedColormap('blues', cdict1)
                        cmap = blues
                    else:
                        cdict2 = {'red':   ((0.0, 1.0, 1.0),
                                            (1.0, 1.0, 1.0)),
                                  'green': ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0)),
                                  'blue':  ((0.0, 1.0, 1.0),
                                            (1.0, 0.0, 0.0))}
                        reds = LinearSegmentedColormap('reds', cdict2)
                        cmap = reds

                    scalefn = lambda x: (numpy.arctan(x*7-4)+1.3) / \
                            (numpy.arctan(1*7-4)+1.3)
                else:
                    cmap = image_cmap
                    scalefn = lambda x: 1.

                # On the first loop through, plot the image, and on the second
                # the contours.

                if i == 0:
                    # Plot the image.

                    if plot_vis:
                        ax[k,l].errorbar(vis.uvdist/1000, vis.amp[:,ind], \
                                yerr=1./vis.weights[:,ind]**0.5, fmt=\
                                vis_color+"o")
                    else:
                        ax[k,l].imshow(plot_image.image[ymin:ymax,xmin:xmax,\
                                ind,0]*scalefn(abs(v[ind]/1e5 - \
                                params["v_sys"])), origin="lower", \
                                interpolation="nearest", vmin=vmin, vmax=vmax, \
                                cmap=cmap)

                    # Add the velocity to the map.

                    if show_velocity:
                        if k == 0 and l == 0:
                            txt = ax[k,l].annotate(r"$v=%{0:s}$ km s$^{{-1}}$".\
                                    format(visibilities["fmt"][index]) % \
                                    (v[ind]/1e5), xy=(0.95,0.85), \
                                    xycoords='axes fraction', \
                                    horizontalalignment="right", \
                                    fontsize=fontsize)
                        else:
                            txt = ax[k,l].annotate(r"$%{0:s}$ km s$^{{-1}}$".\
                                    format(visibilities["fmt"][index]) % \
                                    (v[ind]/1e5), xy=(0.95,0.85), \
                                    xycoords='axes fraction', \
                                    horizontalalignment="right", \
                                    fontsize=fontsize)

                    # Fix the axes labels.

                    if plot_vis:
                        if show_xlabel:
                            ax[-1,l].set_xlabel("Baseline [k$\lambda$]", \
                                    fontsize=fontsize)

                        if plot_type == "residuals":
                            ax[k,l].set_ylim(-vis.amp.max()*1.1, \
                                    vis.amp.max()*1.1)
                        else:
                            ax[k,l].set_ylim(-0.5, vis.amp.max()*1.1)

                        ax[k,l].set_xscale("log", nonposx='clip')

                        # Turn off tick labels if we aren't in the correct
                        # row/column.
                        if l > 0:
                            ax[k,l].set_yticklabels([])
                        if k < visibilities["nrows"][index]-1:
                            ax[k,l].set_xticklabels([])
                    else:
                        # Set the ticks based on the config file.

                        transformx = ticker.FuncFormatter(Transform(xmin, xmax,\
                                visibilities["image_pixelsize"][index],'%.1f"'))
                        transformy = ticker.FuncFormatter(Transform(ymin, ymax,\
                                visibilities["image_pixelsize"][index],'%.1f"'))

                        ax[k,l].set_xticks((xmin + xmax)/2+\
                                ticks[1:-1]/visibilities["image_pixelsize"]\
                                [index]-xmin)
                        ax[k,l].set_yticks((ymin + ymax)/2+\
                                ticks[1:-1]/visibilities["image_pixelsize"]\
                                [index]-ymin)

                        ax[k,l].get_xaxis().set_major_formatter(transformx)
                        ax[k,l].get_yaxis().set_major_formatter(transformy)

                        # Adjust the tick labels.

                        ax[k,l].tick_params(labelsize=fontsize)

                        # Add a label to the x-axis.

                        if show_xlabel:
                            ax[-1,l].set_xlabel("$\Delta$RA", fontsize=fontsize)

                        # Show the size of the beam.

                        if show_beam:
                            bmaj = visibilities["image"][index].header["BMAJ"]/\
                                    abs(visibilities["image"][index].\
                                    header["CDELT1"])
                            bmin = visibilities["image"][index].header["BMIN"]/\
                                    abs(visibilities["image"][index].\
                                    header["CDELT1"])
                            bpa = visibilities["image"][index].header["BPA"]

                            ax[k,l].add_artist(patches.Ellipse(xy=(12.5,12.5), \
                                    width=bmaj, height=bmin, angle=(bpa+90), \
                                    facecolor="white", edgecolor="black"))

                        ax[k,l].set_adjustable('box')

                    if plot_vis:
                        if show_ylabel:
                            ax[k,0].set_ylabel("Amplitude [Jy]", \
                                    fontsize=fontsize)
                    else:
                        if show_ylabel:
                            ax[k,0].set_ylabel("$\Delta$Dec", fontsize=fontsize)

                # Plot the contours.

                elif i == 1:
                    if plot_vis:
                        ax[k,l].plot(vis.uvdist/1000, vis.amp[:,ind], \
                                vis_model_color+"-")
                    else:
                        if len(numpy.where(plot_image.image[ymin:ymax,\
                                xmin:xmax,ind,0] > levels.min())[0]) > 0:
                            ax[k,l].contour(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0], levels=levels, \
                                    colors=contours_colors)

                        # Plot the negative contours, if requested.
                        if negative_levels is not None:
                            if len(numpy.where(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0] < negative_levels.max())\
                                    [0]) > 0:
                                ax[k,l].contour(plot_image.image[ymin:ymax,\
                                        xmin:xmax,ind,0], \
                                        levels=negative_levels, \
                                        linestyles="--", colors=contours_colors)

    # Return the figure and axes.

    return fig, ax

# Define a useful class for plotting.

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin+1)/2)*self.dx)
