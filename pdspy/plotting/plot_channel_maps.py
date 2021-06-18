from ..interferometry import Visibilities, clean, average
from ..constants.physics import c
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy

def plot_channel_maps(visibilities, model, parameters, params, index=0, \
        plot_vis=False, fig=None, image="data", contours="model", \
        model_image="beam-convolve", maxiter=100, threshold=1., uvtaper=None, \
        weighting="natural", robust=2.0, \
        vmin=None, vmax=None, levels=None, negative_levels=None, \
        image_cmap="viridis", contours_colors=None, fontsize="medium", \
        show_velocity=True, show_beam=True, vis_color="b", vis_model_color="g",\
        show_xlabel=True, show_ylabel=True, skip=0, \
        auto_center_velocity=False, v_width=10., beamxy=(0.15,0.15), \
        show_colorbar=False, cax=None, colorbar_location='right', \
        colorbar_orientation='vertical', colorbar_size='10%', \
        colorbar_pad=0.01, units="Jy/beam", vis_marker="o"):

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
            elif plot_type == "weights":
                vis = Visibilities(visibilities["data1d"][index].u, \
                        visibilities["data1d"][index].v, \
                        visibilities["data1d"][index].freq, \
                        1./visibilities["data1d"][index].weights**0.5, \
                        1./visibilities["data1d"][index].weights**0.5,\
                        visibilities["data1d"][index].weights.copy())
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
                            weighting=weighting, robust=robust, \
                            convolution="expsinc", mfs=False, \
                            mode="spectralline", maxiter=maxiter, \
                            threshold=threshold, uvtaper=uvtaper)[0]
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
                        weighting=weighting, robust=robust, \
                        convolution="expsinc", mfs=False, mode="spectralline", \
                        maxiter=0, uvtaper=uvtaper)[0]

            # Get the appropriate scaling for the image.

            if units == "mJy/beam":
                scale = 1000
                colorbar_label = "mJy beam$^{-1}$"
            elif units == "uJy/beam":
                scale = 1e6
                colorbar_label = "$\mu$Jy beam$^{-1}$"
            else:
                scale = 1.
                colorbar_label = "Jy beam$^{-1}$"

            # Set vmin, vmax, and the levels for plotting contours.

            if i == 0:
                if vmin is None:
                    vmin = numpy.nanmin(plot_image.image*scale)
                if vmax is None:
                    vmax = numpy.nanmax(plot_image.image*scale)

            if i == 1:
                if levels is None:
                    levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])*\
                            plot_image.image.max()*scale

            # Get the correct range of pixels for making the sub-image.

            if plot_type in ["data","residuals"] or \
                    model_image != "beam-convolve":
                if "x0" in params:
                    x0 = -params["x0"]/visibilities["image_pixelsize"][index]
                else:
                    x0 = -parameters["x0"]["value"]/\
                            visibilities["image_pixelsize"][index]
                if "y0" in params:
                    y0 = params["y0"]/visibilities["image_pixelsize"][index]
                else:
                    y0 = parameters["y0"]["value"]/\
                            visibilities["image_pixelsize"][index]

                xmin = int(round(x0 + visibilities["image_npix"][index]/2 - \
                      visibilities["x0"][index]/\
                      visibilities["image_pixelsize"][index]+ \
                      ticks[0]/visibilities["image_pixelsize"][index]))
                xmax = int(round(x0 + visibilities["image_npix"][index]/2 - \
                      visibilities["x0"][index]/\
                      visibilities["image_pixelsize"][index]+ \
                      ticks[-1]/visibilities["image_pixelsize"][index]))

                ymin = int(round(y0 + visibilities["image_npix"][index]/2 + \
                      visibilities["y0"][index]/\
                      visibilities["image_pixelsize"][index]+ \
                      ticks[0]/visibilities["image_pixelsize"][index]))
                ymax = int(round(y0 + visibilities["image_npix"][index]/2 + \
                      visibilities["y0"][index]/\
                      visibilities["image_pixelsize"][index]+ \
                      ticks[-1]/visibilities["image_pixelsize"][index]))
            else:
                if model_image == "beam-convolve":
                    xmin, xmax = int(round(visibilities["image_npix"][index]/2+\
                            ticks[0]/visibilities["image_pixelsize"][index]+1\
                            )), int(round(visibilities["image_npix"][index]/2 +\
                            ticks[-1]/visibilities["image_pixelsize"][index]+1))
                    ymin, ymax = int(round(visibilities["image_npix"][index]/2+\
                            ticks[0]/visibilities["image_pixelsize"][index]+1\
                            )), int(round(visibilities["image_npix"][index]/2 +\
                            ticks[-1]/visibilities["image_pixelsize"][index]+1))

        # Get the correct starting point, and skip value if auto calculating
        # the velocity range.

        if auto_center_velocity:
            if "v_sys" in params:
                v_start = params["v_sys"] - v_width/2
                v_end = params["v_sys"] + v_width/2
                v_center = params["v_sys"]
            else:
                v_start = parameters["v_sys"] - v_width/2
                v_end = parameters["v_sys"] + v_width/2
                v_center = parameters["v_sys"]

            center = (numpy.abs(v/1e5 - v_center)).argmin()
            start = (numpy.abs(v/1e5 - v_start)).argmin()
            end = (numpy.abs(v/1e5 - v_end)).argmin()

            nchan = end - start + 1
            skip = int(nchan/visibilities["ncols"][index]/\
                    visibilities["nrows"][index] - 0.5)

            if visibilities["ncols"][index]*visibilities["nrows"][index] % 2 \
                    == 0:
                half = int(visibilities["ncols"][index]*\
                        visibilities["nrows"][index] / 2)
            else:
                half = int((visibilities["ncols"][index]*\
                        visibilities["nrows"][index] - 1) / 2)

            start = center - half*(skip + 1)
            end = center + half*(skip + 1)
        else:
            start = visibilities["ind0"][index]

        # Now loop through the channels and plot.

        for k in range(visibilities["nrows"][index]):
            for l in range(visibilities["ncols"][index]):
                ind = (k*visibilities["ncols"][index] + l)*(skip+1) + start

                # Turn off the axis if ind >= nchannels

                if ind >= v.size:
                    print('Index greater than array size, skipping',v.size,ind)
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
                                vis_color+vis_marker)
                    else:
                        implot = ax[k,l].imshow(plot_image.image[ymin:ymax,\
                                xmin:xmax,ind,0]*scale*scalefn(abs(v[ind]/1e5 -\
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

                        ax[k,l].set_xscale("log", nonpositive='clip')

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

                            xy = ((xmax-xmin)*beamxy[0], (ymax-ymin)*beamxy[1])

                            ax[k,l].add_artist(patches.Ellipse(xy=xy, \
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
                                xmin:xmax,ind,0]*scale > levels.min())[0]) > 0:
                            ax[k,l].contour(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0]*scale, levels=levels, \
                                    colors=contours_colors)

                        # Plot the negative contours, if requested.
                        if negative_levels is not None:
                            if len(numpy.where(plot_image.image[ymin:ymax,\
                                    xmin:xmax,ind,0]*scale < \
                                    negative_levels.max())[0]) > 0:
                                ax[k,l].contour(plot_image.image[ymin:ymax,\
                                        xmin:xmax,ind,0]*scale, \
                                        levels=negative_levels, \
                                        linestyles="--", colors=contours_colors)

        # Add a colorbar to the image.

        if not plot_vis and show_colorbar:
            # If no cax were provided, create them based on the options.

            user_specified_cbar = True

            if cax == None:
                ax_bbox = ax[0,visibilities["ncols"][index]-1].get_position()

                if colorbar_location == 'top':
                    cax = fig.add_axes([ax_bbox.x0, ax_bbox.y1+colorbar_pad, \
                            ax_bbox.width, float(colorbar_size[0:-1])/100*\
                            ax_bbox.height])

                    colorbar_orientation = "horizontal"
                else:
                    cax = fig.add_axes([ax_bbox.x1+colorbar_pad, ax_bbox.y0, \
                            float(colorbar_size[0:-1])/100*ax_bbox.width, \
                            ax_bbox.height])

                    colorbar_orientation = "vertical"

                user_specified_cbar = False

            # Now plot the colorbar.

            cbar = plt.colorbar(implot, cax=cax, \
                    orientation=colorbar_orientation)

            # And make some adustments.

            if not user_specified_cbar:
                if colorbar_location == 'top':
                    cax.xaxis.set_ticks_position('top')
                    cax.xaxis.set_label_position('top')

            cbar.set_label(colorbar_label, size=fontsize)

            cax.tick_params(axis="both", which="major", labelsize=fontsize)

    # Return the figure and axes.

    if show_colorbar:
        return fig, ax, cax
    else:
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
