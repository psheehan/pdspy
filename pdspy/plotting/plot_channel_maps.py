from ..interferometry import Visibilities, clean, average, center as uvcenter
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
        velocity_fontsize=None, \
        show_velocity=True, show_beam=True, vis_color="b", vis_model_color="g",\
        show_xlabel=True, show_ylabel=True, skip=0, \
        auto_center_velocity=False, v_width=10., beamxy=(0.15,0.15), \
        show_colorbar=False, cax=None, colorbar_location='right', \
        colorbar_orientation='vertical', colorbar_size='10%', \
        colorbar_pad=0.01, units="Jy/beam", vis_marker="o", image_data=False):
    r"""
    Plot the millimeter channel maps, along with the best fit model.

    Args:
        :attr:`visibilities` (`dict`):
            Dictionary containing the visibility data, typically as loaded by 
            :code:`utils.load_config` and :code:`utils.load_data`.
        :attr:`model` (`modeling.Model`):
            The radiative transfer model that you would like to plot the 
            visibilities of. Typically this is the output of 
            modeling.run_disk_model.
        :attr:`parameters` (`dict`):
            The parameters dictionary in the config module as loaded in by 
            :code:`utils.load_config`
        :attr:`params` (`dict`):
            The parameters of the model, typically as a dictionary mapping 
            parameter keys from the :code:`parameters` dictionary to their 
            values.
        :attr:`index` (`int`, optional):
            The visibilities dictionary typically contains a list of datasets. 
            `index` indicates which one to plot.
        :attr:`plot_vis` (`bool`, optional):
            If `True`, plot the azimuthally averaged visibilities instead of 
            images. Defautl: `False`
        :attr:`fig` (`tuple`, `(matplotlib.Figure, matplotlib.Axes)`, optional):
            If you've already created a figure and axes to put the plot in, you 
            can supply them here. Otherwise, `plot_channel_maps` will 
            generate them for you. It will use 
            :code:`visibilities["nrows"][index]` rows and 
            :code:`visibilities["ncols"][index]` columns. Default: `None`
        :attr:`image` (`str`, optional):
            Should the image show the `"data"`, `"model"`, or `"residuals"`.
            Default: `"data"`
        :attr:`contours` (`str`, optional):
            Should the image show the `"data"`, `"model"`, or `"residuals"`.
            Or could also not show contours with `"none"`. Default: `"model"`
        :attr:`model_image` (`str`, optional):
            Should the model image be made by convolving a radiatie transfer
            generated image with an estimate of the beam (`"beam-convolve"`), 
            or by generating model visibilities at the correct baselines of the
            data and then using :code:`interferometry.clean` to generate an 
            image (`"CLEAN"`). Default: `"beam-convolve"`
        :attr:`weighting` (`str`, optional):
            What weighting scheme should the model image use if 
            :code:`model_image="CLEAN"`, `"natural"`, `"robust"`, or 
            `"uniform"`. Default:`"robust"`
        :attr:`robust` (`int`, optional):
            The robust parameter when :code:`weighting="robust"`. Default: 2
        :attr:`maxiter` (`int`, optional):
            The maximum number of CLEAN iterations to perform. Default: 2
        :attr:`threshold` (`float`, optional, Jy):
            The stopping threshold for the CLEAN algorithm. Default: 0.001 Jy
        :attr:`uvtaper` (`float`, optional, lambda):
            FWHM of the Gaussian tapering to apply to the weights when running
            CLEAN. Default: None
        :attr:`image_cmap` (`str`, optional):
            Which colormap to use for plotting the image. Default: `"jet"`
        :attr:`colors_contours` (`str` or `list`-like, optional):
            Colors to use for the contours. If `None`, use the default colormap.
            Default: `None`
        :attr:`levels` (`list` of `float`, optional):
            The flux levels at which to plot contours. If `None`, use 
            :code:`[0.1, 0.3, 0.5, 0.7, 0.9] x image.max()`. Default: `None`
        :attr:`negative_levels` (`list` of `float`, optional):
            A list of negative flux levels at which to plot dashed contours.
            Default: `None`
        :attr:`show_beam` (`bool`, optional):
            Should the plot show the size of the beam? Default: `False`
        :attr:`beamxy` (`tuple`, optional):
            If :code:`show_beam=True`, where should the beam be placed in the
            image, in units of axes fraction. Default: `(0.1,0.1)`
        :attr:`show_colorbar` (`bool`, optional):
            Should the plot show a colorbar for the image? Default: `False` 
        :attr:`cax` (`matplotlib.Axes`, optional):
            Pass an existing Axes class to use for the colorbar. If `None`, 
            :code:`plot_continuum_image` will generate a new one. 
            Default:`None`
        :attr:`colorbar_location` (`str`, optional):
            Should the colorbar be located to the `"right"` of the image or 
            on `"top"` of the image. Default: `'right'`
        :attr:`colorbar_size` (`str`, optional):
            The percent width of the image Axes that should be used for the 
            width of the colorbar Axes. Default: '5%'
        :attr:`colorbar_pad` (`str`, optional):
            How much padding should there be between the image Axes and the 
            colorbar Axes. Default: `0.05`
        :attr:`units` (`str`, optional):
            What units should the colorbar use? Options include `"Jy/beam"`, 
            `"mJy/beam"`, and `"uJy/beam"`. Default: `"Jy/beam"`
        :attr:`show_velocity` (`bool`, optional):
            Label each channel with the velocity of that channel? 
            Default: `True`
        :attr:`vis_marker` (`str`, optional)
            If :code:`plot_vis=True`, the marker to use to plot the visibility
            data. Default: `"o"`
        :attr:`vis_color` (`str`, optional)
            If :code:`plot_vis=True`, the color to use to plot the visibility
            data. Default: `"b"` 
        :attr:`vis_model_color` (`str`, optional)
            If :code:`plot_vis=True`, the color to use to plot the visibility 
            model. Default: `"g"`
        :attr:`show_xlabel` (`bool`, optional)
            Show ticks and axes labels on the x-axis? Default: `True`
        :attr:`show_ylabel` (`bool`, optional)
            Show ticks and axes labels on the y-axis? Default: `True`
        :attr:`skip` (`int`, optional)
            The number of channels to skip between each panel of the plot. 
            Default: `0`
        :attr:`auto_center_velocity` (`bool`, optional)
            Automatically center the velocities within the number of plot 
            panels provided. Default: `False`
        :attr:`v_width` (`float`, optional)
            If :code:`auto_center_velocity=True`, the width of the line that 
            you want to show in the channel maps. :code:`skip` will be 
            automatically determined based on the number of Axes provided. 
            Default: `10.`, kms
        :attr:`fontsize` (`str` or `int`):
            What fontsize to use for labels, ticks, etc. Default: `"medium"`

    Returns:
        :attr:`fig` (`matplotlib.Figure`):
            The matplotlib figure that was used for the plot.
        :attr:`ax` (`matplotlib.Axes`):
            The matplotlib axes that were used for the plot.
        :attr:`cax` (`matplotlib.Axes`, optional):
            The matplotlib axes that were used for the colorbar, if 
            :code:`show_colorbar=True`.
    """

    # Set up the figure if none was provided.

    if fig == None:
        fig, ax = plt.subplots(nrows=visibilities["nrows"][index], \
                ncols=visibilities["ncols"][index], figsize=(10,9))
    else:
        fig, ax = fig

    # If no velocity fontsize was specified, use fontsize

    if velocity_fontsize == None:
        velocity_fontsize = fontsize

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
                if image_data:
                    uncentered_vis = uvcenter(visibilities["data"][index], \
                            (-visibilities["x0"][index], \
                            -visibilities["y0"][index]))

                    plot_image = clean(\
                            uncentered_vis, \
                            imsize=visibilities["image_npix"][index], \
                            pixel_size=visibilities["image_pixelsize"][index], \
                            weighting=weighting, robust=robust, \
                            convolution="expsinc", mfs=False, \
                            mode="spectralline", maxiter=maxiter, \
                            threshold=threshold, uvtaper=uvtaper)[0]
                else:
                    plot_image = visibilities["image"][index]
            elif plot_type == "model":
                if model_image == "beam-convolve":
                    plot_image = model.images[visibilities["lam"][index]]
                    plot_image.image = plot_image.image[::-1,:,:,:]
                else:
                    model.visibilities[visibilities["lam"][index]].weights = \
                        visibilities["data"][index].weights

                    uncentered_vis = uvcenter(model.visibilities[\
                            visibilities["lam"][index]], \
                            (-visibilities["x0"][index], \
                            -visibilities["y0"][index]))

                    plot_image = clean(\
                            #model.visibilities[visibilities["lam"][index]], \
                            uncentered_vis, \
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

                uncentered_vis = uvcenter(residuals, \
                        (-visibilities["x0"][index], \
                        -visibilities["y0"][index]))

                plot_image = clean(uncentered_vis, \
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
                      visibilities["image_pixelsize"][index] + \
                      ticks[0]/visibilities["image_pixelsize"][index]))
                xmax = int(round(x0 + visibilities["image_npix"][index]/2 - \
                      visibilities["x0"][index]/\
                      visibilities["image_pixelsize"][index] + \
                      ticks[-1]/visibilities["image_pixelsize"][index]))

                ymin = int(round(y0 + visibilities["image_npix"][index]/2 + \
                      visibilities["y0"][index]/\
                      visibilities["image_pixelsize"][index] + \
                      ticks[0]/visibilities["image_pixelsize"][index]))
                ymax = int(round(y0 + visibilities["image_npix"][index]/2 + \
                      visibilities["y0"][index]/\
                      visibilities["image_pixelsize"][index] + \
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

        # Set vmin, vmax, and the levels for plotting contours.

        if not plot_vis:
            stop = start + (visibilities["nrows"][index] * \
                    visibilities["ncols"][index]-1) * (skip + 1) + 1
            if i == 0:
                if vmin is None:
                    #vmin = numpy.nanmin(plot_image.image[ymin:ymax,xmin:xmax,start:stop,0]*scale)
                    vmin = numpy.nanmin(visibilities["image"][index].image[ymin:ymax,xmin:xmax,start:stop,0]*scale)
                if vmax is None:
                    #vmax = numpy.nanmax(plot_image.image[ymin:ymax,xmin:xmax,start:stop,0]*scale)
                    vmax = numpy.nanmax(visibilities["image"][index].image[ymin:ymax,xmin:xmax,start:stop,0]*scale)

            if i == 1:
                if levels is None:
                    levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])*\
                            plot_image.image.max()*scale

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
                                    fontsize=velocity_fontsize)
                        else:
                            txt = ax[k,l].annotate(r"$%{0:s}$ km s$^{{-1}}$".\
                                    format(visibilities["fmt"][index]) % \
                                    (v[ind]/1e5), xy=(0.95,0.85), \
                                    xycoords='axes fraction', \
                                    horizontalalignment="right", \
                                    fontsize=velocity_fontsize)

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
