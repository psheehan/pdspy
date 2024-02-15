from pdspy.constants.astronomy import M_sun, AU
from pdspy.constants.physics import G
import pdspy.interferometry as uv
import pdspy.imaging as im
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy

def plot_pvdiagram(visibilities, model, parameters, params, index=0, \
        plot_vis=False, image="data", contours="model", \
        model_image="beam-convolve", maxiter=100, threshold=1., uvtaper=None, \
        weighting="natural", robust=2.0, length=100, width=9, \
        image_cmap="Blues", levels=None, fontsize="medium", fig=None, \
        ignore_velocities=None, curve_masses=[0.2,0.5,1.0]):
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
        :attr:`length` (`int`, optional):
            Length of the box, in pixels, to use to extract the PV diagram.
            Default: `100`
        :attr:`width` (`int`, optional):
            Width of the box, in pixels, to use to extract the PV diagram. 
            Default: `100`
        :attr:`image_cmap` (`str`, optional):
            Which colormap to use for plotting the image. Default: `"Blues"`
        :attr:`levels` (`list` of `float`, optional):
            The flux levels at which to plot contours. If `None`, use 
            :code:`[0.1, 0.3, 0.5, 0.7, 0.9] x image.max()`. Default: `None`
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

    # Loop through the visibilities and plot.

    for i in range(2):
        # First pass through plot the image, second pass through, plot contours.
        if i == 0:
            plot_type = image
        else:
            plot_type = contours

            # If we don't want to plot contours, just skip the second round.
            if plot_type == "none":
                continue
        
        # Get the centroid position.

        if plot_type in ["data","residuals"] or \
                model_image != "beam-convolve":
            if "x0" in params:
                x0 = visibilities["image_npix"][index]/2 - \
                        visibilities["x0"][index]/\
                        visibilities["image_pixelsize"][index] - \
                        params["x0"]/\
                        visibilities["image_pixelsize"][index]
            else:
                x0 = visibilities["image_npix"][index]/2 - \
                        visibilities["x0"][index]/\
                        visibilities["image_pixelsize"][index] - \
                        parameters["x0"]["value"]/\
                        visibilities["image_pixelsize"][index]
            if "y0" in params:
                y0 = visibilities["image_npix"][index]/2 + \
                        visibilities["y0"][index]/\
                        visibilities["image_pixelsize"][index] + \
                        params["y0"]/\
                        visibilities["image_pixelsize"][index]
            else:
                y0 = visibilities["image_npix"][index]/2 + \
                        visibilities["y0"][index]/\
                        visibilities["image_pixelsize"][index] + \
                        parameters["y0"]["value"]/\
                        visibilities["image_pixelsize"][index]
        else:
            if model_image == "beam-convolve":
                x0 = int(round(visibilities["image_npix"][index]/2))
                y0 = int(round(visibilities["image_npix"][index]/2))

        # Also grab the position angle.

        pa = (180. - (params["pa"]-90)) * numpy.pi / 180.

        # Get the correct image/contours for plotting.

        if plot_type == "data":
            plot_image = im.extract_pv_diagram(visibilities["image"][index], \
                    xy=(x0,y0), pa=pa, length=length, width=width)

            if ignore_velocities != None:
                scale = numpy.where(numpy.logical_or(plot_image.velocity/1e5 < \
                        ignore_velocities[0], plot_image.velocity/1e5 > \
                        ignore_velocities[1]), 1.0, 0.)
            else:
                scale = numpy.repeat(1.0, plot_image.velocity.size)

        elif plot_type == "model":
            if model_image == "beam-convolve":
                model_image = model.images[visibilities["lam"][index]]
                model_image.header = visibilities["image"][index].header
            else:
                model.visibilities[visibilities["lam"][index]].weights = \
                    visibilities["data"][index].weights

                uncentered_vis = uv.center(model.visibilities[\
                        visibilities["lam"][index]], \
                        (-visibilities["x0"][index], \
                        -visibilities["y0"][index]))

                model_image = uv.clean(\
                        #model.visibilities[visibilities["lam"][index]], \
                        uncentered_vis, \
                        imsize=visibilities["image_npix"][index], \
                        pixel_size=visibilities["image_pixelsize"][index], \
                        weighting=weighting, robust=robust, \
                        convolution="expsinc", mfs=False, \
                        mode="spectralline", maxiter=maxiter, \
                        threshold=threshold, uvtaper=uvtaper)[0]
                model_image.header = visibilities["image"][index].header

            # Extract the PV data.

            plot_image = im.extract_pv_diagram(model_image, xy=(x0,y0), \
                    pa=pa, length=length, width=width)
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

            residuals_image = uv.clean(residuals, \
                    imsize=visibilities["image_npix"][index], \
                    pixel_size=visibilities["image_pixelsize"][index], \
                    weighting=weighting, robust=robust, \
                    convolution="expsinc", mfs=False, mode="spectralline", \
                    maxiter=0, uvtaper=uvtaper)[0]

            residuals_image.header = visibilities["image"][index].header

            # Extract the PV data.

            plot_image = im.extract_pv_diagram(residuals_image, xy=(x0,y0), \
                    pa=pa, length=length, width=width)

        for iv in range(plot_image.velocity.size):
            plot_image.image[:,:,iv,:] *= scale[iv]

        if i == 0:
            # Plot the data.

            ax.imshow(plot_image.image[0,:,:,0].T, origin="lower", \
                    interpolation="nearest", cmap=image_cmap)

            # Transform the axes appropriately.

            dx = plot_image.x[1] - plot_image.x[0]
            dv = (plot_image.velocity[1] - plot_image.velocity[0])/1e5

            nx = plot_image.image.shape[1]
            nv = plot_image.image.shape[2]

            transform_x = ticker.FuncFormatter(Transform(0, plot_image.x[0], dx, '%.1f"'))
            transform_y = ticker.FuncFormatter(Transform(0, plot_image.velocity[0]/1e5, dv, '%.0f'))

            #ticks_x = numpy.array([-6,-3,0,3,6])
            ticks_x = visibilities["image_ticks"][index]
            #ticks_y = numpy.array([-3,-2,-1,0,1,2,3])
            print(plot_image.velocity)
            ticks_y = numpy.linspace(numpy.ceil((plot_image.velocity/1e5).min()), \
                    numpy.floor((plot_image.velocity/1e5).max()), \
                    int(numpy.floor((plot_image.velocity/1e5).max()) - \
                    numpy.ceil((plot_image.velocity/1e5).min())+1))
            print(ticks_y)

            """
            ax.set_xticks(nx/2+ticks_x/dx+0.5)
            ax.set_yticks(nv/2+ticks_y/dv-0.5)
            """
            ax.set_xticks((ticks_x - plot_image.x[0]) / dx)
            ax.set_yticks((ticks_y - plot_image.velocity[0]/1e5) / dv)
            ax.get_xaxis().set_major_formatter(transform_x)
            ax.get_yaxis().set_major_formatter(transform_y)

            # Add axes labels.

            ax.set_xlabel('Offset', fontsize=fontsize, labelpad=10)
            ax.set_ylabel('Velocity [km s$^{-1}$]', fontsize=fontsize, \
                    labelpad=10)

            ax.tick_params(axis="both", which="both", labelsize=fontsize)

            # Set the aspect ratio correctly.

            ax.set_aspect(nx/nv)

            # And the axes range.

            ax.set_xlim(0, nx-1)
            ax.set_ylim(0, nv-1)

            # Plot Keplerian rotation curves for a variety of masses.

            formats = ["-","--",":"]
            for ind, mass in enumerate(curve_masses):
                x = numpy.linspace(0, nx+1, 1000)

                r = (x - nx/2) * dx * 140 * AU

                v = numpy.sqrt(G*mass*M_sun / numpy.abs(r)) / 1e5 * \
                        numpy.sin(75.*numpy.pi/180)
                        #numpy.sin(params["i"]*numpy.pi/180)

                v[r > 0] = -v[r > 0]

                v = v / dv + (params["v_sys"] - \
                        plot_image.velocity[0]/1e5) / dv

                plt.plot(x[r > 0], v[r > 0], "r"+formats[ind], \
                        label="$M_* = {0:3.1f}$ M$_{{\odot}}$".format(mass))
                plt.plot(x[r < 0], v[r < 0], "r"+formats[ind])

            ax.legend(loc="upper right", fontsize=fontsize, handlelength=1)
        else:
            # Contour the model data.

            if levels is None:
                levels = (numpy.arange(5)+0.5)/5. * plot_image.image.max()

            ax.contour(plot_image.image[0,:,:,0].T, colors="black", \
                    levels=levels)

    return fig, ax

# Define a useful class for plotting.

class Transform:
    def __init__(self, xmin, xmin_val, dx, fmt):
        self.xmin = xmin
        self.xmin_val = xmin_val
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        #return self.fmt% ((x-(self.xmax-self.xmin+1)/2)*self.dx)
        return self.fmt% ( self.xmin_val + (x - self.xmin)*self.dx )

