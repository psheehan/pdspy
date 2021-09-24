import pdspy.interferometry as uv
import pdspy.imaging as im
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy

def plot_pvdiagram(visibilities, model, parameters, params, index=0, \
        plot_vis=False, image="data", contours="model", \
        model_image="beam-convolve", maxiter=100, threshold=1., uvtaper=None, \
        weighting="natural", robust=2.0, length=100, width=9, \
        image_cmap="Blues", levels=None, fontsize="medium", fig=None):

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
                        visibilities["image_pixelsize"][index]- \
                        params["x0"]/\
                        visibilities["image_pixelsize"][index]
            else:
                x0 = visibilities["image_npix"][index]/2 - \
                        visibilities["x0"][index]/\
                        visibilities["image_pixelsize"][index]- \
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
        elif plot_type == "model":
            if model_image == "beam-convolve":
                model_image = model.images[visibilities["lam"][index]]
                model_image.header = visibilities["image"][index].header
            else:
                model.visibilities[visibilities["lam"][index]].weights = \
                    visibilities["data"][index].weights

                model_image = uv.clean(\
                        model.visibilities[visibilities["lam"][index]], \
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

        if i == 0:
            # Plot the data.

            ax.imshow(plot_image.image[0,:,:,0].T, origin="lower", \
                    interpolation="nearest", cmap=image_cmap)

            # Transform the axes appropriately.

            dx = plot_image.x[1] - plot_image.x[0]
            dv = (plot_image.velocity[1] - plot_image.velocity[0])/1e5

            nx = plot_image.image.shape[1]
            nv = plot_image.image.shape[2]

            transform_x = ticker.FuncFormatter(Transform(0, nx, dx, '%.0f"'))
            transform_y = ticker.FuncFormatter(Transform(0, nv, dv, '%.0f'))

            ticks_x = numpy.array([-6,-3,0,3,6])
            ticks_y = numpy.array([-3,-2,-1,0,1,2,3])

            ax.set_xticks(nx/2+ticks_x/dx+0.5)
            ax.set_yticks(nv/2+ticks_y/dv-0.5)
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

        else:
            # Contour the model data.

            if levels is None:
                levels = (numpy.arange(5)+0.5)/5. * plot_image.image.max()

            ax.contour(plot_image.image[0,:,:,0].T, colors="black", \
                    levels=levels)

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

