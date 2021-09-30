from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..interferometry import Visibilities, clean
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.stats
import numpy

def plot_continuum_image(visibilities, model, parameters, params, index=0, \
        fig=None, cmap="jet", fontsize="medium", image="data", \
        contours="model", model_image="beam-convolve", \
        weighting="robust", robust=2, maxiter=200, threshold=0.001, 
        uvtaper=None, cmap_contours="none", colors_contours="none",levels=None,\
        negative_levels=None, show_beam=False, beamxy=(0.1,0.1), \
        show_colorbar=False, cax=None, colorbar_location='right', \
        colorbar_orientation='vertical', colorbar_size='5%', colorbar_pad=0.05,\
        units="Jy/beam"):

    # If no figure is provided, create one.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # If no contours colormap is provided, use the same as image.

    if cmap_contours == "none" and colors_contours == "none":
        cmap_contours = cmap
        colors_contours = None
    elif cmap_contours == "none" and colors_contours != "none":
        cmap_contours = None
    elif cmap_contours != "none" and colors_contours == "none":
        colors_contours = None
    elif cmap_contours != "none" and colors_contours != "none":
        colors_contours = None

    # Get the ticks.

    ticks = visibilities["image_ticks"][index]

    # Make the plot.

    for i in range(2):
        # First pass through plot the image, second pass through, plot contours.
        if i == 0:
            plot_type = image
        else:
            plot_type = contours

            # If we don't want to plot contours, just skip the second round.
            if plot_type == "none":
                continue

        # Get the appropriate image (data, model, residual) for plotting.

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
                        convolution="expsinc", mfs=False, mode="continuum", \
                        maxiter=maxiter, threshold=threshold, \
                        uvtaper=uvtaper)[0]
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
                    weighting=weighting, robust=robust, convolution="expsinc", \
                    mfs=False, mode="continuum", maxiter=maxiter, \
                    uvtaper=uvtaper)[0]

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

        # Get the contour levels if none are specified.

        if i == 1:
            if levels is None:
                levels = numpy.array([0.1, 0.3, 0.5, 0.7, 0.9])*\
                        plot_image.image.max()*scale

                # Only include levels above some detection threshold.

                rms = astropy.stats.mad_std(visibilities["image"][index].\
                        image[:,:,0,0])*scale * plot_image.image.max() / \
                        visibilities["image"][index].image.max()

                levels = levels[levels > 3*rms]

        # Get the right xmin, xmax, ymin, ymax.

        if plot_type == "data":
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
        elif plot_type in ["model","residuals"]:
            xmin, xmax = int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[-1]/visibilities["image_pixelsize"][index]))
            ymin, ymax = int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][index])), \
                    int(round(visibilities["image_npix"][index]/2+1 + \
                    ticks[-1]/visibilities["image_pixelsize"][index]))

        if i == 0:
            # Plot the image.

            implot = ax.imshow(plot_image.image[ymin:ymax,xmin:xmax,0,0]*\
                    scale, origin="lower", interpolation="nearest", cmap=cmap)

            if show_beam:
               bmaj = visibilities["image"][index].header["BMAJ"]/\
                       abs(visibilities["image"][index].header["CDELT1"])
               bmin = visibilities["image"][index].header["BMIN"]/\
                       abs(visibilities["image"][index].header["CDELT1"])
               bpa = visibilities["image"][index].header["BPA"]

               xy = ((xmax - xmin)*beamxy[0], (ymax - ymin)*beamxy[1])

               ax.add_artist(patches.Ellipse(xy=xy, width=bmaj, \
                       height=bmin, angle=(bpa+90), facecolor="white", \
                       edgecolor="black"))

            # Add a colorbar to the image.

            if show_colorbar:
                # If no cax were provided, create them based on the options.

                user_specified_cbar = True

                if cax == None:
                    divider = make_axes_locatable(ax)

                    cax = divider.append_axes(colorbar_location, \
                            size=colorbar_size, pad=colorbar_pad)

                    if colorbar_location in ['top','bottom']:
                        colorbar_orientation = "horizontal"
                    else:
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
                    elif colorbar_location == 'left':
                        cax.yaxis.set_ticks_position('left')
                        cax.yaxis.set_label_position('left')

                cbar.set_label(colorbar_label, size=fontsize)

                cax.tick_params(axis="both", which="major", labelsize=fontsize)
        else:
            # Contour the model over the data.

            if len(numpy.where(plot_image.image[ymin:ymax,xmin:xmax,0,0]*\
                    scale > levels.min())[0]) > 0:
                ax.contour(plot_image.image[ymin:ymax,xmin:xmax,0,0]*scale, \
                        cmap=cmap_contours, colors=colors_contours, \
                        levels=levels)

            if negative_levels is not None:
                if len(numpy.where(plot_image.image[ymin:ymax,xmin:xmax,0,0]*\
                        scale < negative_levels.max())[0]) > 0:
                    ax.contour(plot_image.image[ymin:ymax,xmin:xmax,0,0]*scale,\
                            cmap=cmap_contours, colors=colors_contours, \
                            levels=negative_levels, linestyles="--")


    # Transform the axes appropriately.

    transform = ticker.FuncFormatter(Transform(xmin, xmax, \
            visibilities["image_pixelsize"][index], '%.1f"'))

    ax.set_xticks((ticks[1:-1]-ticks[0])/visibilities["image_pixelsize"][index])
    ax.set_yticks((ticks[1:-1]-ticks[0])/visibilities["image_pixelsize"][index])
    ax.get_xaxis().set_major_formatter(transform)
    ax.get_yaxis().set_major_formatter(transform)

    # Adjust the plot and save it.

    ax.set_xlabel("$\Delta$R.A.", fontsize=fontsize)
    ax.set_ylabel("$\Delta$Dec.", fontsize=fontsize)

    # Adjust the label size as well.

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

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
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)
