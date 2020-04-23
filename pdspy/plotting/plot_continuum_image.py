from ..interferometry import Visibilities, clean
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy

def plot_continuum_image(visibilities, model, parameters, params, index=0, \
        fig=None, cmap="jet", fontsize="medium", image="data", \
        contours="model", model_image="beam-convolve", \
        weighting="robust", robust=2, maxiter=200, threshold=0.001, \
        cmap_contours="none", colors_contours="none", levels=None, \
        negative_levels=None, show_beam=False, beamxy=(0.1,0.1)):

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
                        maxiter=maxiter, threshold=threshold)[0]
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
                    mfs=False, mode="continuum", maxiter=0)[0]

        # Get the contour levels if none are specified.

        if i == 1:
            if levels is None:
                levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])*\
                        plot_image.image.max()

        # Get the right xmin, xmax, ymin, ymax.

        if plot_type == "data":
            if "x0" in params:
                x0 = -params["x0"]/visibilities["image_pixelsize"][index]
            else:
                x0 = parameters["x0"]["value"]/\
                        visibilities["image_pixelsize"][index]
            if "y0" in params:
                y0 = params["y0"]/visibilities["image_pixelsize"][index]
            else:
                y0 = -parameters["y0"]["value"]/\
                        visibilities["image_pixelsize"][index]

            xmin = int(round(x0 + visibilities["image_npix"][index]/2 + \
                    visibilities["x0"][index]/\
                    visibilities["image_pixelsize"][index]+ \
                    ticks[0]/visibilities["image_pixelsize"][index]))
            xmax = int(round(x0 + visibilities["image_npix"][index]/2+\
                    visibilities["x0"][index]/\
                    visibilities["image_pixelsize"][index]+ \
                    ticks[-1]/visibilities["image_pixelsize"][index]))

            ymin = int(round(y0 + visibilities["image_npix"][index]/2-\
                    visibilities["y0"][index]/\
                    visibilities["image_pixelsize"][index]+ \
                    ticks[0]/visibilities["image_pixelsize"][index]))
            ymax = int(round(y0 + visibilities["image_npix"][index]/2-\
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

            ax.imshow(plot_image.image[ymin:ymax,xmin:xmax,0,0], \
                    origin="lower", interpolation="nearest", cmap=cmap)

            if show_beam:
               bmaj = visibilities["image"][index].header["BMAJ"]/\
                       abs(visibilities["image"][index].header["CDELT1"])
               bmin = visibilities["image"][index].header["BMIN"]/\
                       abs(visibilities["image"][index].header["CDELT1"])
               bpa = visibilities["image"][index].header["BPA"]

               xy = ((xmax - xmin)*beamxy[0], (ymax - ymin)*beamxy[1])

               ax.add_artist(patches.Ellipse(xy=beamxy, width=bmaj, \
                       height=bmin, angle=(bpa+90), facecolor="white", \
                       edgecolor="black"))
        else:
            # Contour the model over the data.

            if len(numpy.where(plot_image.image[ymin:ymax,xmin:xmax,0,0] > \
                    levels.min())[0]) > 0:
                ax.contour(plot_image.image[ymin:ymax,xmin:xmax,0,0], \
                        cmap=cmap_contours, colors=colors_contours, \
                        levels=levels)

            if negative_levels is not None:
                if len(numpy.where(plot_image.image[ymin:ymax,xmin:xmax,0,0] < \
                        negative_levels.max())[0]) > 0:
                    ax.contour(plot_image.image[ymin:ymax,xmin:xmax,0,0], \
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
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)
