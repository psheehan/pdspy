import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def plot_continuum_image(visibilities, model, parameters, params, index=0, \
        fig=None):

    # If no figure is provided, create one.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # Calculate the minimum and maximum pixels to show of the image.

    ticks = visibilities["image_ticks"][index]

    if "x0" in params:
        x0 = params["x0"]/visibilities["image_pixelsize"][index]
    else:
        x0 = parameters["x0"]["value"]/visibilities["image_pixelsize"][index]
    if "y0" in params:
        y0 = params["y0"]/visibilities["image_pixelsize"][index]
    else:
        y0 = parameters["y0"]["value"]/visibilities["image_pixelsize"][index]

    xmin = int(round(x0 + visibilities["image_npix"][index]/2 + \
            visibilities["x0"][index]/visibilities["image_pixelsize"][index]+ \
            ticks[0]/visibilities["image_pixelsize"][index]))
    xmax = int(round(x0 + visibilities["image_npix"][index]/2+\
            visibilities["x0"][index]/visibilities["image_pixelsize"][index]+ \
            ticks[-1]/visibilities["image_pixelsize"][index]))

    ymin = int(round(y0 + visibilities["image_npix"][index]/2-\
            visibilities["y0"][index]/visibilities["image_pixelsize"][index]+ \
            ticks[0]/visibilities["image_pixelsize"][index]))
    ymax = int(round(y0 + visibilities["image_npix"][index]/2-\
            visibilities["y0"][index]/visibilities["image_pixelsize"][index]+ \
            ticks[-1]/visibilities["image_pixelsize"][index]))

    # Plot the image.

    ax.imshow(visibilities["image"][index].image[ymin:ymax,xmin:xmax,0,0], \
            origin="lower", interpolation="nearest", cmap="jet")

    # Get the model image to show.

    model_image = model.images[visibilities["lam"][index]]

    # Get the appropriate trim for the model.

    xmin, xmax = int(round(visibilities["image_npix"][index]/2+1 + \
            ticks[0]/visibilities["image_pixelsize"][index])), \
            int(round(visibilities["image_npix"][index]/2+1 + \
            ticks[-1]/visibilities["image_pixelsize"][index]))
    ymin, ymax = int(round(visibilities["image_npix"][index]/2+1 + \
            ticks[0]/visibilities["image_pixelsize"][index])), \
            int(round(visibilities["image_npix"][index]/2+1 + \
            ticks[-1]/visibilities["image_pixelsize"][index]))

    # Contour the model over the data.

    ax.contour(model_image.image[ymin:ymax,xmin:xmax,0,0], cmap="jet")

    # Transform the axes appropriately.

    transform = ticker.FuncFormatter(Transform(xmin, xmax, \
            visibilities["image_pixelsize"][index], '%.1f"'))

    ax.set_xticks(visibilities["image_npix"][index]/2+1+\
            ticks[1:-1]/visibilities["image_pixelsize"][index]-xmin)
    ax.set_yticks(visibilities["image_npix"][index]/2+1+\
            ticks[1:-1]/visibilities["image_pixelsize"][index]-ymin)
    ax.get_xaxis().set_major_formatter(transform)
    ax.get_yaxis().set_major_formatter(transform)

    # Adjust the plot and save it.

    ax.set_xlabel("$\Delta$RA")
    ax.set_ylabel("$\Delta$Dec")

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
