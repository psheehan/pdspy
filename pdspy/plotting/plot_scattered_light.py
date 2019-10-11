import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy

def plot_scattered_light(images, model, parameters, params, index=0, \
        fig=None):

    # If no axes are provided, create them.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5,4))
    else:
        fig, ax = fig

    # Scale the image appropriately.

    c = scale_image(images["data"][index], mode="arcsinh")

    # Show the image.

    ax.imshow(c[:,:,0,0], origin="lower", interpolation="nearest", cmap="gray")

    # Scale the model image.

    c = scale_image(model.images[images["lam"][index]], \
            mode=images["plot_mode"][index])

    # Contour the model image over the data.

    levels = numpy.array([0.05,0.25,0.45,0.65,0.85,1.0]) * \
            (c.max() - c.min()) + c.min()

    ax.contour(c[:,:,0,0], colors='gray', levels=levels)

    # Adjust the ticks.

    transform3 = ticker.FuncFormatter(Transform(0, images["npix"][index], \
            images["pixelsize"][index], '%.1f"'))

    ticks = images["ticks"][index]

    ax.set_xticks(images["npix"][index]/2+ticks/images["pixelsize"][index])
    ax.set_yticks(images["npix"][index]/2+ticks/images["pixelsize"][index])
    ax.get_xaxis().set_major_formatter(transform3)
    ax.get_yaxis().set_major_formatter(transform3)

    # Adjust the axes labels.

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
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)

# Define a function to scale an image to look nice.

def scale_image(image, mode="linear"):
    vmin = image.image.min()
    vmax = numpy.percentile(image.image, 95)

    a = 1000.
    b = (image.image - vmin) / (vmax - vmin)

    if mode == "linear":
        c = b
    elif mode == "arcsinh":
        c = numpy.arcsinh(10*b)/3.
    elif mode == "log":
        c = numpy.log10(a*b+1)/numpy.log10(a)
    else:
        print("Not a valid mode!")

    return c
