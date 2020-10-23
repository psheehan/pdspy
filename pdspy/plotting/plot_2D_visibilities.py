import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_2D_visibilities(visibilities, model, parameters, params, index=0, \
        fig=None):

    # Generate a figure and axes if none is provided.

    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9.,4.))
    else:
        fig, ax = fig

    # Calculate the pixel range to show.

    ticks = visibilities["ticks"][index]

    xmin, xmax = int(round(visibilities["npix"][index]/2+ticks[0]/\
            (visibilities["binsize"][index]/1000))), \
            int(round(visibilities["npix"][index]/2+ticks[-1]/\
            (visibilities["binsize"][index]/1000)))
    ymin, ymax = int(round(visibilities["npix"][index]/2+ticks[0]/\
            (visibilities["binsize"][index]/1000))), \
            int(round(visibilities["npix"][index]/2+ticks[-1]/\
            (visibilities["binsize"][index]/1000)))

    # How to scale the real part.

    vmin = min(0, visibilities["data1d"][index].real.min())
    vmax = visibilities["data1d"][index].real.max()

    # Show the real component.

    ax[0].imshow(visibilities["data2d"][index].real.reshape(\
            (visibilities["npix"][index],visibilities["npix"][index]))\
            [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
            interpolation="nearest", vmin=vmin, vmax=vmax, cmap="jet")

    ax[0].contour(model.visibilities[visibilities["lam"][index]+"_2d"].real.\
            reshape((visibilities["npix"][index],visibilities["npix"][index]))\
            [xmin:xmax,xmin:xmax][:,::-1], cmap="jet")

    # How to scale the imaginary part.

    vmin = -visibilities["data1d"][index].real.max()
    vmax =  visibilities["data1d"][index].real.max()

    # Show the imaginary component.

    ax[1].imshow(visibilities["data2d"][index].imag.reshape(\
            (visibilities["npix"][index],visibilities["npix"][index]))\
            [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
            interpolation="nearest", vmin=vmin, vmax=vmax, cmap="jet")

    ax[1].contour(model.visibilities[visibilities["lam"][index]+"_2d"].imag.\
            reshape((visibilities["npix"][index],visibilities["npix"][index]))\
            [xmin:xmax,xmin:xmax][:,::-1], cmap="jet")

    # Adjust the axes ticks.

    transform1 = ticker.FuncFormatter(Transform(xmin, xmax, \
            visibilities["binsize"][index]/1000, '%.0f'))

    ax[0].set_xticks(visibilities["npix"][index]/2+ticks[1:-1]/\
            (visibilities["binsize"][index]/1000)-xmin)
    ax[0].set_yticks(visibilities["npix"][index]/2+ticks[1:-1]/\
            (visibilities["binsize"][index]/1000)-ymin)
    ax[0].get_xaxis().set_major_formatter(transform1)
    ax[0].get_yaxis().set_major_formatter(transform1)

    ax[1].set_xticks(visibilities["npix"][index]/2+ticks[1:-1]/\
            (visibilities["binsize"][index]/1000)-xmin)
    ax[1].set_yticks(visibilities["npix"][index]/2+ticks[1:-1]/\
            (visibilities["binsize"][index]/1000)-ymin)
    ax[1].get_xaxis().set_major_formatter(transform1)
    ax[1].get_yaxis().set_major_formatter(transform1)

    # Adjust the plot and save it.

    ax[0].set_xlabel("U [k$\lambda$]")
    ax[0].set_ylabel("V [k$\lambda$]")

    ax[1].set_xlabel("U [k$\lambda$]")
    ax[1].set_ylabel("V [k$\lambda$]")

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
