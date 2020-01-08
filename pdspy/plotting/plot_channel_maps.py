from ..constants.physics import c
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy

def plot_channel_maps(visibilities, model, parameters, params, index=0, \
        plot_vis=False, fig=None, image="data", contours="model"):

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

    # Plot the image.

    vmin = numpy.nanmin(visibilities["image"][index].image)
    vmax = numpy.nanmax(visibilities["image"][index].image)

    # Get the centroid position.

    ticks = visibilities["image_ticks"][index]

    for i in range(2):
        # First pass through plot the image, second pass through, plot contours.
        if i == 0:
            data = image
        else:
            data = contours

        # Get the correct dataset for what we are plotting (visibilities vs. 
        # image, data vs. model vs. residuals)

        if plot_vis:
            if data == "data":
                vis = visibilities["data1d"][index]
            elif data == "model":
                vis = model.visibilities[visibilities["lam"][index]]
        else:
            if data == "data":
                plot_image = visibilities["image"][index]

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
            elif data == "model":
                plot_image = model.images[visibilities["lam"][index]]

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
                ind = k*visibilities["ncols"][index] + l + \
                        visibilities["ind0"][index]

                # Turn off the axis if ind >= nchannels

                if ind >= v.size:
                    ax[k,l].set_axis_off()
                    continue

                # On the first loop through, plot the image, and on the second
                # the contours.

                if i == 0:
                    # Plot the image.

                    if plot_vis:
                        ax[k,l].errorbar(vis.uvdist, vis.amp[:,ind], \
                                yerr=1./vis.weights[:,ind]**0.5, fmt="bo")
                    else:
                        ax[k,l].imshow(plot_image.image[ymin:ymax,xmin:xmax,\
                                ind,0], origin="lower", \
                                interpolation="nearest", vmin=vmin, vmax=vmax)

                    # Add the velocity to the map.

                    txt = ax[k,l].annotate(r"$v=%{0:s}$ km s$^{{-1}}$".format(\
                            visibilities["fmt"][index]) % (v[ind]/1e5),\
                            xy=(0.1,0.8), xycoords='axes fraction')

                    # Fix the axes labels.

                    if plot_vis:
                        ax[-1,l].set_xlabel("U-V Distance [k$\lambda$]")
                    else:
                        transform = ticker.FuncFormatter(Transform(xmin, xmax, \
                                visibilities["image_pixelsize"][index],'%.1f"'))

                        ax[k,l].set_xticks(visibilities["image_npix"][index]/2+\
                                ticks[1:-1]/visibilities["image_pixelsize"]\
                                [index]-xmin)
                        ax[k,l].set_yticks(visibilities["image_npix"][index]/2+\
                                ticks[1:-1]/visibilities["image_pixelsize"]\
                                [index]-ymin)

                        ax[k,l].get_xaxis().set_major_formatter(transform)
                        ax[k,l].get_yaxis().set_major_formatter(transform)

                        ax[-1,l].set_xlabel("$\Delta$RA")

                        # Show the size of the beam.

                        bmaj = visibilities["image"][index].header["BMAJ"] / \
                                abs(visibilities["image"][index].\
                                header["CDELT1"])
                        bmin = visibilities["image"][index].header["BMIN"] / \
                                abs(visibilities["image"][index].\
                                header["CDELT1"])
                        bpa = visibilities["image"][index].header["BPA"]

                        ax[k,l].add_artist(patches.Ellipse(xy=(12.5,17.5), \
                                width=bmaj, height=bmin, angle=(bpa+90), \
                                facecolor="white", edgecolor="black"))

                        ax[k,l].set_adjustable('box')

                    if plot_vis:
                        ax[k,0].set_ylabel("Amplitude [Jy]")

                        ax[k,l].set_xscale("log", nonposx='clip')
                    else:
                        ax[k,0].set_ylabel("$\Delta$Dec")

                # Plot the contours.

                elif i == 1:
                    if plot_vis:
                        ax[k,l].plot(vis.uvdist, vis.amp[:,ind], "g-")
                    else:
                        levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, \
                                0.95])*model.images[\
                                visibilities["lam"][index]].image.max()

                        ax[k,l].contour(plot_image.image[ymin:ymax,xmin:xmax,\
                                ind,0], levels=levels)

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
