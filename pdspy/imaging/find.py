import pdspy.mcmc as mcmc
import numpy
import scipy.ndimage.filters
import scipy.ndimage.morphology
import scipy.optimize
import matplotlib.pyplot as plt

def find(image, threshold=5):

    # Find potential sources in the image.

    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    local_max = scipy.ndimage.filters.maximum_filter(image.image, \
            footprint=neighborhood) == image.image

    background = (image.image == 0)

    eroded_background = scipy.ndimage.morphology.binary_erosion(background, \
            structure=neighborhood, border_value=1)

    detected_peaks = local_max - eroded_background

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # For each of the potential sources see if the detection is significant
    # at a provided level, and if so, fit it with a Gaussian function.

    sources = []

    for coords in potential_sources:
        # Check whether the potential source meets the detection threshold.

        if image.image[coords[0], coords[1]] < threshold * \
                image.unc[coords[0], coords[1]]:
            detected_peaks[coords[0], coords[1]] = 0.
            continue

        # Set up the parameter guesses for fitting.

        xmin, xmax = max(0,coords[1]-20), min(coords[1]+20,image.image.shape[1])
        ymin, ymax = max(0,coords[0]-20), min(coords[0]+20,image.image.shape[1])

        x, y = numpy.meshgrid(numpy.linspace(xmin, xmax-1, xmax - xmin), \
                numpy.linspace(ymin, ymax-1, ymax - ymin))

        z = image.image[ymin:ymax,xmin:xmax]
        sigma_z = image.unc[ymin:ymax,xmin:xmax]

        xc, yc = coords[1], coords[0]
        params = numpy.array([xc, yc, 1.0, 1.0, 0.0, image.image[yc,xc]])
        sigma_params = numpy.array([3.0, 3.0, 0.2, 0.2, numpy.pi/10, 1.0])

        # Try a least squares fit.

        func = lambda p, x, y, z, sigma: \
                ((z - gaussian2d(x, y, p)) / sigma).reshape((z.size,))

        p, cov, infodict, mesg, ier = scipy.optimize.leastsq(func, params, \
                args=(x, y, z, sigma_z), full_output=True)

        # Fix the least squares result for phi because it seems to like to go
        # crazy.

        p[4] = numpy.fmod(numpy.fmod(p[4], numpy.pi)+numpy.pi, numpy.pi)

        # Calculate the uncertainties on the parameters.

        if (type(cov) == type(None)):
            sigma_p = p
        else:
            sigma_p = numpy.sqrt(numpy.diag((func(p, x, y, z, \
                    sigma_z)**2).sum()/(y.size - p.size) * cov))

        # Now do a few iterations of MCMC to really get the parameters.

        limits = [{"limited":[False,False], "limits":[0.0,0.0]} \
                for i in range(6)]

        limits[4] = {"limited":[True,True], "limits":[p[4]-numpy.pi/4, \
                p[4]+numpy.pi/4]}

        accepted_params = mcmc.mcmc2d(x, y, z, sigma_z, p, sigma_p, \
                gaussian2d, args={}, nsteps=1e5, limits=limits)

        p = accepted_params.mean(axis=0)
        sigma_p = accepted_params.std(axis=0)

        # Add the newly found source to the list of sources.

        new_source = numpy.empty((2*p.size,), dtype=p.dtype)
        new_source[0::2] = p
        new_source[1::2] = sigma_p

        sources.append(tuple(new_source))

        # Plot the histograms of the MCMC fit to make sure they look good.

        fig, ax = plt.subplots(nrows=2, ncols=3)

        for i in range(2):
            for j in range(3):
                ax[i,j].hist(accepted_params[:,3*i+j], bins=20)

        plt.show()

        # Plot the image slice.

        fig, ax = plt.subplots(nrows=2, ncols=2)

        ax[0,0].imshow(z, origin="lower", interpolation="nearest", \
                vmin=z.min(), vmax=z.max())
        ax[0,1].imshow(sigma_z, origin="lower", interpolation="nearest", \
                vmin=z.min(), vmax=z.max())
        ax[1,0].imshow(gaussian2d(x, y, p), origin="lower", \
                interpolation="nearest", vmin=z.min(), vmax=z.max())
        ax[1,1].imshow(z - gaussian2d(x, y, p), origin="lower", \
                interpolation="nearest", vmin=z.min(), vmax=z.max())

        plt.show()

    sources = numpy.core.records.fromrecords(sources, names="x,x_unc,y,y_unc,"+\
            'sigma_x,sigma_x_unc,sigma_y,sigma_y_unc,pa,pa_unc,f,f_unc')

    return sources

def gaussian2d(x, y, params):

    x0, y0, sigmax, sigmay, pa, f0 = tuple(params)

    xp=(x-x0)*numpy.cos(pa)-(y-y0)*numpy.sin(pa)
    yp=(x-x0)*numpy.sin(pa)+(y-y0)*numpy.cos(pa)

    return f0*numpy.exp(-1*xp**2/(2*sigmax**2))*numpy.exp(-1*yp**2/(2*sigmay**2))
