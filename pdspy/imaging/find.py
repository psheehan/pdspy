from ..constants.astronomy import arcsec
import pdspy.mcmc as mcmc
import numpy
import scipy.ndimage.filters
import scipy.ndimage.morphology
import scipy.optimize
import matplotlib.pyplot as plt
import os
import astropy
import astropy.coordinates

def find(image, threshold=5, include_radius=20, window_size=40, \
        output_plots=None):

    # If plots of the fits have been requested, make the directory if it 
    # doesn't already exist.

    if output_plots != None:
        os.system("rm -r "+output_plots)
        os.mkdir(output_plots)

    # Find potential sources in the image.

    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    local_max = scipy.ndimage.filters.maximum_filter(image.image[:,:,0,0], \
            footprint=neighborhood) == image.image[:,:,0,0]

    background = (image.image[:,:,0,0] == 0)

    eroded_background = scipy.ndimage.morphology.binary_erosion(background, \
            structure=neighborhood, border_value=1)

    detected_peaks = local_max - eroded_background

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # For each of the potential sources see if the detection is significant
    # at a provided level, and if so, fit it with a Gaussian function.

    sources = []

    for coords in potential_sources:
        # Check whether the potential source meets the detection threshold.

        if image.image[coords[0], coords[1], 0, 0] < threshold * \
                image.unc[coords[0], coords[1], 0, 0]:
            detected_peaks[coords[0], coords[1]] = 0.
            continue

        # Set up the parameter guesses for fitting.

        half_window = window_size / 2

        xmin = max(0,coords[1]-half_window)
        xmax = min(coords[1]+half_window,image.image.shape[1])
        ymin = max(0,coords[0]-half_window)
        ymax = min(coords[0]+half_window,image.image.shape[1])

        x, y = numpy.meshgrid(numpy.linspace(xmin, xmax-1, xmax - xmin), \
                numpy.linspace(ymin, ymax-1, ymax - ymin))

        z = image.image[ymin:ymax,xmin:xmax,0,0]
        sigma_z = image.unc[ymin:ymax,xmin:xmax,0,0]

        xc, yc = coords[1], coords[0]
        params = numpy.array([xc, yc, 1.0, 1.0, 0.0, image.image[yc,xc,0,0]])
        sigma_params = numpy.array([3.0, 3.0, 0.2, 0.2, numpy.pi/10, 1.0])

        # Find any sources within a provided radius and include them in the 
        # fit.

        nsources = 1

        for coords2 in potential_sources:
            if image.image[coords2[0], coords2[1], 0, 0] < threshold * \
                    image.unc[coords2[0], coords2[1], 0, 0]:
                detected_peaks[coords2[0], coords2[1]] = 0.
                continue

            d = numpy.sqrt( (coords[0] - coords2[0])**2 + \
                    (coords[1] - coords2[1])**2 )

            if (d < include_radius) and (d > 0):
                xc, yc = coords2[1], coords2[0]
                params = numpy.hstack([params, numpy.array([xc, yc, 1.0, 1.0, \
                        0.0, image.image[yc,xc,0,0]])])
                sigma_params = numpy.hstack([sigma_params, numpy.array([3.0, \
                        3.0, 0.2, 0.2, numpy.pi/10, 1.0])])

                nsources += 1

        # Try a least squares fit.

        func = lambda p, n, x, y, z, sigma: \
                ((z - gaussian2d(x, y, p, n)) / sigma).reshape((z.size,))

        p, cov, infodict, mesg, ier = scipy.optimize.leastsq(func, params, \
                args=(nsources, x, y, z, sigma_z), full_output=True)

        # Fix the least squares result for phi because it seems to like to go
        # crazy.

        p[4] = numpy.fmod(numpy.fmod(p[4], numpy.pi)+numpy.pi, numpy.pi)

        # Calculate the uncertainties on the parameters.

        if (type(cov) == type(None)):
            sigma_p = p
        else:
            sigma_p = numpy.sqrt(numpy.diag((func(p, nsources, x, y, z, \
                    sigma_z)**2).sum()/(y.size - p.size) * cov))

        # Now do a few iterations of MCMC to really get the parameters.

        """
        limits = [{"limited":[False,False], "limits":[0.0,0.0]} \
                for i in range(6*nsources)]

        for i in range(nsources):
            limits[i*6+4] = {"limited":[True,True], "limits":[p[i*6+4]- \
                    numpy.pi/5, p[i*6+4]+numpy.pi/5]}

        accepted_params = mcmc.mcmc2d(x, y, z, sigma_z, p, sigma_p, \
                gaussian2d, args={'n':nsources}, nsteps=nsources*1e5, \
                limits=limits)

        p = accepted_params.mean(axis=0)
        sigma_p = accepted_params.std(axis=0)
        """

        # Add the newly found source to the list of sources.

        new_source = numpy.empty((12,), dtype=p.dtype)
        new_source[0::2] = p[0:6]
        new_source[1::2] = sigma_p[0:6]

        #sources.append(tuple(new_source))
        sources.append(new_source)

        # Plot the histograms of the MCMC fit to make sure they look good.

        """
        fig, ax = plt.subplots(nrows=2, ncols=3)

        for i in range(2):
            for j in range(3):
                ax[i,j].hist(accepted_params[:,3*i+j], bins=20)

        plt.show()
        """

        # Plot the image slice.

        if output_plots != None:
            fig, ax = plt.subplots(nrows=2, ncols=2)

            ax[0,0].set_title("Data")
            ax[0,0].imshow(z, origin="lower", interpolation="nearest", \
                    vmin=z.min(), vmax=z.max())
            ax[0,1].set_title("Uncertainty")
            ax[0,1].imshow(sigma_z, origin="lower", interpolation="nearest", \
                    vmin=z.min(), vmax=z.max())
            ax[1,0].set_title("Model")
            ax[1,0].imshow(gaussian2d(x, y, p, nsources), origin="lower", \
                    interpolation="nearest", vmin=z.min(), vmax=z.max())
            ax[1,1].set_title("Data-Model")
            ax[1,1].imshow(z - gaussian2d(x, y, p, nsources), origin="lower", \
                    interpolation="nearest", vmin=z.min(), vmax=z.max())

            fig.savefig(output_plots+"/source_{0:d}.pdf".format(len(sources)))

            plt.close(fig)

    if len(sources) > 0:
        sources = astropy.table.Table(numpy.array(sources), names=("x", \
                "x_unc","y","y_unc","sigma_x","sigma_x_unc","sigma_y", \
                "sigma_y_unc","pa", "pa_unc","f",'f_unc'))

    if hasattr(image, "wcs"):
        ra, dec = image.wcs.wcs_pix2world(sources["x"], sources["y"], 1)

        temp = astropy.coordinates.SkyCoord(ra, dec, unit='deg')

        sources['ra'] = temp.ra.to_string(unit=astropy.units.hour)
        sources['dec'] = temp.dec.to_string()

        sources['ra_unc'] = abs(image.wcs.wcs.cdelt[0]) * sources['x_unc'] / \
                (180. / numpy.pi) / arcsec
        sources['dec_unc'] = abs(image.wcs.wcs.cdelt[1]) * sources['y_unc'] / \
                (180. / numpy.pi) / arcsec

        sources['FWHM_x'] = 2.35482 * abs(image.wcs.wcs.cdelt[0]) * \
                sources['sigma_x'] / (180. / numpy.pi) / arcsec
        sources['FWHM_y'] = 2.35482 * abs(image.wcs.wcs.cdelt[1]) * \
                sources['sigma_y'] / (180. / numpy.pi) / arcsec
        sources['FWHM_x_unc'] = 2.35482 * abs(image.wcs.wcs.cdelt[0]) * \
                sources['sigma_x_unc'] / (180. / numpy.pi) / arcsec
        sources['FWHM_y_unc'] = 2.35482 * abs(image.wcs.wcs.cdelt[1]) * \
                sources['sigma_y_unc'] / (180. / numpy.pi) / arcsec

        sources['flux'] = sources['f'] * sources['sigma_x'] * \
                sources['sigma_y'] * 2 * numpy.pi
        sources['flux_unc'] = 2*numpy.pi * numpy.sqrt( \
                (sources['f_unc']*sources['sigma_x']*sources['sigma_y'])**2+\
                (sources['f']*sources['sigma_x_unc']*sources['sigma_y'])**2+\
                (sources['f']*sources['sigma_x'] * sources['sigma_y_unc'])**2)

        if "BMAJ" in image.header:
            beam_per_pixel = abs(image.wcs.wcs.cdelt.prod()) / \
                    (numpy.pi*image.header['BMAJ']*image.header['BMIN']/ \
                    (4*numpy.log(2)))

            sources['flux'] *= beam_per_pixel
            sources['flux_unc'] *= beam_per_pixel

    return sources

def gaussian2d(x, y, params, n=1):

    model = numpy.zeros(x.shape)

    for i in range(n):
        x0, y0, sigmax, sigmay, pa, f0 = tuple(params[i*6:i*6+6])

        xp=(x-x0)*numpy.cos(pa)-(y-y0)*numpy.sin(pa)
        yp=(x-x0)*numpy.sin(pa)+(y-y0)*numpy.cos(pa)

        model += f0*numpy.exp(-1*xp**2/(2*sigmax**2))* \
                numpy.exp(-1*yp**2/(2*sigmay**2))
    
    return model
