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
        source_list=None, list_search_radius=1.0, beam=[1.0,1.0,0.0], \
        output_plots=None):

    # If plots of the fits have been requested, make the directory if it 
    # doesn't already exist.

    if output_plots != None:
        os.system("rm -r "+output_plots)
        os.mkdir(output_plots)

    # Create a base image to look for peaks in.

    if type(source_list) != type(None):
        coords = astropy.coordinates.SkyCoord(source_list["ra"].tolist(), \
                source_list["dec"].tolist(), 'icrs')

        pixcoords = image.wcs.wcs_world2pix(coords.ra.degree, \
                coords.dec.degree, 1, ra_dec_order=True)

        arcsec_in_pixels = arcsec / (abs(image.wcs.wcs.cdelt[0]) * numpy.pi/180)

        x, y = numpy.meshgrid(numpy.arange(image.image.shape[1]), \
                numpy.arange(image.image.shape[0]))

        mask = numpy.zeros((image.image.shape[0], image.image.shape[1]), \
                dtype=bool)

        for pixcoord in zip(pixcoords[0], pixcoords[1]):
            mask[numpy.sqrt( (pixcoord[0] - x)**2 + \
                    (pixcoord[1] -y)**2 ) < 0.5*list_search_radius*\
                    arcsec_in_pixels] = True

        base_image = numpy.where(mask, image.image[:,:,0,0], numpy.nan)
    else:
        base_image = image.image[:,:,0,0]

    # Find potential sources in the image.

    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    local_max = scipy.ndimage.filters.maximum_filter(base_image, \
            footprint=neighborhood) == base_image

    background = (base_image == 0)

    eroded_background = scipy.ndimage.morphology.binary_erosion(background, \
            structure=neighborhood, border_value=1)

    detected_peaks = local_max - eroded_background

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # First, throw away any potential source that does not meet the
    # threshold cut requirement.

    for coords in potential_sources:
        if image.image[coords[0], coords[1], 0, 0] < threshold * \
                image.unc[coords[0], coords[1], 0, 0]:
            detected_peaks[coords[0], coords[1]] = 0.

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # Search for potential sources that are probably the same source and
    # make sure we're only finding it once.

    good = numpy.repeat(True, len(potential_sources))
    for i in range(len(potential_sources)):
        for j in range(len(potential_sources)):
            if (j != i) and good[i] and good[j]:
                coords = potential_sources[i]
                coords2 = potential_sources[j]

                d = numpy.sqrt( (coords[0] - coords2[0])**2 + \
                        (coords[1] - coords2[1])**2 )

                if (d < include_radius) and (d > 0):
                    inbetween = bresenham_line(coords[0],coords[1],coords2[0], \
                            coords2[1])

                    for k, coords3 in enumerate(inbetween):
                        if image.image[coords3[0],coords3[1], 0, 0] < 2.0 * \
                                image.unc[coords3[0], coords3[1], 0 ,0]:
                            break

                        if k == len(inbetween)-1:
                            if image.image[coords[0], coords[1]] > \
                                    image.image[coords2[0], coords2[1]]:
                                detected_peaks[coords2[0], coords2[1]] = 0.
                                good[j] = False
                            else:
                                detected_peaks[coords[0], coords[1]] = 0.
                                good[i] = False

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # Now we have a good list of detected sources. Fit all of them with a
    # Gaussian to measure positions and fluxes.

    sources = []

    for coords in potential_sources:
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
        params = numpy.array([xc, yc, beam[0], beam[1], beam[2], \
                image.image[yc,xc,0,0]])
        sigma_params = numpy.array([1.0, 1.0, 0.2*beam[0], 0.2*beam[1], \
                numpy.pi/10, 1.0])

        # Find any sources within a provided radius and include them in the 
        # fit.

        nsources = 1

        for coords2 in potential_sources:
            d = numpy.sqrt( (coords[0] - coords2[0])**2 + \
                    (coords[1] - coords2[1])**2 )

            if (d < include_radius) and (d > 0):
                xc, yc = coords2[1], coords2[0]
                params = numpy.hstack([params, numpy.array([xc, yc, beam[0], \
                        beam[1], beam[2], image.image[yc,xc,0,0]])])
                sigma_params = numpy.hstack([sigma_params, numpy.array([1.0, \
                        1.0, 0.2*beam[0], 0.2*beam[1], numpy.pi/10, 1.0])])

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

        # Add the newly found source to the list of sources.

        new_source = numpy.empty((14,), dtype=p.dtype)
        new_source[0:12][0::2] = p[0:6]
        new_source[0:12][1::2] = sigma_p[0:6]

        new_source[12] = z[numpy.logical_and(z / sigma_z > 2.0, \
                numpy.sqrt((coords[1]-x)**2 + (coords[0]-y)**2) < 10.0)].sum()
        new_source[13] = numpy.sqrt(sigma_z[numpy.logical_and(z/sigma_z > 2.0, \
                numpy.sqrt((coords[1]-x)**2 + (coords[0]-y)**2) < 10.0)]**2).\
                sum()

        sources.append(new_source)

        # Plot the image slice.

        #z[numpy.logical_or(z / sigma_z < 2.0, \
        #        numpy.sqrt((coords[1] - x)**2 + (coords[0] - y)**2) > 10.0)] = 0
        #z [numpy.sqrt((coords[1] - x)**2 + (coords[0] - y)**2) > 10.0] = 0.0

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

            fig.savefig(output_plots+"/source_{0:d}.pdf".format(len(sources)-1))

            plt.close(fig)

    if len(sources) > 0:
        sources = astropy.table.Table(numpy.array(sources), names=("x", \
                "x_unc","y","y_unc","sigma_x","sigma_x_unc","sigma_y", \
                "sigma_y_unc","pa", "pa_unc","f",'f_unc',"Flux","Flux_unc"))

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
            sources['Flux'] *= beam_per_pixel
            sources['Flux_unc'] *= beam_per_pixel

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

def bresenham_line(x,y,x2,y2):
    """Brensenham line algorithm"""
    steep = 0
    coords = []
    dx = abs(x2 - x)
    if (x2 - x) > 0: sx = 1
    else: sx = -1
    dy = abs(y2 - y)
    if (y2 - y) > 0: sy = 1
    else: sy = -1
    if dy > dx:
        steep = 1
        x,y = y,x
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx
    for i in range(0,dx):
        if steep: coords.append((y,x))
        else: coords.append((x,y))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    coords.append((x2,y2))
    return coords
