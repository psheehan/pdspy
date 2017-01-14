from ..constants.astronomy import arcsec
from ..table import Table
import numpy
import scipy.ndimage.filters
import scipy.ndimage.morphology
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import os
import astropy
import astropy.coordinates

def find(image, threshold=5, include_radius=20, window_size=40, \
        source_list=None, list_search_radius=1.0, list_threshold=5, \
        beam=[1.0,1.0,0.0], user_aperture=False, aperture=15, \
        fit_aperture=15, include_flux_unc=False, flux_unc=0.1, \
        bootstrap_unc=True, output_plots=None, just_find=False):

    # If plots of the fits have been requested, make the directory if it 
    # doesn't already exist.

    if output_plots != None:
        os.system("rm -r "+output_plots)
        os.mkdir(output_plots)

    # Find potential sources in the image.

    base_image = image.image[:,:,0,0]

    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    local_max = scipy.ndimage.filters.maximum_filter(base_image, \
            footprint=neighborhood) == base_image

    background = (base_image == 0)

    eroded_background = scipy.ndimage.morphology.binary_erosion(background, \
            structure=neighborhood, border_value=1)

    detected_peaks = local_max - eroded_background

    potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))
    
    # If we are providing a source list, the threshold should be lower than
    # for a blind search. Throw away for good any sources that don't meet
    # this threshold.

    if type(source_list) != type(None):
        for coords in potential_sources:
            if image.image[coords[0], coords[1], 0, 0] < list_threshold * \
                    image.unc[coords[0], coords[1], 0, 0]:
                detected_peaks[coords[0], coords[1]] = 0.

        potential_sources = numpy.column_stack(numpy.nonzero(detected_peaks))

    # First, throw away any potential source that does not meet the
    # threshold cut requirement.

    for coords in potential_sources:
        if image.image[coords[0], coords[1], 0, 0] < threshold * \
                image.unc[coords[0], coords[1], 0, 0]:
            detected_peaks[coords[0], coords[1]] = 0.

    # Next, if a source list was provided for searching purposes, throw away
    # any objects not within the requisite radius from a listed source.

    if type(source_list) != type(None):
        coords = astropy.coordinates.SkyCoord(source_list["ra"].tolist(), \
                source_list["dec"].tolist(), frame='icrs')

        pixcoords = image.wcs.wcs_world2pix(coords.ra.degree, \
                coords.dec.degree, 1, ra_dec_order=True)

        arcsec_in_pixels = arcsec / (abs(image.wcs.wcs.cdelt[0]) * numpy.pi/180)

        for peak_coords in potential_sources:
            for count, pixcoord in enumerate(zip(pixcoords[0], pixcoords[1])):
                if (numpy.sqrt((pixcoord[0] - peak_coords[1])**2 + \
                               (pixcoord[1] - peak_coords[0])**2) <\
                               0.5 * list_search_radius * arcsec_in_pixels):
                    #detected_peaks[peak_coords[0], peak_coords[1]] = 1.
                    if image.image[peak_coords[0], peak_coords[1], 0, 0] > \
                            list_threshold * image.unc[peak_coords[0], \
                            peak_coords[1], 0, 0]:
                        detected_peaks[peak_coords[0], peak_coords[1]] = 1.
                        break
                #elif (count == len(pixcoords[0])-1):
                #    detected_peaks[peak_coords[0], peak_coords[1]] = 0.

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

    if just_find:
        return potential_sources

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

        z = image.image[int(ymin):int(ymax),int(xmin):int(xmax),0,0]
        sigma_z = image.unc[int(ymin):int(ymax),int(xmin):int(xmax),0,0]

        beam_to_sigma = arcsec / (abs(image.wcs.wcs.cdelt[0]) * numpy.pi/180) /\
                2.355

        xc, yc = coords[1], coords[0]
        params = numpy.array([xc, yc, beam[0]*beam_to_sigma, \
                beam[1]*beam_to_sigma, beam[2], image.image[yc,xc,0,0]])

        bm = [image.header["BMAJ"]/abs(image.wcs.wcs.cdelt[0])/2.355,\
                image.header["BMIN"]/abs(image.wcs.wcs.cdelt[0])/2.355,\
                image.header["BPA"]*numpy.pi/180.]

        # Fit the source with a Gaussian.

        try:
            p, sigma_p = fit_source(coords, x, y, z, sigma_z, params, \
                    bootstrap_unc=bootstrap_unc, beam=bm)
        except ValueError:
            continue

        # Create a new source to add.

        new_source = numpy.empty((16,), dtype=p.dtype)
        new_source[0:12][0::2] = p[0:6]
        new_source[0:12][1::2] = sigma_p[0:6]

        # Before doing aperture photometry, find any sources within a provided 
        # radius and fit them so they can be subtracted out of the sky
        # subtraction window.

        nsources = 1

        for coords2 in potential_sources:
            d = numpy.sqrt( (coords[0] - coords2[0])**2 + \
                    (coords[1] - coords2[1])**2 )

            if (d < include_radius) and (d > 0):
                xc, yc = coords2[1], coords2[0]
                params = numpy.array([xc, yc, beam[0]*beam_to_sigma, \
                        beam[1]*beam_to_sigma, beam[2], image.image[yc,xc,0,0]])

                try:
                    new_p, sigma_new_p = fit_source(coords2, x, y, z, sigma_z, \
                            params, bootstrap_unc=False)
                except ValueError:
                    continue

                p = numpy.hstack([p, new_p])
                nsources += 1

        # Do some aperture photometry for the source.

        if nsources > 1:
            new_z = z.copy() - gaussian2d(x, y, p[6:], nsources-1)
        else:
            new_z = z.copy()

        if not user_aperture:
            aperture = 3 * numpy.sqrt(new_source[4]*new_source[6])

        try:
            sky = numpy.median(new_z[numpy.logical_and(\
                    numpy.sqrt((coords[1]-x)**2 + (coords[0]-y)**2) > aperture,\
                    numpy.sqrt((coords[1]-x)**2 + (coords[0]-y)**2) <= \
                    4*aperture)])
        except IndexError:
            sky = -1.0e-5
            print("Error in source:", len(sources))

        new_source[12] = image.image[coords[0], coords[1], 0, 0] - sky
        new_source[13] = image.unc[coords[0], coords[1], 0, 0]
        new_source[14] = (new_z[numpy.sqrt((new_source[0]-x)**2 + \
                (new_source[2]-y)**2) < aperture] - sky).sum()
        new_source[15] = numpy.sqrt((sigma_z[numpy.sqrt((new_source[0]-x)**2+ \
                (new_source[2]-y)**2) < aperture]**2).sum())

        # Add the newly found source to the list of sources.

        sources.append(new_source)

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
            ax[1,1].imshow(z - gaussian2d(x, y, p, nsources), \
                    origin="lower", interpolation="nearest", vmin=z.min(), \
                    vmax=z.max())

            circle1 = plt.Circle((new_source[0] - coords[1] + half_window, \
                    new_source[2] - coords[0] + half_window), aperture, \
                    edgecolor='r', facecolor="none")
            ax[0,0].add_artist(circle1)
            circle2 = plt.Circle((new_source[0] - coords[1] + half_window, \
                    new_source[2] - coords[0] + half_window), \
                    4*aperture, edgecolor='r', facecolor="none")
            ax[0,0].add_artist(circle2)

            fig.savefig(output_plots+"/source_{0:d}.pdf".format(len(sources)-1))

            plt.close(fig)

    if len(sources) > 0:
        sources = Table(numpy.array(sources), names=("x", \
                "x_unc","y","y_unc","sigma_x","sigma_x_unc","sigma_y", \
                "sigma_y_unc","pa", "pa_unc","f",'f_unc',"Peak_Flux", \
                "Peak_Flux_unc","Flux","Flux_unc"))

    if hasattr(image, "wcs"):
        try:
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

            if include_flux_unc:
                sources['flux_unc'] = numpy.sqrt(sources['flux_unc']**2 + \
                        (flux_unc * sources['flux'])**2)
                sources['Flux_unc'] = numpy.sqrt(sources['Flux_unc']**2 + \
                        (flux_unc * sources['Flux'])**2)
        except:
            print(sources["x"], sources["y"], temp)

    return sources

def fit_source(coords, x, y, z, sigma_z, params, bootstrap_unc=True, \
        beam=None):
    # Try a least squares fit.

    func = lambda p, n, x, y, z, sigma: \
            ((z - gaussian2d(x, y, p, n)) / sigma).reshape((z.size,))

    # Try to automatically determine the aperture.

    z3 = z.copy()
    z3[z/sigma_z < 2.] = 0.0

    xm, ym = x[0, int(x.shape[1]/2)], y[int(y.shape[0]/2), 0]

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z3[i,j] != 0.:
                inbetween = bresenham_line(int(coords[0]-ym+z.shape[0]/2), \
                        int(coords[1] - xm + z.shape[1]/2),i,j)

                for k, coords3 in enumerate(inbetween):
                    if z3[coords3[0],coords3[1]] < 2.0 * \
                            sigma_z[coords3[0], coords3[1]]:
                        z3[i,j] = 0.0

    params, cov, infodict, mesg, ier = scipy.optimize.leastsq(func, \
            params, args=(1, x, y, z3, sigma_z), full_output=True)

    fit_aperture = 3 * numpy.sqrt(abs(params[2]*params[3]))

    # Only fit to pixels in a certain aperture around the source. This is 
    # because otherwise the fitting routine has a habit of finding other 
    # peaks or negative dips when fitting faint sources.

    for i in range(1):
        z2 = numpy.zeros(z.shape)
        sigma_z2 = numpy.zeros(z.shape) + 1.0e20

        if i > 0:
            params[0:6] = p[0:6]
            ap = 3 * numpy.sqrt(abs(params[2]*params[3]))
        else:
            ap = fit_aperture

        z2[numpy.sqrt((x - params[0])**2 + (y - params[1])**2) < ap] = \
                z[numpy.sqrt((x - params[0])**2 + (y - params[1])**2) < ap]
        sigma_z2[numpy.sqrt((x - params[0])**2 + (y - params[1])**2) < ap]=\
                sigma_z[numpy.sqrt((x - params[0])**2 + (y - params[1])**2)\
                < ap]

        p, cov, infodict, mesg, ier = scipy.optimize.leastsq(func, \
                params, args=(1, x, y, z2, sigma_z2), full_output=True)

    if (type(cov) == type(None)):
        raise ValueError('The fit failed')

    # Fix the least squares result for phi because it seems to like to go
    # crazy.

    p[4] = numpy.fmod(numpy.fmod(p[4], numpy.pi)+numpy.pi, numpy.pi)

    # Take the absolute values of sigma_x and sigma_y in case they 
    # managed to go negative.

    p[2:4] = abs(p[2:4])

    # Now calculate the uncertainty on the parameters using bootstrapping.

    if bootstrap_unc:
        s_res = sigma_z.mean()
        ps = []

        for i in range(100):
            randomDelta = interferometer_noise(z.shape, s_res, beam)
            randomdataZ = z + randomDelta
            randomdataSigma_Z = sigma_z.copy()

            randomdataZ[numpy.sqrt((x-params[0])**2 + (y-params[1])**2) >= \
                    ap] = 0.
            randomdataSigma_Z[numpy.sqrt((x - params[0])**2 + \
                    (y - params[1])**2) >= ap] = 1.0e20

            randomfit, randomcov = scipy.optimize.leastsq(func, params, \
                    args=(1, x, y, randomdataZ, randomdataSigma_Z), \
                    full_output=False)

            ps.append(randomfit)

        ps = numpy.array(ps)

        pfit_bootstrap = numpy.mean(ps,0)
        perr_bootstrap = numpy.std(ps,0)

        pfit_bootstrap[4] = numpy.fmod(numpy.fmod(pfit_bootstrap[4], \
                numpy.pi)+numpy.pi, numpy.pi)

        return p, perr_bootstrap

    # Calculate the uncertainties on the parameters.

    else:
        sigma_p = numpy.sqrt(numpy.diag((func(p, 1, x, y, z2, \
                sigma_z2)**2).sum()/(y.size - p.size) * cov))

        return p, sigma_p

def interferometer_noise(shape, sigma, beam):

    test = numpy.random.normal(0.0, sigma, shape[0]*shape[1]).reshape(shape)

    x, y = numpy.meshgrid(numpy.arange(shape[1]),numpy.arange(shape[0]))

    gaus = gaussian2d(x, y, [shape[1]/2, shape[0]/2, beam[0], beam[1], \
            beam[2], 1])

    new_test = scipy.signal.fftconvolve(test, gaus, mode='same')
    new_test *= sigma / new_test.std()

    return new_test

def gaussian2d(x, y, params, n=1):

    model = numpy.zeros(x.shape)

    for i in range(n):
        x0, y0, sigmax, sigmay, pa, f0 = tuple(params[i*6:i*6+6])

        xp=(x-x0)*numpy.sin(pa)-(y-y0)*numpy.cos(pa)
        yp=(x-x0)*numpy.cos(pa)+(y-y0)*numpy.sin(pa)

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
