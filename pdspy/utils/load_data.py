from .. import interferometry as uv
from .. import spectroscopy as sp
from .. import imaging as im
import numpy

def load_data(config, model="disk", gridsize1D=20):
    r"""
    Loads the data, as specified by the configuration module.

    Args:
        :attr:`config` (`module`):
            The configuration information for the model, as read in by the :code:utils.load_config function.
        :attr:`model` (`str`, optional):
            Are you running disk_model_* or flared_model_*. Default: `"disk"`
        :attr:`model` (`int` or `list-like`, optional):
            The number of bins to use when averaging the visibility data into 1D. Default: `40`

    Returns:
        :attr:`visibilities` (`dict`):
            A dictionary containing the visibility data and the information about it from the configuration module.
        :attr:`images` (`dict`):
            A dictionary containing the image data and the information about it from the configuration module.
        :attr:`spectra` (`dict`):
            A dictionary containing the spectra and the information about it from the configuration module.
    """


    # Set up the places where we will put all of the data.

    config.visibilities["data"] = []
    config.visibilities["data1d"] = []
    config.visibilities["data2d"] = []
    config.visibilities["image"] = []
    config.spectra["data"] = []
    config.spectra["binned"] = []
    config.images["data"] = []

    # Make sure gridsize1D has the proper number of elements.

    if gridsize1D == 20:
        gridsize1D = [20 for i in range(len(config.visibilities["file"]))]

    # Read in the millimeter visibilities.

    for j in range(len(config.visibilities["file"])):
        # Read the raw data.

        if config.visibilities["file"][j].split(".")[-1] == "ms":
            data = uv.readms(config.visibilities["file"][j], \
                    spw=config.visibilities["spw"][j], \
                    tolerance=config.visibilities["tolerance"][j], \
                    datacolumn=config.visibilities["datacolumn"][j])
        else:
            data = uv.Visibilities()
            data.read(config.visibilities["file"][j])

        # Center the data. => need to update!

        data = uv.center(data, [config.visibilities["x0"][j], \
                config.visibilities["y0"][j], 1.])

        # Set any weights < 0 to be equal to zero to prevent odd fits.

        data.weights[data.weights < 0] = 0

        # Take the complex conjugate to make sure orientation is correct.

        #data.imag *= -1

        # Add the data to the dictionary structure.

        config.visibilities["data"].append(data)

        # Scale the weights of the visibilities to force them to be fit well.

        config.visibilities["data"][j].weights *= \
                config.visibilities["weight"][j]

        # Average the visibilities radially.

        if model == "disk":
            config.visibilities["data1d"].append(uv.average(data, \
                    gridsize=gridsize1D[j], radial=True, log=True, \
                    logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*0.95, \
                    logmax=data.uvdist.max()*1.05))
        elif model == "flared":
            config.visibilities["data1d"].append(uv.average(data, \
                    gridsize=gridsize1D[j], radial=True, log=True, \
                    logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*0.95, \
                    logmax=data.uvdist.max()*1.05, mode="spectralline"))

        # Also average the data into a 2D grid, if disk model..

        if model == "disk":
            config.visibilities["data2d"].append(uv.grid(data, \
                    gridsize=config.visibilities["gridsize"][j], \
                    binsize=config.visibilities["binsize"][j]))

        # Read in the image.

        config.visibilities["image"].append(im.readimfits(\
                config.visibilities["image_file"][j]))

    ######################
    # Read in the spectra.
    ######################

    for j in range(len(config.spectra["file"])):
        config.spectra["data"].append(sp.Spectrum())
        config.spectra["data"][j].read(config.spectra["file"][j])

        # Adjust the weight of the SED, as necessary.

        config.spectra["data"][j].unc /= config.spectra["weight"][j]**0.5

        # Merge the SED with the binned Spitzer spectrum.

        if config.spectra["bin?"][j]:
            wave = numpy.linspace(config.spectra["data"][j].wave.min(), \
                    config.spectra["data"][j].wave.max(), \
                    config.spectra["nbins"][j])
            flux = numpy.interp(wave, config.spectra["data"][j].wave, \
                    config.spectra["data"][j].flux)

            good = flux > 0.

            wave = wave[good]
            flux = flux[good]

            config.spectra["binned"].append(sp.Spectrum(wave, flux))
        else:
            config.spectra["binned"].append(config.spectra["data"][j])

        # Merge all the spectra together in one big SED.

        const_unc = 0.1 / config.spectra["weight"][j]**0.5

        try:
            config.spectra["total"].wave = numpy.concatenate((\
                    config.spectra["total"].wave, \
                    config.spectra["binned"][j].wave))
            config.spectra["total"].flux = numpy.concatenate((\
                    config.spectra["total"].flux, \
                    numpy.log10(config.spectra["binned"][j].flux)))
            config.spectra["total"].unc = numpy.concatenate((\
                    config.spectra["total"].unc, \
                    numpy.repeat(const_unc, \
                    config.spectra["binned"][j].wave.size)))

            order = numpy.argsort(config.spectra["total"].wave)

            config.spectra["total"].wave = config.spectra["total"].wave[order]
            config.spectra["total"].flux = config.spectra["total"].flux[order]
            config.spectra["total"].unc = config.spectra["total"].unc[order]
        except:
            config.spectra["total"] = sp.Spectrum(\
                    config.spectra["binned"][j].wave, \
                    numpy.log10(config.spectra["binned"][j].flux), \
                    numpy.repeat(const_unc, \
                    config.spectra["binned"][j].wave.size))

    #####################
    # Read in the images.
    #####################

    for j in range(len(config.images["file"])):
        config.images["data"].append(im.Image())
        config.images["data"][j].read(config.images["file"][j])

    return config.visibilities, config.images, config.spectra
