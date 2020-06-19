from ..constants.astronomy import arcsec
from ..modeling import check_parameters
from ..modeling import base_parameters
import importlib
import sys

def load_config():
    # Import the configuration file information.

    sys.path.insert(0, '')

    import config
    importlib.reload(config)

    # If some set of information isn't supplied, provide it.

    if not hasattr(config, "visibilities"):
        config.visibilities = {\
            "file":[], \
            "pixelsize":[],\
            "freq":[],\
            "lam":[],\
            "npix":[],\
            "weight":[],\
            "x0":[],\
            "y0":[],\
            # Info for the image.
            "image_file":[],\
            "image_pixelsize":[],\
            "image_npix":[],\
            # Info for the plots.
            "nrows":[],\
            "ncols":[],\
            "ind0":[],\
            "ticks":[],\
            "image_ticks":[],\
            "fmt":[],\
            }

    if not hasattr(config, "images"):
        config.images = images = {\
            "file":[],\
            "npix":[],\
            "pixelsize":[],\
            "lam":[],\
            "bmaj":[],\
            "bmin":[],\
            "bpa":[],\
            "ticks":[],\
            "plot_mode":[],\
            }

    if not hasattr(config, "spectra"):
        config.spectra = {\
            "file":[],\
            "bin?":[],\
            "nbins":[],\
            "weight":[],\
            }

    if not hasattr(config, "parameters"):
        config.parameters = base_parameters

    if not hasattr(config, "priors"):
        config.priors = {}

    # Get the correct binsize and number of bins for averaging the visibility
    # data.

    config.visibilities["gridsize"] = []
    config.visibilities["binsize"] = []

    for i in range(len(config.visibilities["file"])):
        config.visibilities["gridsize"].append(config.visibilities["npix"][i])
        config.visibilities["binsize"].append(1/ \
                (config.visibilities["npix"][i]* \
                config.visibilities["pixelsize"][i]*arcsec))

    # Make sure the visibilities dictionary has subsampling and averaging
    # parameters.

    if not "subsample" in config.visibilities:
        config.visibilities["subsample"] = [1 for i in range(len(\
                config.visibilities["file"]))]
    if not "averaging" in config.visibilities:
        config.visibilities["averaging"] = [1 for i in range(len(\
                config.visibilities["file"]))]
    if not "hanning" in config.visibilities:
        config.visibilities["hanning"] = [False for i in range(len(\
                config.visibilities["file"]))]

    # Make sure the spectra dictionary has a "weight" entry.

    if not "weight" in config.spectra:
        config.spectra["weight"] = [1. for i in range(len(\
                config.spectra["file"]))]

    # Check that all of the parameters are correct.

    config.parameters = check_parameters(config.parameters, \
            nvis=len(config.visibilities["file"]))

    # Return all of the configuration information.

    return config
