from ..constants.astronomy import arcsec
from ..modeling import check_parameters
from ..modeling import base_parameters
import importlib
import sys

def load_config(path=''):
    r"""
    Load a pdspy configuration file in as a module.

    Args:
        :attr:`path` (`str`):
            The path to the directory where the information for the radiative transfer model is stored.

    Returns:
        :attr:`config` (`module`):
            A python module containing the configuration information for pdspy modeling.
    """

    # Import the configuration file information.

    sys.path.append(path)

    import config
    importlib.reload(config)

    sys.path.remove(path)

    # If some set of information isn't supplied, provide it.

    if not hasattr(config, "visibilities"):
        config.visibilities = {\
            "file":[], \
            "pixelsize":[],\
            "freq":[],\
            "lam":[],\
            "npix":[],\
            "nphi":[],\
            'nr':[],\
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

    # Make sure the fmt keyword is in the visibilities dictionary.

    if not "fmt" in config.visibilities:
        config.visibilities["fmt"] = ['4.1f' for i in range(len(\
                config.visibilities["file"]))]

    # Make sure the visibilities dictionary has SPW, tolerance, and whether
    # corrected or data column in it.

    if not "spw" in config.visibilities:
        config.visibilities["spw"] = [[0] for i in range(len(\
                config.visibilities["file"]))]
    if not "tolerance" in config.visibilities:
        config.visibilities["tolerance"] = [0.01 for i in range(len(\
                config.visibilities["file"]))]
    if not "datacolumn" in config.visibilities:
        config.visibilities["datacolumn"] = ["corrected" for i in range(len(\
                config.visibilities["file"]))]

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

    # Make sure all of the controls for circular images are in the visibilities
    # dictionary.

    if not "nphi" in config.visibilities:
        config.visibilities["nphi"] = [128 for i in range(len(\
                config.visibilities["file"]))]
    if not "nr" in config.visibilities:
        config.visibilities["nr"] = [1 for i in range(len(\
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
