################################################################################
#
# Set up the list of datasets that will be fit.
#
################################################################################

# Define the necessary info for the visiblity datasets.

visibilities = {
        "file":["path/to/file1"],
        "binsize" = [8057.218995847603],
        "pixelsize" = [0.1],
        "freq" = ["230GHz"],
        "lam" = ["1300"],
        "npix" = [256],
        "gridsize" = [256],
        "weight" = [10.]
        # Info for the image.
        "image_file":["path/to/image_file1"]
        "image_pixelsize":[0.05]
        "image_npix":[1024]
        }

# Something similar for images.

images = {
        "file":["path/to/file1"]
        "npix":[128]
        "pixelsize":[0.1]
        "lam":["0.8"]
        "bmaj":[1.0]
        "bmin":[1.0]
        "bpa":[0.0]
        }

# Something similar for spectra.

spectra = {
        "file":["path/to/file1"]
        "bin?":[False]
        "nbins":[25]
        }

################################################################################
#
# Set up the list of parameters and default fixed values.
#
################################################################################

parameters = {
        # Stellar parameters.
        "logMstar":{"fixed":True, "value":1.0, "limits":[-1.,1.]},
        "Tstar":{"fixed":True, "value":4000., "limits":[500.,10000.]},
        "logLstar":{"fixed":True, "value":1.0, "limits":[-1.,2.]},
        # Disk parameters.
        "logMdisk":{"fixed":True, "value":-4., "limits":[-10.,-2.5]},
        "logRin":{"fixed":True, "value":-1., "limits":[-1.,4.]},
        "logRdisk":{"fixed":True, "value":2., "limits":[0.,4.]},
        "h0":{"fixed":True, "value":0.1, "limits":[0.01,0.5]},
        "gamma":{"fixed":True, "value":1.0, "limits":[-0.5,2.0]},
        "beta":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Envelope parameters.
        "logMenv":{"fixed":True, "value":-3., "limits":[-10., -2.]},
        "logRin_env":{"fixed":True, "value":"logRin", "limits":[-1., 4.]},
        "logRenv":{"fixed":True, "value":3., "limits": [2.,5.]},
        "logRc":{"fixed":True, "value":"logRdisk", "limits":[-1.,4.]},
        "fcav":{"fixed":True, "value":0.5, "limits":[0.,1.]},
        "ksi":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Dust parameters.
        "logamax":{"fixed":True, "value":0., "limits":[0.,5.]},
        "p":{"fixed":True, "value":3.5, "limits":[2.5,4.5]},
        # Viewing parameters.
        "i":{"fixed":True, "value":45., "limits":[0.,180.]},
        "p.a.":{"fixed":True, "value":0., "limits":[0.,360.]},
        "x0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "y0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "dpc":{"fixed":True, "value":140., "limits":[1.,1e6]},
        "Ak":{"fixed":True, "value":0., "limits":[0.,1.]},
        }

