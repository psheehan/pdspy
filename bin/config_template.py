import numpy

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
        "weight" = [10.],
        "ticks":[numpy.array([-250,-200,-100,0,100,200,250])],
        # Info for the image.
        "image_file":["path/to/image_file1"],
        "image_pixelsize":[0.05],
        "image_npix":[1024],
        "image_ticks":[numpy.array([-6.0,-5.0,-2.5,0,2.5,5.0,6.0])],
        }

# Something similar for images.

images = {
        "file":["path/to/file1"],
        "npix":[128],
        "pixelsize":[0.1],
        "lam":["0.8"],
        "bmaj":[1.0],
        "bmin":[1.0],
        "bpa":[0.0],
        "ticks":[numpy.array([-6,-3,0,3,6])],
        "plot_mode":["linear"],
        }

# Something similar for spectra.

spectra = {
        "file":["path/to/file1"],
        "bin?":[False],
        "nbins":[25],
        }

################################################################################
#
# Set up the list of parameters and default fixed values.
#
################################################################################

parameters = {
        # Stellar parameters.
        "logM_star":{"fixed":True, "value":1.0, "limits":[-1.,1.]},
        "T_star":{"fixed":True, "value":4000., "limits":[500.,10000.]},
        "logL_star":{"fixed":True, "value":0.0, "limits":[-1.,2.]},
        # Disk parameters.
        "logM_disk":{"fixed":True, "value":-4., "limits":[-10.,-2.5]},
        "logR_in":{"fixed":True, "value":-1., "limits":[-1.,4.]},
        "logR_disk":{"fixed":True, "value":2., "limits":[0.,4.]},
        "h_0":{"fixed":True, "value":0.1, "limits":[0.01,0.5]},
        "gamma":{"fixed":True, "value":1.0, "limits":[-0.5,2.0]},
        "beta":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Envelope parameters.
        "logM_env":{"fixed":True, "value":-3., "limits":[-10., -2.]},
        "logR_in_env":{"fixed":True, "value":"logR_in", "limits":[-1., 4.]},
        "logR_env":{"fixed":True, "value":3., "limits": [2.,5.]},
        "logR_c":{"fixed":True, "value":"logR_disk", "limits":[-1.,4.]},
        "f_cav":{"fixed":True, "value":0.5, "limits":[0.,1.]},
        "ksi":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Dust parameters.
        "loga_max":{"fixed":True, "value":0., "limits":[0.,5.]},
        "p":{"fixed":True, "value":3.5, "limits":[2.5,4.5]},
        # Viewing parameters.
        "i":{"fixed":True, "value":45., "limits":[0.,180.]},
        "pa":{"fixed":True, "value":0., "limits":[0.,360.]},
        "x0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "y0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "dpc":{"fixed":True, "value":140., "limits":[1.,1e6]},
        "Ak":{"fixed":True, "value":0., "limits":[0.,1.]},
        }
