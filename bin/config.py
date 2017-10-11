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
