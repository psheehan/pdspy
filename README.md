# pdspy: A MCMC Tool for Continuum and Spectral Line Radiative Transfer Modeling

Welcome to the documentation for pdspy! This code is meant to fit Monte Carlo Radiative Transfer models for protostellar/protoplanetary disks to ALMA continuum and spectral line datasets using Markov Chain Monte Carlo fitting. There are currently three well tested tools to run models:

+ **disk_model.py**: Used to fit ALMA continuum visibilities and broadband spectral energy distributions (SEDs) with full radiative transfer models.

+ **disk_model_powerlaw.py**: Used to fit ALMA continuum visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

+ **flared_model.py**: Used to fit ALMA spectral line visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

Further capabilities (e.g. fitting spectral line data with a radiative equilibrium calculation) are being developed. If you are interested in new features, do let me know and I would be happy to either add them myself, or to work with you to add them. The documentation is currently included below, but will be ported to a more extensive, better laid out format soon. For more extensive details on what the code does, please see these papers:

   + [Disk Masses for Embedded Class I Protostars in the Taurus Molecular Cloud](https://ui.adsabs.harvard.edu/abs/2017ApJ...851...45S/abstract)
   + [High-precision Dynamical Masses of Pre-main-sequence Stars with ALMA and Gaia](https://ui.adsabs.harvard.edu/abs/2019ApJ...874..136S/abstract)
   
If you have any questions about using the code (or this documentation), requests for features, or suggestions for improvement, please don't hesitate to send me an e-mail.

## Installation

### Installing the code with Anaconda

Anaconda is probably the easiest way to install pdspy

1. Download the code from this webpage

2. In a terminal, in the directory where the code was downloaded to:

   ```
   conda build pdspy -c conda-forge
   conda install pdspy -c conda-forge --use-local
   ```

### Installing the code with pip

1. In a terminal, run:

   ```
   pip install pdspy
   ```

2. Install GALARIO. Unfortunately, GALARIO is not pip-installable, so you will need to follow the instructions [here](https://mtazzari.github.io/galario/).

### Installing the code manually

1. Download the code from this webpage. Git clone is recommended if you would like to be able to pull updates:

   ```
   git clone https://github.com/psheehan/pdspy.git
   ```

2. Install the Python dependencies (recommended with pip, when available):

   numpy  
   scipy  
   matplotlib  
   emcee  
   corner  
   hyperion  
   h5py  
   mpi4py  
   galario  
   Cython  
   astropy < 4.0  
   schwimmbad  
   dynesty

3. In a terminal, go to the directory where the code was downloaded, and into the code directory. Run:

   ```
   python setup.py install
   ```
   
   or
   
   ```
   pip install -e .
   ```
   
### Other dependencies

The other codes that are needed to run pdspy are [Hyperion](http://www.hyperion-rt.org) and [RADMC-3D](http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/). If you are a [Homebrew](https://brew.sh) user, you can do this with:

   ```
   brew tap psheehan/science
   brew install hyperion
   brew install radmc3d
   ```

## Fitting data with radiative transfer models.

This section is for setting up your data and running a fit of a disk radiative transfer model to it.

### Preparing your data to be in the correct format

Once CASA6 is released, there will be functionality to read data directly from CASA MS files, however until CASA6 is available, I put the data into my own HDF5 format. Here’s how:

1. Within CASA, use the “exportuvfits” to split every spectral window that you care about into a separate UV FITS file. Each MS file should go into a separate “.vis” file:

   ```
   filenameA.vis/  
   |---- filenameA.1.uv.fits  
   |---- filenameA.2.uv.fits  
   .  
   .  
   .  
   \---- filenameA.N.uv.fits  
   filenameB.vis/  
   |---- filenameB.1.uv.fits  
   |---- filenameB.2.uv.fits  
   .  
   .  
   .  
   \---- filenameB.N.uv.fits
   ```

   I’ll typically organize this by array configuration and band, so it may look like this:

   ```
   source_Band3_track1.vis  
   source_Band3_track2.vis  
   source_Band6_track1.vis  
   ```

2. Use the below code to turn the “.vis” files into HDF5 files. 

   ```python
   import pdspy.interferometry as uv  
   import glob  

   files = glob.glob("*Band3*.vis")  

   data = []  
   for file in files:  
       data.append(uv.freqcorrect(uv.readvis(file)))  

   vis = uv.concatenate(data)  

   vis.write("Source_Band3.hdf5")
   ```

   It’ll grab all of the “\*.vis” files that match the wildcard at the beginning, so you can adjust that to decide which sets of files get grabbed. So in the above example you could run it once with “\*Band3\*.vis” to merge the Band 3 data into one file, and then “\*Band5\*.vis” to merge the Band 6 data into a single dataset.

### Setting up a configuration file

You can find a basic configuration file in the pdspy bin directory ([config_template.py](https://github.com/psheehan/pdspy/blob/master/bin/config_template.py)) as an example, and I think it should be close to what you’ll want for your application. The visibilities dictionary requests a bunch of information about the visibility data. The things in particular you’ll want to update are:

**“file”:** the HDF5 visibility files the were created above. Can list as many as you’d like, I just put in 2 as an example. (All of the entries in the visibilities dictionary should be lists with the same number of elements).

**“freq”/“lam”:** The frequency/wavelength of the observations. Freq should be a string, “lam” a number.

**“x0”/“y0”:** If the data is far off-center, these are initial corrections to approximately center the data. I believe positive x0 means west and positive y0 is south (i.e. perfectly backwards; a relic of not catching the problem until I was in too deep).

**“image_file”:** every HDF5 file should have a corresponding FITS image to show the best fit model over. All of the other “image_*” parameters correspond to values from the image: pixelsize, npix

Then at the bottom the **parameters** dictionary gives you a giant list of parameters that can be turned on or off. When a parameter has “fixed”:True, then it is fixed at a value of “value”. If “fixed”:False, then it’s a free parameter constrained by “limits”. For a full list of parameters, see [here](https://github.com/psheehan/pdspy/blob/master/pdspy/modeling/base_parameters.py)

The **“flux_unc\*”** parameters at the bottom add a flux uncertainty to the observations, with “sigma”:0.1 = 10% uncertainty (but that can be changed), and a Gaussian prior. You can add as many of these as you have visibility files, so you can tune the flux uncertainty separately for each dataset.

### Running a model

Make sure “/path/to/pdspy/bin” is in your PATH so that you can see the disk_model.py function. There are currently three well tested tools to run models:

+ **disk_model.py**: Used to fit ALMA continuum visibilities and broadband spectral energy distributions (SEDs) with full radiative transfer models.

+ **disk_model_powerlaw.py**: Used to fit ALMA continuum visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

+ **flared_model.py**: Used to fit ALMA spectral line visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

From there the most basic way to run any one of these models is in the directory with config.py and entering:

disk_model.py --object <Object Name>

If you want to run with parallel RADMC-3D, to speed up the code, you can run:

disk_model.py --object <Object Name> --ncpus N

Progress is saved, so if you want to resume a fit that stopped for some reason, you can add:

disk_model.py --object <Object Name> --ncpus N --resume

You can also use MPI to run multiple single core models at once:

mpirun -np N disk_model.py --object <Object Name> --ncpus 1

Or some combination of simultaneous models and parallel RADMC-3D:

mpirun -np N disk_model.py --object <Object Name> --ncpus M

(where NxM should be <= the number of cores on your computer). The last two commands for running the code (using MPI) make it adaptable so that it can be run on supercomputers as well, for an even bigger boost. If you want to do this, let me know and I can provide some more details of how to efficiently run over multiple supercomputer nodes.
