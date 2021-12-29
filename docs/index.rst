.. pdspy documentation master file, created by
   sphinx-quickstart on Sat Sep 26 13:13:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pdspy's documentation!
=================================

This code is meant to fit Monte Carlo Radiative Transfer models for protostellar/protoplanetary disks to ALMA continuum and spectral line datasets using Markov Chain Monte Carlo fitting. There are currently two well tested tools to run models:

+ **disk_model_emcee3.py or disk_model_nested.py**: Used to fit ALMA continuum visibilities and broadband spectral energy distributions (SEDs) with full radiative transfer models.

+ **flared_model_emcee3.py or flared_model_nested.py**: Used to fit ALMA spectral line visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

Further capabilities (e.g. fitting spectral line data with a radiative equilibrium calculation) are being developed. If you are interested in new features, do let me know and I would be happy to either add them myself, or to work with you to add them. For more extensive details on what the code does, please see these papers:

   + `Disk Masses for Embedded Class I Protostars in the Taurus Molecular Cloud <https://ui.adsabs.harvard.edu/abs/2017ApJ...851...45S/abstract>`_
   + `High-precision Dynamical Masses of Pre-main-sequence Stars with ALMA and Gaia <https://ui.adsabs.harvard.edu/abs/2019ApJ...874..136S/abstract>`_
   
If you have any questions about using the code (or this documentation), requests for features, or suggestions for improvement, please don't hesitate to send me an e-mail.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation.rst
   upgrading.rst
   models.rst
   fitting.rst
   postprocess.rst

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modeling
   plotting
   utils

Changelog
=========

2.0.0 (2021-11-07)
------------------

A major upgrade to pdspy, which is not backwards compatible to versions < 2.0.0. Updates include:

* x0, y0 now defined in the proper sense as opposed to backwards as they were before
* Position angles reported are now defined correctly, so no post-processing needed
* Supply the code directly with CASA MS files (requires Python 3.6 and CASA6)
* Fourier Transforms of model images are now done entirely with either GALARIO or TriFT, and an option to choose between them exists.

1.5.2 (2021-10-19)
------------------

Further updates to v1.5, including:

* Better estimate of the CLEAN beam shape in uv.clean
* New routine to plot PV diagrams of the results of model fits
* Use KDE to find peaks in the posteriors of model fits.
* A number of minor bug fixes

1.5.1 (2021-09-08)
------------------

Some minor updates to v1.5, including:

* Adding a tool to load results of fits under the hood, with a number of options
* Ability to adjust weighting of images in plot_channel_maps
* Fully working readms function

1.5.0 (2020-09-24)
------------------

A significant series of updates, but not quite enough to get to version 2.0, hence 1.5. The changes include:

* Adding support for emcee v3
* Adding support for fitting with dynesty
* Adding a non-vertically isothermal option to run_flared_model
* Added plotting tools

And many more bug fixes and smaller features.

1.0.0 (2018-12-20)
------------------

Initial commit.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Support
-------

If you are having issues, please let me know at psheehan at northwestern dot edu.

License
-------

The project is licensed under the GPL-3.0 license.
