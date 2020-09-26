.. pdspy documentation master file, created by
   sphinx-quickstart on Sat Sep 26 13:13:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pdspy's documentation!
=================================

This code is meant to fit Monte Carlo Radiative Transfer models for protostellar/protoplanetary disks to ALMA continuum and spectral line datasets using Markov Chain Monte Carlo fitting. There are currently three well tested tools to run models:

+ **disk_model.py**: Used to fit ALMA continuum visibilities and broadband spectral energy distributions (SEDs) with full radiative transfer models.

+ **disk_model_powerlaw.py**: Used to fit ALMA continuum visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

+ **flared_model.py**: Used to fit ALMA spectral line visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

Further capabilities (e.g. fitting spectral line data with a radiative equilibrium calculation) are being developed. If you are interested in new features, do let me know and I would be happy to either add them myself, or to work with you to add them. The documentation is currently included below, but will be ported to a more extensive, better laid out format soon. For more extensive details on what the code does, please see these papers:

   + `Disk Masses for Embedded Class I Protostars in the Taurus Molecular Cloud <https://ui.adsabs.harvard.edu/abs/2017ApJ...851...45S/abstract>`_
   + `High-precision Dynamical Masses of Pre-main-sequence Stars with ALMA and Gaia <https://ui.adsabs.harvard.edu/abs/2019ApJ...874..136S/abstract>`_
   
If you have any questions about using the code (or this documentation), requests for features, or suggestions for improvement, please don't hesitate to send me an e-mail.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation.rst
   fitting.rst



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
