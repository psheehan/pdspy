=======================
Upgrading to v2 from v1
=======================

pdspy v2.0.0 represents a major upgrade to the pdspy infrastructure, and is not backwards compatible with models that were run using earlier versions.

What's new?
"""""""""""

v1.* of the pdspy code fixed longstanding issues with the orientation parameters (x0, y0, pa, etc.). Previously x0 and y0 were entirely backwards from their traditional definitions, and pa was even worse. For models with density reductions, the mass of the component with the density reduction was not being calculated accurately. Now the values that are reported from the modeling are entirely accurate.

Why is it not backwards compatible?
===================================

The definition of a number of parameters fundamentally changed and so if you try to run models with the results of v1.* fits, they will not give correct results using v2.* code.

What needs to be updated?
=========================

There are a few places in particular that you need to look to make sure that your files are up-to-date and ready for v2:

* config.py

  * *x0/y0*: These are now defined such that in a CASA image x0 is positive to the left and y0 is positive up.

* Visibility HDF5 files: These files *must* be recreated from their UVFITS or CAASA MS origins. In order to get images oriented correctly when made from the visibility data, pdspy now takes the complex conjugate of the data when reading from UVFITS or MS files. The HDF5 visibility files store and read the data exactly as-is, which means that they need to be recreated in order to have the proper orientation.

This sounds like a pain...
==========================

You're telling me... but to make your life easier, there is a helper script upgrade_to_pdspy2 that *should* take care of the transition for you. To use, simply go to the directory of an existing model and run the script. *Note:* The script will make permanent changes to files in that directory, and though it makes a backup copy of everything, it would perhaps be wise to make your own backup beforehand.
