from os import system

import sys
if sys.version_info.major > 2:
    from subprocess import STDOUT, run
else:
    from subprocess32 import STDOUT, run

def thermal(noscat=None, nphot_therm=None, nphot_scat=None, setthreads=1, \
        inclfreefree=None, nofreefree=None, inclgascont=None, nogascont=None, \
        verbose=True, timelimit=7200, nice=None):

    if nice != None:
        command="nice -{0:d} radmc3d mctherm".format(nice)
    else:
        command="radmc3d mctherm"

    if (noscat == True):
        command += " noscat"
    if (nphot_therm != None):
        command += " nphot_therm {0:d}".format(nphot_therm)
    if (setthreads != 1):
        command += " setthreads {0:d}".format(setthreads)
    if (inclfreefree == True):
        command += " inclfreefree"
    if (nofreefree == True):
        command += " nofreefree"
    if (inclgascont == True):
        command += " inclgascont"
    if (nogascont == True):
        command += " nogascont"

    if not verbose:
        f = open("radmc3d.out","w")
        output = run(command.split(" "), stdout=f, stderr=f, timeout=timelimit)
        f.close()
    else:
        output = run(command.split(" "), stderr=STDOUT, timeout=timelimit)

def scattering(nphot_scat=None, setthreads=1, inclfreefree=None, \
        nofreefree=None, inclgascont=None, nogascont=None, \
        verbose=True, nice=None, loadlambda=None):

    if nice != None:
        command="nice -{0:d} radmc3d mcscat".format(nice)
    else:
        command="radmc3d mcscat "

    if (nphot_scat != None):
        command += " nphot_scat {0:d}".format(nphot_therm)
    if (setthreads != 1):
        command += " setthreads {0:d}".format(setthreads)
    if (inclfreefree == True):
        command += " inclfreefree"
    if (nofreefree == True):
        command += " nofreefree"
    if (inclgascont == True):
        command += " inclgascont"
    if (nogascont == True):
        command += " nogascont"
    if (loadlambda == True):
        command += "loadlambda "

    if not verbose:
        command += " > radmc3d.out"

    system(command)

def sed(nrrefine=None, fluxcons=None, norefine=None, nofluxcons=None, \
        noscat=None, sizeau=None, sizepc=None, zoomau=None, zoompc=None, \
        truepix=None, truezoom=None, pointau=None, pointpc=None, incl=None, \
        phi=None, posang=None, circ=None, apert=None, useapert=None, \
        noapert=None, nphot_scat=None, inclstar=None, nostar=None, \
        inclline=None, noline=None, incldust=None, nodust=None, \
        inclfreefree=None, nofreefree=None, inclgascont=None, nogascont=None, \
        loadlambda=None, verbose=True, nice=None):

    if nice != None:
        command="nice -{0:d} radmc3d spectrum ".format(nice)
    else:
        command="radmc3d spectrum "

    if (nrrefine != None):
        command += "nrrefine {0:i} ".format(nrrefine)
    if (fluxcons == True):
        command += "fluxcons "
    if (norefine == True):
        command += "norefine "
    if (nofluxcons == True):
        command += "nofluxcons "
    if (noscat == True):
        command += "noscat "
    if (sizeau != None):
        command += "sizeau {0:f} ".format(sizeau)
    if (sizepc != None):
        command += "sizepc {0:f} ".format(sizepc)
    if (zoomau != None):
        command += "zoomau {0:f} {1:f} {2:f} {3:f} ".format(zoomau[0], \
                zoomau[1], zoomau[2], zoomau[3])
    if (zoompc != None):
        command += "zoompc {0:f} {1:f} {2:f} {3:f} ".format(zoompc[0], \
                zoompc[1], zoompc[2], zoompc[3])
    if (truepix == True):
        command += "truepix "
    if (truezoom == True):
        command += "truezoom "
    if (pointau != None):
        command += "pointau {0:f} {1:f} {2:f} ".format(pointau[0], pointau[1], \
                pointau[2])
    if (pointpc != None):
        command += "pointpc {0:f} {1:f} {2:f} ".format(pointpc[0], pointpc[1], \
                pointpc[2])
    if (incl != None):
        command += "incl {0:f} ".format(incl)
    if (phi != None):
        command += "phi {0:f} ".format(phi)
    if (posang != None):
        command += "posang {0:f} ".format(posang)
    if (circ == True):
        command += "circ "
    if (apert == True):
        command += "apert "
    if (useapert == True):
        command += "useapert "
    if (noapert == True):
        command += "noapert "
    if (nphot_scat != None):
        command += "nphot_scat {0:d} ".format(nphot_scat)
    if (inclstar == True):
        command += "inclstar "
    if (nostar == True):
        command += "nostar "
    if (inclline == True):
        command += "inclline "
    if (noline == True):
        command += "noline "
    if (incldust == True):
        command += "incldust "
    if (nodust == True):
        command += "nodust "
    if (inclfreefree == True):
        command += "inclfreefree "
    if (nofreefree == True):
        command += "nofreefree "
    if (inclgascont == True):
        command += "inclgascont "
    if (nogascont == True):
        command += "nogascont "
    if (loadlambda == True):
        command += "loadlambda "

    if not verbose:
        command += " > radmc3d.out"

    system(command)

def image(lam=None, npix=None, npixx=None, npixy=None, nrrefine=None, \
        fluxcons=None, norefine=None, nofluxcons=None, noscat=None, \
        ilambda=None, inu=None, color=None, loadcolor=None, loadlambda=None, \
        sizeau=None, sizepc=None, zoomau=None, zoompc=None, truepix=None, \
        truezoom=None, pointau=None, pointpc=None, incl=None, phi=None, \
        posang=None, imageunform=None, imageformatted=None, tracetau=None, \
        tracecolumn=None, tracenormal=None, apert=None, useapert=None, \
        noapert=None, nphot_scat=None, inclstar=None, nostar=None, \
        inclline=None, noline=None, incldust=None, nodust=None, \
        inclfreefree=None, nofreefree=None, inclgascont=None, nogascont=None, \
        widthkms=None, vkms=None, linenlam=None, iline=None, imolspec=None, \
        doppcatch=None, verbose=True, nice=None, unstructured=False, \
        circ=False):

    if nice != None:
        command="nice -{0:d} radmc3d image ".format(nice)
    else:
        command="radmc3d image "

    if (circ):
        command += "circ "
    if (lam != None):
        command += "lambda "+lam+" "
    if (iline != None):
        command += "iline {0:d} ".format(iline)
    if (imolspec != None):
        command += "imolspec {0:d} ".format(imolspec)
    if (widthkms != None):
        command += "widthkms {0:f} ".format(widthkms)
    if (vkms != None):
        command += "vkms {0:f} ".format(vkms)
    if (doppcatch == True):
        command += "doppcatch "
    if (linenlam != None):
        command += "linenlam {0:d} ".format(linenlam)
    if (npix != None):
        command += "npix {0:d} ".format(npix)
    if (npixx != None):
        command += "npixx {0:i} ".format(npixx)
    if (npixy != None):
        command += "npixy {0:i} ".format(npixy)
    if (nrrefine != None):
        command += "nrrefine {0:i} ".format(nrrefine)
    if (fluxcons == True):
        command += "fluxcons "
    if (norefine == True):
        command += "norefine "
    if (nofluxcons == True):
        command += "nofluxcons "
    if (noscat == True):
        command += "noscat "
    if (ilambda != None):
        command += "ilambda {0:i} ".format(ilambda)
    if (inu != None):
        command += "inu {0:i} ".format(inu)
    if (color == True):
        command += "color "
    if (loadcolor == True):
        command += "loadcolor "
    if (loadlambda == True):
        command += "loadlambda "
    if (sizeau != None):
        command += "sizeau {0:f} ".format(sizeau)
    if (sizepc != None):
        command += "sizepc {0:f} ".format(sizepc)
    if (zoomau != None):
        command += "zoomau {0:f} {1:f} {2:f} {3:f} ".format(zoomau[0], \
                zoomau[1], zoomau[2], zoomau[3])
    if (zoompc != None):
        command += "zoompc {0:f} {1:f} {2:f} {3:f} ".format(zoompc[0], \
                zoompc[1], zoompc[2], zoompc[3])
    if (truepix == True):
        command += "truepix "
    if (truezoom == True):
        command += "truezoom "
    if (pointau != None):
        command += "pointau {0:f} {1:f} {2:f} ".format(pointau[0], pointau[1], \
                pointau[2])
    if (pointpc != None):
        command += "pointpc {0:f} {1:f} {2:f} ".format(pointpc[0], pointpc[1], \
                pointpc[2])
    if (incl != None):
        command += "incl {0:f} ".format(incl)
    if (phi != None):
        command += "phi {0:f} ".format(phi)
    if (posang != None):
        command += "posang {0:f} ".format(posang)
    if (imageunform == True):
        command += "imageunform "
    if (imageformatted == True):
        command += "imageformatted "
    if (tracetau == True):
        command += "tracetau "
    if (tracecolumn == True):
        command += "tracecolumn "
    if (tracenormal == True):
        command += "tracenormal "
    if (apert == True):
        command += "apert "
    if (useapert == True):
        command += "useapert "
    if (noapert == True):
        command += "noapert "
    if (nphot_scat != None):
        command += "nphot_scat {0:i} ".format(nphot_scat)
    if (inclstar == True):
        command += "inclstar "
    if (nostar == True):
        command += "nostar "
    if (inclline == True):
        command += "inclline "
    if (noline == True):
        command += "noline "
    if (incldust == True):
        command += "incldust "
    if (nodust == True):
        command += "nodust "
    if (inclfreefree == True):
        command += "inclfreefree "
    if (nofreefree == True):
        command += "nofreefree "
    if (inclgascont == True):
        command += "inclgascont "
    if (nogascont == True):
        command += "nogascont "

    if unstructured:
        command += "diag_subpix "

    if not verbose:
        command += " > radmc3d.out"

    system(command)


# The code below has every single command, if you need to implement more command line options.
"""
def run_thermal(npix=None, npixx=None, npixy=None, nrrefine=None, \
        fluxcons=None, norefine=None, nofluxcons=None, noscat=None, \
        ilambda=None, inu=None, color=None, loadcolor=None, loadlambda=None, \
        sizeau=None, sizepc=None, zoomau=None, zoompc=None, truepix=None, \
        truezoom=None, pointau=None, pointpc=None, incl=None, phi=None, \
        posang=None, imageunform=None, imageformatted=None, circ=None, \
        tracetau=None, tracecolumn=None, tracenormal=None, apert=None, \
        useapert=None, noapert=None, nphot_therm=None, nphot_scat=None, \
        nphot_mcmono=None, inclstar=None, nostar=None, inclline=None, \
        noline=None, incldust=None, nodust=None, inclfreefree=None, \
        nofreefree=None, inclgascont=None, nogascont=None):

    command="radmc3d mctherm "

    if (npix != None):
        command += "npix {0:i} ".format(npix)
    if (npixx != None):
        command += "npixx {0:i} ".format(npixx)
    if (npixy != None):
        command += "npixy {0:i} ".format(npixy)
    if (nrrefine != None):
        command += "nrrefine {0:i} ".format(nrrefine)
    if (fluxcons == True):
        command += "fluxcons "
    if (norefine == True):
        command += "norefine "
    if (nofluxcons == True):
        command += "nofluxcons "
    if (noscat == True):
        command += "noscat "
    if (ilambda != None):
        command += "ilambda {0:i} ".format(ilambda)
    if (inu != None):
        command += "inu {0:i} ".format(inu)
    if (color == True):
        command += "color "
    if (loadcolor == True):
        command += "loadcolor "
    if (loadlambda == True):
        command += "loadlambda "
    if (sizeau != None):
        command += "sizeau {0:f} ".format(sizeau)
    if (sizepc != None):
        command += "sizepc {0:f} ".format(sizepc)
    if (zoomau != None):
        command += "zoomau {0:f} {1:f} {2:f} {3:f} ".format(zoomau[0], \
                zoomau[1], zoomau[2], zoomau[3])
    if (zoompc != None):
        command += "zoompc {0:f} {1:f} {2:f} {3:f} ".format(zoompc[0], \
                zoompc[1], zoompc[2], zoompc[3])
    if (truepix == True):
        command += "truepix "
    if (truezoom == True):
        command += "truezoom "
    if (pointau != None):
        command += "pointau {0:f} {1:f} {2:f} ".format(pointau[0], pointau[1], \
                pointau[2])
    if (pointpc != None):
        command += "pointpc {0:f} {1:f} {2:f} ".format(pointpc[0], pointpc[1], \
                pointpc[2])
    if (incl != None):
        command += "incl {0:f} ".format(incl)
    if (phi != None):
        command += "phi {0:f} ".format(phi)
    if (posang != None):
        command += "posang {0:f} ".format(posang)
    if (imageunform == True):
        command += "imageunform "
    if (imageformatted == True):
        command += "imageformatted "
    if (circ == True):
        command += "circ "
    if (tracetau == True):
        command += "tracetau "
    if (tracecolumn == True):
        command += "tracecolumn "
    if (tracenormal == True):
        command += "tracenormal "
    if (apert == True):
        command += "apert "
    if (useapert == True):
        command += "useapert "
    if (noapert == True):
        command += "noapert "
    if (nphot_therm != None):
        command += "nphot_therm {0:i} ".format(nphot_therm)
    if (nphot_scat != None):
        command += "nphot_scat {0:i} ".format(nphot_scat)
    if (nphot_mcmono != None):
        command += "nphot_mcmono {0:i} ".format(nphot_mcmono)
    if (inclstar == True):
        command += "inclstar "
    if (nostar == True):
        command += "nostar "
    if (inclline == True):
        command += "inclline "
    if (noline == True):
        command += "noline "
    if (incldust == True):
        command += "incldust "
    if (nodust == True):
        command += "nodust "
    if (inclfreefree == True):
        command += "inclfreefree "
    if (nofreefree == True):
        command += "nofreefree "
    if (inclgascont == True):
        command += "inclgascont "
    if (nogascont == True):
        command += "nogascont "

    system(command)
"""
