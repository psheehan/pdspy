#!/usr/bin/env python3

from ..constants.physics import c, m_p, G
from ..constants.physics import k as k_b
from ..constants.astronomy import M_sun, AU
from .YSOModel import YSOModel
from .. import interferometry as uv
from .. import spectroscopy as sp
from .. import misc
from .. import dust
from .. import gas
import scipy.signal
import numpy
import time
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

def run_flared_model(visibilities, params, parameters, plot=False, ncpus=1, \
        source="flared", plot_vis=False, nice=None):

    # Set the values of all of the parameters.

    p = {}
    for key in parameters:
        if parameters[key]["fixed"]:
            if parameters[key]["value"] in parameters.keys():
                if parameters[parameters[key]["value"]]["fixed"]:
                    value = parameters[parameters[key]["value"]]["value"]
                else:
                    value = params[parameters[key]["value"]]
            else:
                value = parameters[key]["value"]
        else:
            value = params[key]

        if key[0:3] == "log":
            p[key[3:]] = 10.**value
        else:
            p[key] = value

    # Make sure alpha and beta are defined.

    if p["disk_type"] in ["exptaper","dartois-exptaper"]:
        t_rdisk = p["T0"] * (p["R_disk"] / 1.)**-p["q"]
        p["h_0"] = ((k_b*(p["R_disk"]*AU)**3*t_rdisk) / (G*p["M_star"]*M_sun * \
                2.37*m_p))**0.5 / AU
    else:
        p["h_0"] = ((k_b * AU**3 * p["T0"]) / (G*p["M_star"]*M_sun * \
                2.37*m_p))**0.5 / AU
    p["beta"] = 0.5 * (3 - p["q"])
    p["alpha"] = p["gamma"] + p["beta"]

    # Set up the dust.

    dustopac = p["dust_file"]

    dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/"+dustopac)

    ddust = dust_gen(p["a_max"] / 1e4, p["p"])
    edust = dust_gen(1.0e-4, 3.5)

    # Set up the gas.

    gases = []
    abundance = []
    freezeout = []

    index = 1
    while index > 0:
        if "gas_file"+str(index) in p:
            g = gas.Gas()
            g.set_properties_from_lambda(gas.__path__[0]+"/data/"+\
                    p["gas_file"+str(index)])

            gases.append(g)
            abundance.append(p["abundance"+str(index)])
            freezeout.append(p["freezeout"+str(index)])

            index += 1
        else:
            index = -1

    # Make sure we are in a temp directory to not overwrite anything.

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))
    os.chdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for key in p:
        f.write("{0:s} = {1}\n".format(key, p[key]))
    f.close()

    # Set up the model. 

    m = YSOModel()
    m.add_star(mass=p["M_star"], luminosity=p["L_star"],temperature=p["T_star"])

    if p["envelope_type"] == "ulrich":
        p["R_grid"] = p["R_env"]
    else:
        p["R_grid"] = max(5*p["R_disk"],300)
    m.set_spherical_grid(p["R_in"], p["R_grid"], 100, 51, 2, code="radmc3d")

    if p["disk_type"] == "exptaper":
        m.add_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
                aturb=p["a_turb"])
    elif p["disk_type"] == "dartois-exptaper":
        m.add_dartois_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], \
                rmax=p["R_disk"], plrho=p["alpha"], h0=p["h_0"], plh=p["beta"],\
                dust=ddust, tmid0=p["tmid0"], tatm0=p["tatm0"], zq0=p["zq0"], \
                pltgas=p["pltgas"], delta=p["delta"], gas=gases, \
                abundance=abundance, freezeout=freezeout, aturb=p["a_turb"])
    elif p["disk_type"] == "dartois-truncated":
        m.add_dartois_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                tmid0=p["tmid0"], tatm0=p["tatm0"], zq0=p["zq0"], \
                pltgas=p["pltgas"], delta=p["delta"], gas=gases, \
                abundance=abundance, freezeout=freezeout, aturb=p["a_turb"])
    else:
        m.add_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
                aturb=p["a_turb"])

    if p["envelope_type"] == "ulrich":
        m.add_ulrich_envelope(mass=p["M_env"], rmin=p["R_in"], rmax=p["R_env"],\
                cavpl=p["ksi"], cavrfact=p["f_cav"], dust=edust, \
                t0=p["T0_env"], tpl=p["q_env"], gas=gases, abundance=abundance,\
                aturb=p["a_turb_env"], rcent=p["R_c"])
    else:
        pass

    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the images/visibilities/SEDs.

    for j in range(len(visibilities["file"])):
        # Shift the wavelengths by the velocities.

        b = p["v_sys"]*1.0e5 / c
        lam = c / visibilities["data"][j].freq / 1.0e-4
        wave = lam * numpy.sqrt((1. - b) / (1. + b))

        # Set the wavelengths for RADMC3D to use.

        m.set_camera_wavelength(wave)

        if p["docontsub"]:
            m.run_image(name=visibilities["lam"][j], nphot=1e5, npix=25, \
                    lam=None, pixelsize=2*p["R_env"]*1.25/p["dpc"] / 25, \
                    tgas_eq_tdust=True, scattering_mode_max=0, incl_dust=True, \
                    incl_lines=True, loadlambda=True, incl=p["i"], pa=p["pa"], \
                    dpc=p["dpc"], code="radmc3d", verbose=False, \
                    writeimage_unformatted=True, setthreads=ncpus, nice=nice, \
                    unstructured=True)

            m.run_image(name="cont", nphot=1e5, npix=25, lam=None, \
                    pixelsize=2*p["R_env"]*1.25/p["dpc"]/25,tgas_eq_tdust=True,\
                    scattering_mode_max=0, incl_dust=True, incl_lines=False, \
                    loadlambda=True, incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                    code="radmc3d", verbose=False, writeimage_unformatted=True,\
                    setthreads=ncpus, nice=nice, unstructured=True)

            m.images[visibilities["lam"][j]].image -= m.images["cont"].image
        else:
            m.run_image(name=visibilities["lam"][j], nphot=1e5, npix=25, \
                    lam=None, pixelsize=2*p["R_env"]*1.25/p["dpc"] / 25, \
                    tgas_eq_tdust=True, scattering_mode_max=0, incl_dust=False,\
                    incl_lines=True, loadlambda=True, incl=p["i"], pa=p["pa"], \
                    dpc=p["dpc"], code="radmc3d", verbose=False, \
                    writeimage_unformatted=True, setthreads=ncpus, nice=nice, \
                    unstructured=True)

        # Extinct the data, if included.

        velocity = c * (float(visibilities["freq"][j])*1.0e9 - \
                visibilities["data"][j].freq)/(float(visibilities["freq"][j])*\
                1.0e9) / 1.0e5

        tau = p["tau0"] * numpy.exp(-(velocity - p["v_ext"])**2 / \
                (2*p["sigma_vext"]**2))

        extinction = numpy.exp(-tau)

        for i in range(len(m.images[visibilities["lam"][j]].freq)):
            m.images[visibilities["lam"][j]].image[:,i] *= extinction[i]

        # Invert to get the visibilities.

        m.visibilities[visibilities["lam"][j]] = uv.interpolate_model(\
                visibilities["data"][j].u, visibilities["data"][j].v, \
                visibilities["data"][j].freq, \
                m.images[visibilities["lam"][j]], dRA=-p["x0"], dDec=-p["y0"], \
                nthreads=ncpus, code="trift")

        if plot:
            lam = c / visibilities["image"][j].freq / 1.0e-4
            wave = lam * numpy.sqrt((1. - b) / (1. + b))

            m.set_camera_wavelength(wave)

            if p["docontsub"]:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus, nice=nice)

                m.run_image(name="cont", nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=False, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus, nice=nice)

                m.images[visibilities["lam"][j]].image -= m.images["cont"].image
            else:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=False, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus, nice=nice)

            # Extinct the data, if included.

            velocity = c * (float(visibilities["freq"][j])*1.0e9 - \
                    visibilities["image"][j].freq) / \
                    (float(visibilities["freq"][j])*1.0e9) / 1.0e5

            tau = p["tau0"] * numpy.exp(-(velocity - p["v_ext"])**2 / \
                    (2*p["sigma_vext"]**2))

            extinction = numpy.exp(-tau)

            for i in range(len(m.images[visibilities["lam"][j]].freq)):
                m.images[visibilities["lam"][j]].image[:,:,i,:] *= extinction[i]

            # Now convolve with the beam.

            x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                    numpy.linspace(-256,255,512))

            beam = misc.gaussian2d(x, y, 0., 0., \
                    visibilities["image"][j].header["BMAJ"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    visibilities["image"][j].header["BMIN"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    (90-visibilities["image"][j].header["BPA"])*\
                    numpy.pi/180., 1.0)

            for ind in range(len(wave)):
                m.images[visibilities["lam"][j]].image[:,:,ind,0] = \
                        scipy.signal.fftconvolve(\
                        m.images[visibilities["lam"][j]].image[:,:,ind,0], \
                        beam, mode="same")

            if plot_vis:
                m.visibilities[visibilities["lam"][j]+"1D"] = \
                        uv.average(m.visibilities[visibilities["lam"][j]], \
                        gridsize=20, radial=True, log=True, \
                        logmin=m.visibilities[visibilities["lam"][j]].uvdist[\
                        numpy.nonzero(m.visibilities[visibilities["lam"][j]].\
                        uvdist)].min()*0.95, logmax=m.visibilities[\
                        visibilities["lam"][j]].uvdist.max()*1.05, \
                        mode="spectralline")

    os.system("rm params.txt")
    os.chdir(original_dir)
    os.system("rmdir /tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    return m
