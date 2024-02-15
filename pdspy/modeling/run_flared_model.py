#!/usr/bin/env python3

from ..constants.physics import c, m_p, G
from ..constants.physics import k as k_b
from ..constants.astronomy import M_sun, AU, arcsec
from .YSOModel import YSOModel
from .. import interferometry as uv
from .. import spectroscopy as sp
from .. import misc
from .. import dust
from .. import gas
import scipy.signal
import tempfile
import numpy
import time
import os

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

def run_flared_model(visibilities, params, parameters, plot=False, ncpus=1, \
        source="flared", plot_vis=False, nice=None, ftcode="galario", \
        no_images=False):

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
                p["mu"]*m_p))**0.5 / AU
    else:
        p["h_0"] = ((k_b * AU**3 * p["T0"]) / (G*p["M_star"]*M_sun * \
                p["mu"]*m_p))**0.5 / AU
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
    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for key in p:
        f.write("{0:s} = {1}\n".format(key, p[key]))
    f.close()

    # Set up the model. 

    m = YSOModel()
    m.add_star(mass=p["M_star"], luminosity=p["L_star"],temperature=p["T_star"])

    if "dartois" in p["disk_type"]:
        ntheta = 101
    else:
        ntheta = 51

    if "ulrich" in p["envelope_type"]:
        p["R_grid"] = 20*p["R_env"]
    else:
        p["R_grid"] = max(5*p["R_disk"],300)

    if ftcode == "trift":
        nr = int(numpy.ceil((numpy.log10(p["R_grid"]) - -1.) / 0.04))
        p["R_in_grid"] = 0.1
    else:
        nr = 100
        p["R_in_grid"] = p["R_in"]
    m.set_spherical_grid(p["R_in_grid"], p["R_grid"], nr, ntheta, 2, \
            code="radmc3d")

    if p["disk_type"] == "exptaper":
        m.add_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
                aturb=p["a_turb"], gamma_taper=p["gamma_taper"])
    elif p["disk_type"] == "dartois-exptaper":
        m.add_dartois_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], \
                rmax=p["R_disk"], plrho=p["alpha"], h0=p["h_0"], plh=p["beta"],\
                dust=ddust, tmid0=p["Tmid0"], tatm0=p["Tatm0"], zq0=p["zq0"], \
                pltgas=p["pltgas"], delta=p["delta"], gas=gases, \
                abundance=abundance, freezeout=freezeout, aturb=p["a_turb"], \
                gamma_taper=p["gamma_taper"])
    elif p["disk_type"] == "dartois-truncated":
        m.add_dartois_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                tmid0=p["Tmid0"], tatm0=p["Tatm0"], zq0=p["zq0"], \
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
    elif p["envelope_type"] == "ulrich-extended":
        m.add_ulrichextended_envelope(mass=p["M_env"], rmin=p["R_in"], \
                rmax=p["R_env"], rcent=p["R_c"], cavpl=p["ksi"], \
                cavrfact=p["f_cav"], theta_open=p["theta_open"], \
                zoffset=p["zoffset"], t0=p["T0_env"], tpl=p["q_env"], \
                aturb=p["a_turb_env"], dust=edust, gas=gases, \
                abundance=abundance)
    elif p["envelope_type"] == "ulrich-tapered":
        m.add_tapered_ulrich_envelope(mass=p["M_env"], rmin=p["R_in"], \
                rmax=p["R_env"], gamma=p["gamma_env"], cavpl=p["ksi"], \
                cavrfact=p["f_cav"], dust=edust, t0=p["T0_env"], \
                tpl=p["q_env"], gas=gases, abundance=abundance,\
                aturb=p["a_turb_env"], rcent=p["R_c"])
    elif p["envelope_type"] == "ulrich-tapered-extended":
        m.add_tapered_ulrichextended_envelope(mass=p["M_env"], rmin=p["R_in"], \
                rmax=p["R_env"], gamma=p["gamma_env"], rcent=p["R_c"], \
                cavpl=p["ksi"], cavrfact=p["f_cav"], \
                theta_open=p["theta_open"], zoffset=p["zoffset"], \
                t0=p["T0_env"], tpl=p["q_env"], aturb=p["a_turb_env"], \
                dust=edust, gas=gases, abundance=abundance)
    else:
        pass

    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    if no_images:
        return m

    # Run the images/visibilities/SEDs.

    for j in range(len(visibilities["file"])):
        # If sub-velocity resolution is requested, adjust the frequencies.

        if visibilities["subsample"][j] * visibilities["averaging"][j] > 1:
            dfreq = visibilities["data"][j].freq[1:] - \
                    visibilities["data"][j].freq[0:-1]
            freq = []
            for i in range(0, visibilities["data"][j].freq.size):
                freq.append(visibilities["data"][j].freq[i] + dfreq[i%\
                        (visibilities["data"][j].freq.size-1)] * \
                        (numpy.linspace(1./(visibilities["subsample"][j]*\
                        visibilities["averaging"][j]*2), 1.-1/(\
                        visibilities["subsample"][j]*\
                        visibilities["averaging"][j]*2), \
                        visibilities["subsample"][j]*\
                        visibilities["averaging"][j]) - 0.5))

            freq = numpy.concatenate(freq)
        else:
            freq = visibilities["data"][j].freq

        # Shift the wavelengths by the velocities.

        b = p["v_sys"]*1.0e5 / c
        lam = c / freq / 1.0e-4
        wave = lam * numpy.sqrt((1. - b) / (1. + b))

        # Set the wavelengths for RADMC3D to use.

        m.set_camera_wavelength(wave)

        if ftcode == "galario":
            if p["docontsub"]:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["npix"][j], lam=None, \
                        pixelsize=visibilities["pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice)

                m.run_image(name="cont", nphot=1e5, \
                        npix=visibilities["npix"][j], lam=None, \
                        pixelsize=visibilities["pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=False, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice)

                m.images[visibilities["lam"][j]].image -= m.images["cont"].image
            else:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["npix"][j], lam=None, \
                        pixelsize=visibilities["pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=False, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice)
        else:
            if p["docontsub"]:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, lam=None, \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice, unstructured=True, nostar=True, \
                        camera_circ_nrphiinf=visibilities["nphi"][j], \
                        camera_circ_dbdr=visibilities["nr"][j])

                m.run_image(name="cont", nphot=1e5, lam=None, \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=False, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice, unstructured=True, nostar=True, \
                        camera_circ_nrphiinf=visibilities["nphi"][j], \
                        camera_circ_dbdr=visibilities["nr"][j])

                m.images[visibilities["lam"][j]].image -= m.images["cont"].image
            else:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, lam=None, \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=False, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, \
                        writeimage_unformatted=True, setthreads=ncpus, \
                        nice=nice, unstructured=True, nostar=True, \
                        camera_circ_nrphiinf=visibilities["nphi"][j], \
                        camera_circ_dbdr=visibilities["nr"][j])

        # Extinct the data, if included.

        velocity = c * (float(visibilities["freq"][j])*1.0e9 - freq)/ \
                (float(visibilities["freq"][j])*1.0e9) / 1.0e5

        tau = p["tau0"] * numpy.exp(-(velocity - p["v_ext"])**2 / \
                (2*p["sigma_vext"]**2))

        extinction = numpy.exp(-tau)

        if ftcode == "galario":
            for i in range(len(m.images[visibilities["lam"][j]].freq)):
                m.images[visibilities["lam"][j]].image[:,:,i,:] *= extinction[i]
        else:
            for i in range(len(m.images[visibilities["lam"][j]].freq)):
                m.images[visibilities["lam"][j]].image[:,i] *= extinction[i]

        # And then if sub-sampling, sum the image along the frequency axis to
        # get back to the right size, and also Hanning smooth.

        if visibilities["subsample"][j] * visibilities["averaging"][j] > 1 or \
                visibilities["hanning"][j]:
            # Using regular images.
            if ftcode == "galario":
                recombined = numpy.empty((visibilities["npix"][j], \
                        visibilities["npix"][j], visibilities["data"][j].freq.\
                        size*visibilities["averaging"][j],1))
                for i in range(freq.size // visibilities["subsample"][j]): 
                    recombined[:,:,i,0] = m.images[visibilities["lam"][j]].\
                            image[:,:,i*visibilities["subsample"][j]:(i+1)*\
                            visibilities["subsample"][j],0].mean(axis=2)

                # Now do the Hanning smoothing.

                if visibilities["hanning"][j]:
                    hanning_window = numpy.hanning(5) / numpy.hanning(5).sum()

                    recombined = scipy.signal.fftconvolve(recombined, \
                            hanning_window.reshape((1,1,hanning_window.size,\
                            1)), axes=2, mode="same")

                # Finally, average by the binning.

                binned = numpy.empty((visibilities["npix"][j], \
                        visibilities["npix"][j], visibilities["data"][j].freq.\
                        size, 1))
                for i in range(visibilities["data"][j].freq.size): 
                    binned[:,:,i,0] = recombined[:,:,i*\
                            visibilities["averaging"][j]:(i+1)*\
                            visibilities["averaging"][j],0].mean(axis=2)
            # Using unstructured images.
            else:
                recombined = numpy.empty((m.images[visibilities["lam"][j]].x.\
                        size, visibilities["data"][j].freq.size*\
                        visibilities["averaging"][j]))
                for i in range(freq.size // visibilities["subsample"][j]): 
                    recombined[:,i] = m.images[visibilities["lam"][j]].\
                            image[:,i*visibilities["subsample"][j]:(i+1)*\
                            visibilities["subsample"][j]].mean(axis=1)

                # Now do the Hanning smoothing.

                if visibilities["hanning"][j]:
                    hanning_window = numpy.hanning(5) / numpy.hanning(5).sum()

                    recombined = scipy.signal.fftconvolve(recombined, \
                            hanning_window.reshape((1,hanning_window.size)), \
                            axes=1, mode="same")

                # Finally, average by the binning.

                binned = numpy.empty((m.images[visibilities["lam"][j]].x.size, \
                        visibilities["data"][j].freq.size))
                for i in range(visibilities["data"][j].freq.size): 
                    binned[:,i] = recombined[:,i*visibilities["averaging"]\
                            [j]:(i+1)*visibilities["averaging"][j]].mean(axis=1)

            m.images[visibilities["lam"][j]].image = binned
            m.images[visibilities["lam"][j]].freq = visibilities["data"][j].freq

        # Invert to get the visibilities.

        m.visibilities[visibilities["lam"][j]] = uv.interpolate_model(\
                visibilities["data"][j].u, visibilities["data"][j].v, \
                visibilities["data"][j].freq, \
                m.images[visibilities["lam"][j]], dRA=p["x0"], dDec=p["y0"], \
                nthreads=ncpus, code=ftcode, nxy=visibilities["npix"][j], \
                dxy=visibilities["pixelsize"][j])

        if plot:
            # If sub-velocity resolution is requested, adjust the frequencies.

            if visibilities["subsample"][j] * visibilities["averaging"][j] > 1:
                dfreq = visibilities["image"][j].freq[1:] - \
                        visibilities["image"][j].freq[0:-1]
                freq = []
                for i in range(0, visibilities["image"][j].freq.size):
                    freq.append(visibilities["image"][j].freq[i] + dfreq[i%\
                            (visibilities["image"][j].freq.size-1)] * \
                            (numpy.linspace(1./(visibilities["subsample"][j]*\
                            visibilities["averaging"][j]*2), 1.-1/(\
                            visibilities["subsample"][j]*\
                            visibilities["averaging"][j]*2), \
                            visibilities["subsample"][j]*\
                            visibilities["averaging"][j]) - 0.5))

                freq = numpy.concatenate(freq)
            else:
                freq = visibilities["image"][j].freq

            # Get the wavelengths.

            lam = c / freq / 1.0e-4
            wave = lam * numpy.sqrt((1. - b) / (1. + b))

            m.set_camera_wavelength(wave)

            if p["docontsub"]:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, setthreads=ncpus, \
                        nice=nice)

                m.run_image(name="cont", nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=False, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, setthreads=ncpus, \
                        nice=nice)

                m.images[visibilities["lam"][j]].image -= m.images["cont"].image
            else:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=False, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                        code="radmc3d", verbose=False, setthreads=ncpus, \
                        nice=nice)

            # Extinct the data, if included.

            velocity = c * (float(visibilities["freq"][j])*1.0e9 - freq) / \
                    (float(visibilities["freq"][j])*1.0e9) / 1.0e5

            tau = p["tau0"] * numpy.exp(-(velocity - p["v_ext"])**2 / \
                    (2*p["sigma_vext"]**2))

            extinction = numpy.exp(-tau)

            for i in range(len(m.images[visibilities["lam"][j]].freq)):
                m.images[visibilities["lam"][j]].image[:,:,i,:] *= extinction[i]

            # And then if sub-sampling, sum the image along the frequency axis 
            # to get back to the right size, and also Hanning smooth.

            if visibilities["subsample"][j] * visibilities["averaging"][j] > 1 \
                    or visibilities["hanning"][j]:
                recombined = numpy.empty((visibilities["image_npix"][j], \
                        visibilities["image_npix"][j], \
                        visibilities["image"][j].freq.size*\
                        visibilities["averaging"][j],1))
                for i in range(freq.size // visibilities["subsample"][j]): 
                    recombined[:,:,i,0] = m.images[visibilities["lam"][j]].\
                            image[:,:,i*visibilities["subsample"][j]:(i+1)*\
                            visibilities["subsample"][j],0].mean(axis=2)

                # Now do the Hanning smoothing.

                if visibilities["hanning"][j]:
                    hanning_window = numpy.hanning(5) / numpy.hanning(5).sum()

                    recombined = scipy.signal.fftconvolve(recombined, \
                            hanning_window.reshape((1,1,hanning_window.size,\
                            1)), axes=2, mode="same")

                # Finally, average by the binning.

                binned = numpy.empty((visibilities["image_npix"][j], \
                        visibilities["image_npix"][j], \
                        visibilities["image"][j].freq.size, 1))
                for i in range(visibilities["image"][j].freq.size): 
                    binned[:,:,i,0] = recombined[:,:,i * \
                            visibilities["averaging"][j]:(i+1)*\
                            visibilities["averaging"][j],0].mean(axis=2)

                m.images[visibilities["lam"][j]].image = binned
                m.images[visibilities["lam"][j]].freq = \
                        visibilities["image"][j].freq

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

            for ind in range(len(m.images[visibilities["lam"][j]].freq)):
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

    return m
