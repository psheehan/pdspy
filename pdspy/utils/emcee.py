from ..modeling import run_disk_model, run_flared_model
import numpy

# Define a likelihood function.

def lnlike(params, visibilities, images, spectra, parameters, plot, \
        model="disk", ncpus=1, ncpus_highmass=1, with_hyperion=False, \
        timelimit=3600, source="ObjName", nice=19, verbose=False, \
        ftcode="galario"):

    if model == "disk":
        m = run_disk_model(visibilities, images, spectra, params, \
                parameters, plot, ncpus=ncpus, ncpus_highmass=ncpus_highmass, \
                with_hyperion=with_hyperion, timelimit=timelimit, \
                source=source, nice=nice, verbose=verbose, ftcode=ftcode)
    elif model == "flared":
        m = run_flared_model(visibilities, params, parameters, plot, \
                ncpus=ncpus, source=source, nice=nice, ftcode=ftcode)

    # Catch whether the model timed out.

    if m == 0.:
        return -numpy.inf

    # A list to put all of the chisq into.

    chisq = []

    # Calculate the chisq for the visibilities.

    for j in range(len(visibilities["file"])):
        good = visibilities["data"][j].weights > 0

        chisq.append(-0.5*numpy.sum((visibilities["data"][j].real - \
                m.visibilities[visibilities["lam"][j]].real)**2 * \
                visibilities["data"][j].weights) - \
                numpy.sum(numpy.log(visibilities["data"][j].weights[good]/ \
                (2*numpy.pi))) + \
                -0.5*numpy.sum((visibilities["data"][j].imag - \
                m.visibilities[visibilities["lam"][j]].imag)**2 * \
                visibilities["data"][j].weights) - \
                numpy.sum(numpy.log(visibilities["data"][j].weights[good]/ \
                (2*numpy.pi))))


    # Calculate the chisq for all of the images.

    for j in range(len(images["file"])):
        chisq.append(-0.5 * (numpy.sum((images["data"][j].image - \
                m.images[images["lam"][j]].image)**2 / \
                images["data"][j].unc**2)))

    # Calculate the chisq for the SED.

    if "total" in spectra:
        chisq.append(-0.5 * (numpy.sum((spectra["total"].flux - \
                m.spectra["SED"].flux)**2 / spectra["total"].unc**2)))

    # Return the sum of the chisq.

    return numpy.array(chisq).sum()

# Define a prior function.

def lnprior(params, parameters, priors, visibilities):
    for key in parameters:
        if not parameters[key]["fixed"]:
            if parameters[key]["limits"][0] <= params[key] <= \
                    parameters[key]["limits"][1]:
                pass
            else:
                return -numpy.inf

    # Make sure that the radii are correct.

    if "logR_in" in params:
        R_in = 10.**params["logR_in"]
    else:
        R_in = 10.**parameters["logR_in"]["value"]

    if "logR_disk" in params:
        R_disk = 10.**params["logR_disk"]
    else:
        R_disk = 10.**parameters["logR_disk"]["value"]

    if "logR_env" in params:
        R_env = 10.**params["logR_env"]
    else:
        R_env = 10.**parameters["logR_env"]["value"]

    if R_in <= R_disk <= R_env:
        pass
    else:
        return -numpy.inf

    # Check that we aren't allowing any absurdly dense models.

    if "logR_env" in params and "logM_env" in params:
        if params["logR_env"] < 0.5 * params["logM_env"] + 4.:
            return -numpy.inf
        else:
            pass

    # Check that the cavity actually falls within the disk.

    if not parameters["logR_cav"]["fixed"]:
        if R_in <= 10.**params["logR_cav"] <= R_disk:
            pass
        else:
            return -numpy.inf

    # Check that the midplane temperature is below the atmosphere temperature.

    if ("logTmid0" in params) or ("logTmid0" in parameters):
        if "logTmid0" in params:
            Tmid0 = 10.**params["logTmid0"]
        else:
            Tmid0 = 10.**parameters["logTmid0"]["value"]

        if "logTatm0" in params:
            Tatm0 = 10.**params["logTatm0"]
        else:
            Tatm0 = 10.**parameters["logTatm0"]["value"]

        if Tmid0 < Tatm0:
            pass
        else:
            return -numpy.inf

    # Check that the gap is reasonable.

    if not parameters["logR_gap1"]["fixed"]:
        if R_in <= 10.**params["logR_gap1"] - params["w_gap1"]/2:
            pass
        else:
            return -numpy.inf

    # Everything was correct, so continue on.

    lnprior = 0.0

    # Add in the priors.

    for i in range(len(visibilities["file"])):
        if (not parameters["flux_unc{0:d}".format(i+1)]["fixed"]) and \
                (parameters["flux_unc{0:d}".format(i+1)]["prior"] == \
                "gaussian"):
            lnprior += -0.5 * (params["flux_unc{0:d}".format(i+1)] - \
                    parameters["flux_unc{0:d}".format(i+1)]["value"])**2 / \
                    parameters["flux_unc{0:d}".format(i+1)]["sigma"]**2

    # The prior on parallax (distance).

    if (not parameters["dpc"]["fixed"]) and ("parallax" in priors):
        parallax_mas = 1. / params["dpc"] * 1000

        lnprior += -0.5 * (parallax_mas - priors["parallax"]["value"])**2 / \
                priors["parallax"]["sigma"]**2
    elif (not parameters["dpc"]["fixed"]) and ("dpc" in priors):
        lnprior += -0.5 * (params["dpc"] - priors["dpc"]["value"])**2 / \
                priors["dpc"]["sigma"]**2

    # A prior on stellar mass from the IMF.

    if (not parameters["logM_star"]["fixed"]) and ("Mstar" in priors):
        Mstar = 10.**params["logM_star"]

        if priors["Mstar"]["value"] == "chabrier":
            if Mstar <= 1.:
                lnprior += numpy.log(0.158 * 1./(numpy.log(10.) * Mstar) * \
                        numpy.exp(-(numpy.log10(Mstar) - numpy.log10(0.08))**2/\
                        (2*0.69**2)))
            else:
                lnprior += numpy.log(4.43e-2 * Mstar**-1.3 * \
                        1./(numpy.log(10.) * Mstar))

    # Return

    return lnprior

# Define a probability function.

def lnprob(p, visibilities, images, spectra, parameters, priors, plot, \
        model="disk", ncpus=1, ncpus_highmass=1, with_hyperion=False, \
        timelimit=3600, source="ObjName", nice=19, verbose=False, \
        ftcode="galario"):

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    lp = lnprior(params, parameters, priors, visibilities)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(params, visibilities, images, spectra, parameters, \
            plot, model=model, ncpus=ncpus, ncpus_highmass=ncpus_highmass, \
            with_hyperion=with_hyperion, timelimit=timelimit, source=source, \
            nice=nice, verbose=verbose, ftcode=ftcode)
