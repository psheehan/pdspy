import numpy

def propose_point_emcee(parameters, model="disk"):
    m_env = numpy.random.uniform(-6., parameters["logM_env"]["limits"][1],1)[0]

    # Set up R_env, R_disk, and R_in as they depend on eachother.

    if model == "disk":
        r_env = numpy.random.uniform(max(parameters["logR_env"]["limits"][0], \
                0.5*m_env+4.), parameters["logR_env"]["limits"][1],1)[0]
    elif model == "flared":
        r_env = numpy.random.uniform(parameters["logR_env"]["limits"][0],\
                parameters["logR_env"]["limits"][1],1)[0]

    r_disk = numpy.random.uniform(max(numpy.log10(5.), \
            parameters["logR_disk"]["limits"][0]), min(numpy.log10(500.), \
            r_env, parameters["logR_disk"]["limits"][1]),1)[0]
    r_in = numpy.random.uniform(parameters["logR_in"]["limits"][0],\
            min(parameters["logR_in"]["limits"][1], \
            numpy.log10((10.**r_disk)/2)),1)[0]

    # Also set up R_cav as it depends on those as well.

    r_cav = numpy.random.uniform(max(r_in,parameters["logR_cav"]["limits"][0]),\
            min(numpy.log10(0.75*10.**r_disk),\
            parameters["logR_cav"]["limits"][1]),1)[0]

    # Same thing for R_gap and w_gap.

    r_gap1 = numpy.random.uniform(numpy.log10(10.**r_in+\
            parameters["w_gap1"]["limits"][0]/2), \
            numpy.log10(0.75*10.**r_disk),1)[0]

    w_gap1 = numpy.random.uniform(parameters["w_gap1"]["limits"][0],\
            min(parameters["w_gap1"]["limits"][1],\
            2*(10.**r_gap1-10.**r_in)), 1)[0]

    # If we are using logTatm0 and logTmid0, they depend on eachother.

    if "logTatm0" in parameters:
        tatm0 = numpy.random.uniform(parameters["logTatm0"]\
                ["limits"][0],parameters["logTatm0"]["limits"][1],1)[0]
        tmid0 = numpy.random.uniform(parameters["logTmid0"]["limits"][0],\
                min(parameters["logTatm0"]["limits"][1], tatm0),1)[0]

    # Loop through and generate the point proposal.

    p = []

    for key in sorted(parameters.keys()):
        if parameters[key]["fixed"]:
            pass
        elif key == "logR_in":
            p.append(r_in)
        elif key == "logR_disk":
            p.append(r_disk)
        elif key == "logR_env":
            p.append(r_env)
        elif key == "logR_cav":
            p.append(r_cav)
        elif key == "logR_gap1":
            p.append(r_gap1)
        elif key == "w_gap1":
            p.append(w_gap1)
        elif key == "logM_disk" and model == "disk":
            p.append(numpy.random.uniform(-6.,parameters[key]["limits"][1], \
                    1)[0])
        elif key == "logM_env":
            p.append(m_env)
        elif key == "h_0":
            p.append(numpy.random.uniform(parameters[key]["limits"][0], \
                    0.2, 1)[0])
        elif key == "logTatm0":
            p.append(tatm0)
        elif key == "logTmid0":
            p.append(tmid0)
        elif key[0:8] == "flux_unc":
            p.append(numpy.random.normal(parameters[key]["value"], 0.001, 1)[0])
        else:
            p.append(numpy.random.uniform(parameters[key]["limits"][0], \
                    parameters[key]["limits"][1], 1)[0])

    # Return the proposed point.

    return p
