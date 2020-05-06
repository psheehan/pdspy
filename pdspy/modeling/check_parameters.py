from .base_parameters import base_parameters

def check_parameters(parameters, nvis=3):
    # Make sure the code is backwards compatible to a time when only a single 
    # gas file was being supplied.

    if "gas_file" in parameters:
        parameters["gas_file1"] = parameters["gas_file"]
        parameters["logabundance1"] = parameters["logabundance"]

    # Make sure that the envelope dust is the same as the disk dust, if it is 
    # not specified.

    if "dust_file" in parameters and not "envelope_dust" in parameters:
        parameters["envelope_dust"] = parameters["dust_file"]

    # Now loop through and make sure all of the parameters are present and 
    # accounted for.

    for key in base_parameters:
        if not key in parameters:
            parameters[key] = base_parameters[key]

    # Make sure there are enough instances of freezeout for all of the gas 
    # files provided.

    index = 1
    while index > 0:
        if "gas_file"+str(index) in parameters:
            if not "logabundance{0:d}".format(index) in parameters:
                parameters["logabundance"+str(index)] = \
                        base_parameters["logabundance1"]
            if not "freezeout{0:d}".format(index) in parameters:
                parameters["freezeout"+str(index)] = \
                        base_parameters["freezeout1"]

            index += 1
        else:
            index = -1

    # Make sure there are enough instances of flux_unc for all of the 
    # data files that are being fit.

    for i in range(nvis):
        if not "flux_unc{0:d}".format(i+1) in parameters:
            parameters["flux_unc{0:d}".format(i+1)] = parameters["flux_unc1"]

    return parameters
