from .base_parameters import base_parameters

def check_parameters(parameters):
    # Make sure the code is backwards compatible to a time when only a single 
    # gas file was being supplied.

    if "gas_file" in parameters:
        parameters["gas_file1"] = parameters["gas_file"]
        parameters["logabundance1"] = parameters["logabundance"]

    # Make sure that the envelope dust is the same as the disk dust, if it is 
    # not specified.

    if "disk_dust" in parameters and not "envelope_dust" in parameters:
        parameters["envelope_dust"] = parameters["dust_file"]

    # Now loop through and make sure all of the parameters are present and 
    # accounted for.

    for key in base_parameters:
        if not key in parameters:
            parameters[key] = base_parameters[key]

    return parameters
