import numpy

def Teff_from_SpT(SpT, relation="HH14"):

    # Set up the data.

    SpT_list = ['F5','F8','G0',"G2", "G5", "G8", "K0", "K2", "K5", "K7", "M0", \
            "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    SpT_numbers = [float(test[0].replace("F","4").replace("G","3").\
            replace("K","2").replace("M","1") + str(9-float(test[1:]))) \
            for test in SpT_list]

    if relation == 'HH14':
        Teff_list = [6600, 6130, 5930, 5690, 5430, 5180, 4870, 4710, 4210, \
                4020, 3900, 3720, 3560, 3410, 3190, 2980, 2860, 2770]
    elif relation == 'PM13':
        Teff_list = [6420, 6100, 6050, 5870, 5500, 5210, 5030, 4760, 4140, \
                3970, 3770, 3630, 3490, 3360, 3160, 2880]

        SpT_list = SpT_list[0:-2]
        SpT_numbers = SpT_numbers[0:-2]

    # Now turn the provided spectral type into a number.

    SpT_number = float(SpT[0].replace("F","4").replace("G","3").\
            replace("K","2").replace("M","1") + str(9-float(SpT[1:])))

    # Finally, interpolate to a temperature.

    Teff = numpy.interp(SpT_number, SpT_numbers[::-1], Teff_list[::-1])

    return Teff
