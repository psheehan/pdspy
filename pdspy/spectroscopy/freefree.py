import numpy

def freefree(nu, F_nu_ff, nu_turn, pl_turn):
    if type(nu) == numpy.ndarray:
        flux = numpy.where(nu < nu_turn, F_nu_ff * (nu / nu_turn)**pl_turn, \
                F_nu_ff * (nu / nu_turn)**-0.1)

        flux = numpy.where(nu > 1.0e13, 0., flux)

        return flux
    else:
        if nu < nu_turn:
            return F_nu_ff * (nu / nu_turn)**pl_turn
        elif nu > 1.0e13:
            return 0.
        else:
            return F_nu_ff * (nu / nu_turn)**-0.1
