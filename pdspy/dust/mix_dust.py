import numpy
import scipy
from .Dust import Dust
from ..constants.math import pi

def mix_dust(dust, abundance, medium=None, rule="Bruggeman", filling=1.):

    if rule == "Bruggeman":
        meff = numpy.zeros(dust[0].lam.size,dtype=complex)
        rho = 0.0
        
        for i in range(dust[0].lam.size):
            temp = scipy.optimize.fsolve(bruggeman,numpy.array([1.0,0.0]),\
                    args=(dust,abundance,i, filling))
            meff[i] = temp[0]+1j*temp[1]
        
        for i in range(len(dust)):
            rho += dust[i].rho*abundance[i]

        rho *= filling
    
    elif rule == "MaxGarn":
        numerator = 0.0+1j*0.0
        denominator = 0.0+1j*0.0
        rho = 0.0
        
        for i in range(len(dust)):
            gamma = 3. / (dust[i].m**2 + 2)

            numerator += abundance[i] * gamma * dust[i].m**2
            denominator += abundance[i] * gamma

            rho += dust[i].rho*abundance[i]

        mmix = numpy.sqrt(numerator / denominator)
        
        F = (mmix**2 - 1.) / (mmix**2 + 2.)

        meff = numpy.sqrt((1. + 2.*filling*F) / (1. - filling*F))

        rho *= filling

    new = Dust()
    new.set_density(rho)
    new.set_optical_constants(dust[0].lam, meff.real, meff.imag)
    
    return new

def bruggeman(meff, dust, abundance, index, filling):
    
    m_eff = meff[0]+1j*meff[1]
    tot = 0+0j
    
    for j in range(len(dust)):
        tot += filling * abundance[j]*(dust[j].m[index]**2-m_eff**2)/ \
                (dust[j].m[index]**2+2*m_eff**2)

    # Add in the void.

    tot += (1 - filling) * (1. - m_eff**2) / (1. + 2*m_eff**2)
    
    return numpy.array([tot.real,tot.imag])
