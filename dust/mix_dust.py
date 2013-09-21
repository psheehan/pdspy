import numpy
import scipy
from .Dust import Dust
from ..constants.math import pi

def mix_dust(dust, abundance, medium=None, rule="Bruggeman"):

    if rule == "Bruggeman":
        meff = numpy.zeros(dust[0].lam.size,dtype=complex)
        rho = 0.0
        
        for i in range(dust[0].lam.size):
            temp = scipy.optimize.fsolve(bruggeman,numpy.array([1.0,0.0]),\
                    args=(dust,abundance,i))
            meff[i] = temp[0]+1j*temp[1]
        
        for i in range(len(dust)):
            rho += dust[i].rho*abundance[i]
    
    elif rule == "MaxGarn":
        sigma = 0.0+1j*0.0
        rho = 0.0
        
        for i in range(len(dust)):
            sigma += abundance[i]*(dust[i].m**2-medium.m**2)/ \
                    (dust[i].m**2+2*medium.m**2)

            rho += dust[i].rho*abundance[i]
        
        meff = numpy.sqrt(medium.m**2*(1+3*sigma/(1-sigma)))

    new = Dust()
    new.set_density(rho)
    new.set_optical_constants(dust[0].lam, meff.real, meff.imag)
    
    return new

def bruggeman(meff, dust, abundance, index):
    
    m_eff = meff[0]+1j*meff[1]
    tot = 0+0j
    
    for j in range(len(dust)):
        tot += abundance[j]*(dust[j].m[index]**2-m_eff**2)/ \
                (dust[j].m[index]**2+2*m_eff**2)
    
    return numpy.array([tot.real,tot.imag])
